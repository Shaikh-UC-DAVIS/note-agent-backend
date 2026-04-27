import math
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Link, Note, Object, ObjectMention, Span, Task, User, Workspace
from app.auth import get_current_user
from app.ml_client import chat_remote, embed_remote


GRAPH_EXPANSION_CAP = 6
TASK_BRIDGE_CAP = 6
TOTAL_SPAN_CAP = 20
TASK_QUESTION_SIM_THRESHOLD = 0.30
TASK_SPAN_SIM_THRESHOLD = 0.35


def _cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


async def _task_bridge(
    db: AsyncSession,
    workspace_id: UUID,
    question: str,
    qvec: list[float],
    seed_span_rows: list,
    already_in_context: set[str],
    budget: int,
) -> tuple[list[dict], list[dict]]:
    """Bidirectional task<->note bridge using embeddings only — no explicit linkage required.

    Returns (extra_spans, task_mentions):
      - extra_spans: spans pulled from notes that match a relevant task semantically
      - task_mentions: task records that should be surfaced because they relate to seed spans
    """
    if budget <= 0:
        return [], []

    task_rows = (await db.execute(
        select(Task.id, Task.title, Task.description, Task.status, Task.due_date, Task.note_id)
        .where(Task.workspace_id == workspace_id)
    )).all()
    if not task_rows:
        return [], []

    task_texts = [
        f"{(r[1] or '').strip()}. {(r[2] or '').strip()}".strip(". ").strip() or (r[1] or "")
        for r in task_rows
    ]
    try:
        emb = await embed_remote(task_texts)
        task_vecs = emb.get("vectors") or []
    except Exception:
        task_vecs = []
    if len(task_vecs) != len(task_rows):
        return [], []

    # Direction A: question → task → linked-or-similar notes' spans
    relevant_tasks: list[tuple[float, int]] = []
    for i, vec in enumerate(task_vecs):
        sim = _cosine(qvec, vec)
        if sim >= TASK_QUESTION_SIM_THRESHOLD:
            relevant_tasks.append((sim, i))
    relevant_tasks.sort(reverse=True)
    relevant_tasks = relevant_tasks[:3]

    extra_spans: list[dict] = []
    surfaced_task_ids: set[UUID] = set()

    for sim, idx in relevant_tasks:
        task_id, title, desc, status, due_date, note_id = task_rows[idx]
        surfaced_task_ids.add(task_id)
        task_vec = task_vecs[idx]

        span_q = (
            select(
                Span.id,
                Span.text,
                Span.note_id,
                Note.title,
                Span.embedding.cosine_distance(task_vec).label("dist"),
            )
            .join(Note, Note.id == Span.note_id)
            .where(Note.workspace_id == workspace_id, Span.embedding.is_not(None))
            .order_by(Span.embedding.cosine_distance(task_vec))
            .limit(3)
        )
        rows = (await db.execute(span_q)).all()
        for span_id, text, n_id, n_title, dist in rows:
            sid = str(span_id)
            if sid in already_in_context:
                continue
            extra_spans.append({
                "span_id": span_id,
                "sid": sid,
                "text": text,
                "note_id": n_id,
                "note_title": n_title,
                "relation": f"task '{title}' ({status})",
            })
            already_in_context.add(sid)
            if len(extra_spans) >= budget:
                break
        if len(extra_spans) >= budget:
            break

    # Direction B: seed spans → tasks (surface tasks whose embedding matches any seed span)
    task_mentions: list[dict] = []
    if seed_span_rows:
        seed_vecs = []
        for row in seed_span_rows:
            seed_id = row[0]
            res = (await db.execute(select(Span.embedding).where(Span.id == seed_id))).first()
            if res and res[0] is not None:
                seed_vecs.append(list(res[0]))
        for i, vec in enumerate(task_vecs):
            task_id = task_rows[i][0]
            if task_id in surfaced_task_ids:
                continue
            best = max((_cosine(sv, vec) for sv in seed_vecs), default=0.0)
            if best >= TASK_SPAN_SIM_THRESHOLD:
                task_id, title, desc, status, due_date, note_id = task_rows[i]
                task_mentions.append({
                    "task_id": str(task_id),
                    "title": title,
                    "status": status,
                    "due_date": due_date.isoformat() if due_date else None,
                    "score": round(float(best), 3),
                })
                surfaced_task_ids.add(task_id)

        for sim, idx in relevant_tasks:
            task_id, title, desc, status, due_date, note_id = task_rows[idx]
            task_mentions.append({
                "task_id": str(task_id),
                "title": title,
                "status": status,
                "due_date": due_date.isoformat() if due_date else None,
                "score": round(float(sim), 3),
            })

    return extra_spans, task_mentions


async def _graph_expand_spans(
    db: AsyncSession,
    workspace_id: UUID,
    seed_span_ids: list[UUID],
    already_in_context: set[str],
    budget: int,
) -> list[dict]:
    """Walk 1 hop across links from objects mentioned in seed spans, return extra spans.

    Each returned dict has the same shape as the direct-hit spans plus a
    `relation` string (e.g. "Contradicts Claim") so the LLM can reason about
    structural relationships, not just semantic similarity.
    """
    if budget <= 0 or not seed_span_ids:
        return []

    seed_obj_rows = (await db.execute(
        select(ObjectMention.object_id)
        .join(Object, Object.id == ObjectMention.object_id)
        .where(
            ObjectMention.span_id.in_(seed_span_ids),
            Object.workspace_id == workspace_id,
        )
        .distinct()
    )).all()
    seed_object_ids = [r[0] for r in seed_obj_rows]
    if not seed_object_ids:
        return []

    link_rows = (await db.execute(
        select(Link.src_object_id, Link.dst_object_id, Link.type, Link.confidence)
        .where(
            Link.workspace_id == workspace_id,
            or_(
                Link.src_object_id.in_(seed_object_ids),
                Link.dst_object_id.in_(seed_object_ids),
            ),
        )
    )).all()

    seed_set = set(seed_object_ids)
    neighbors: dict[UUID, tuple[str, float]] = {}
    for src, dst, ltype, conf in link_rows:
        other = dst if src in seed_set else src
        if other in seed_set:
            continue
        prev = neighbors.get(other)
        c = float(conf or 0.0)
        if prev is None or c > prev[1]:
            neighbors[other] = (str(ltype), c)

    if not neighbors:
        return []

    neighbor_ids = list(neighbors.keys())

    mention_rows = (await db.execute(
        select(
            ObjectMention.object_id,
            Span.id,
            Span.text,
            Span.note_id,
            Note.title,
            Object.type,
            Object.canonical_text,
            ObjectMention.confidence,
        )
        .join(Span, Span.id == ObjectMention.span_id)
        .join(Note, Note.id == Span.note_id)
        .join(Object, Object.id == ObjectMention.object_id)
        .where(
            ObjectMention.object_id.in_(neighbor_ids),
            Note.workspace_id == workspace_id,
        )
        .order_by(ObjectMention.confidence.desc().nullslast())
    )).all()

    picked: dict[UUID, dict] = {}
    for obj_id, span_id, text, note_id, note_title, obj_type, canonical, _conf in mention_rows:
        if obj_id in picked:
            continue
        sid = str(span_id)
        if sid in already_in_context:
            continue
        link_type, _ = neighbors[obj_id]
        picked[obj_id] = {
            "span_id": span_id,
            "sid": sid,
            "text": text,
            "note_id": note_id,
            "note_title": note_title,
            "relation": f"{link_type} → {obj_type}: {canonical}",
        }
        if len(picked) >= budget:
            break

    return list(picked.values())

router = APIRouter(tags=["chat"])


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = []
    top_k: int = Field(default=6, ge=1, le=20)


@router.post("/workspaces/{workspace_id}/chat")
async def chat(
    workspace_id: UUID,
    body: ChatBody,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = (await db.execute(
        select(Workspace).where(
            Workspace.id == workspace_id,
            Workspace.owner_user_id == user.id,
        )
    )).scalar_one_or_none()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    try:
        emb = await embed_remote([body.question])
        qvec = emb["vectors"][0] if emb.get("vectors") else None
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"embed failed: {e}")
    if not qvec:
        raise HTTPException(status_code=502, detail="empty embedding")

    rows = (await db.execute(
        select(
            Span.id,
            Span.text,
            Span.note_id,
            Note.title,
            Span.embedding.cosine_distance(qvec).label("dist"),
        )
        .join(Note, Note.id == Span.note_id)
        .where(Note.workspace_id == workspace_id, Span.embedding.is_not(None))
        .order_by(Span.embedding.cosine_distance(qvec))
        .limit(body.top_k)
    )).all()

    if not rows:
        return {
            "answer": "I don't have any indexed notes yet to answer from. Try adding some notes first and wait for them to finish processing.",
            "citations": [],
            "sources": [],
        }

    spans_payload = [{"id": str(r[0]), "text": r[1]} for r in rows]
    id_to_meta: dict[str, dict] = {
        str(r[0]): {
            "note_id": str(r[2]),
            "note_title": r[3],
            "score": max(0.0, 1.0 - float(r[4] or 0)),
            "text": r[1],
            "via_graph": False,
            "via_task": False,
            "relation": None,
        }
        for r in rows
    }

    already = set(id_to_meta.keys())
    remaining_budget = min(GRAPH_EXPANSION_CAP, TOTAL_SPAN_CAP - len(spans_payload))
    try:
        extras = await _graph_expand_spans(
            db=db,
            workspace_id=workspace_id,
            seed_span_ids=[r[0] for r in rows],
            already_in_context=already,
            budget=remaining_budget,
        )
    except Exception:
        extras = []

    for ex in extras:
        sid = ex["sid"]
        annotated = f"[graph-related via {ex['relation']}] {ex['text']}"
        spans_payload.append({"id": sid, "text": annotated})
        id_to_meta[sid] = {
            "note_id": str(ex["note_id"]),
            "note_title": ex["note_title"],
            "score": 0.0,
            "text": ex["text"],
            "via_graph": True,
            "via_task": False,
            "relation": ex["relation"],
        }

    task_budget = min(TASK_BRIDGE_CAP, TOTAL_SPAN_CAP - len(spans_payload))
    try:
        task_extras, task_mentions = await _task_bridge(
            db=db,
            workspace_id=workspace_id,
            question=body.question,
            qvec=qvec,
            seed_span_rows=rows,
            already_in_context=already,
            budget=task_budget,
        )
    except Exception:
        task_extras, task_mentions = [], []

    for ex in task_extras:
        sid = ex["sid"]
        annotated = f"[related to {ex['relation']}] {ex['text']}"
        spans_payload.append({"id": sid, "text": annotated})
        id_to_meta[sid] = {
            "note_id": str(ex["note_id"]),
            "note_title": ex["note_title"],
            "score": 0.0,
            "text": ex["text"],
            "via_graph": False,
            "via_task": True,
            "relation": ex["relation"],
        }

    augmented_question = body.question
    if task_mentions:
        seen = set()
        deduped = []
        for tm in task_mentions:
            if tm["task_id"] in seen:
                continue
            seen.add(tm["task_id"])
            deduped.append(tm)
        task_mentions = deduped[:6]
        bullets = "\n".join(
            f"- {tm['title']} (status: {tm['status']}"
            + (f", due: {tm['due_date']}" if tm.get("due_date") else "")
            + ")"
            for tm in task_mentions
        )
        augmented_question = (
            f"{body.question}\n\n"
            f"Known tasks in this workspace that relate to the question or its retrieved notes:\n{bullets}\n\n"
            "When the question is about a task, mention any matching notes you cite. "
            "When the question is about notes, mention any related task and its status."
        )

    try:
        result = await chat_remote(
            question=augmented_question,
            spans=spans_payload,
            history=[t.model_dump() for t in body.history],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"chat failed: {e}")

    sources = []
    for sid in (result.get("citations") or []):
        meta = id_to_meta.get(sid)
        if not meta:
            continue
        preview = (meta["text"] or "").strip().replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "…"
        sources.append({
            "span_id": sid,
            "note_id": meta["note_id"],
            "note_title": meta["note_title"],
            "score": meta["score"],
            "preview": preview,
            "via_graph": meta["via_graph"],
            "relation": meta["relation"],
        })

    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations") or [],
        "sources": sources,
        "model": result.get("model"),
        "graph_expanded": len(extras),
        "task_extras": len(task_extras),
        "tasks": task_mentions,
    }
