import math
import re
from collections import OrderedDict
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
                    "description": desc,
                    "score": round(float(best), 3),
                })
                surfaced_task_ids.add(task_id)

    # Always surface question-relevant tasks (Direction A), regardless of whether
    # we had seed spans. This makes tasks-only workspaces answerable.
    for sim, idx in relevant_tasks:
        task_id, title, desc, status, due_date, note_id = task_rows[idx]
        if any(tm["task_id"] == str(task_id) for tm in task_mentions):
            continue
        task_mentions.append({
            "task_id": str(task_id),
            "title": title,
            "status": status,
            "due_date": due_date.isoformat() if due_date else None,
            "description": desc,
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

_TASK_INTENT_RE = re.compile(
    r"\b("
    r"task|tasks|todo|todos|to-do|to do|"
    r"what'?s left|whats left|anything left|left to do|"
    r"on my list|my list|to do list|"
    r"open items|action items|"
    r"in progress|in-progress|still need to do|"
    r"finished|completed|done\b|to be done"
    r")\b",
    re.IGNORECASE,
)


def _is_task_question(text: str) -> bool:
    return bool(_TASK_INTENT_RE.search(text or ""))


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

    direct_hits = [
        {
            "span_id": str(r[0]),
            "text": r[1],
            "note_id": str(r[2]),
            "note_title": r[3] or "Untitled",
            "score": max(0.0, 1.0 - float(r[4] or 0)),
            "via_graph": False,
            "via_task": False,
            "relation": None,
        }
        for r in rows
    ]
    already = {h["span_id"] for h in direct_hits}

    remaining_budget = min(GRAPH_EXPANSION_CAP, TOTAL_SPAN_CAP - len(direct_hits))
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

    task_budget = min(TASK_BRIDGE_CAP, TOTAL_SPAN_CAP - len(direct_hits) - len(extras))
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

    # Group all retrieved spans by note (preserving rank order)
    note_groups: "OrderedDict[str, dict]" = OrderedDict()

    def _add(span_id, text, note_id, note_title, score, via_graph, via_task, relation):
        bucket = note_groups.get(note_id)
        if bucket is None:
            bucket = {
                "note_id": note_id,
                "note_title": note_title or "Untitled",
                "span_ids": [],
                "snippets": [],
                "score": 0.0,
                "via_graph": False,
                "via_task": False,
                "relations": [],
            }
            note_groups[note_id] = bucket
        bucket["span_ids"].append(span_id)
        bucket["snippets"].append((text or "").strip())
        if score > bucket["score"]:
            bucket["score"] = score
        if via_graph:
            bucket["via_graph"] = True
        if via_task:
            bucket["via_task"] = True
        if relation and relation not in bucket["relations"]:
            bucket["relations"].append(relation)

    for h in direct_hits:
        _add(h["span_id"], h["text"], h["note_id"], h["note_title"],
             h["score"], False, False, None)
    for ex in extras:
        _add(ex["sid"], ex["text"], str(ex["note_id"]), ex["note_title"],
             0.0, True, False, ex.get("relation"))
    for ex in task_extras:
        _add(ex["sid"], ex["text"], str(ex["note_id"]), ex["note_title"],
             0.0, False, True, ex.get("relation"))

    task_intent = _is_task_question(body.question)

    # Build task context entries (used in two cases: task-intent in the question,
    # or a workspace with no notes at all).
    def _task_to_entry(title, desc, status, due_date):
        lines = [
            f"Task: {title}",
            f"Status: {status or 'unknown'}",
        ]
        if due_date:
            lines.append(f"Due: {due_date.isoformat()}")
        if desc:
            d = (desc or "").strip()
            if len(d) > 1000:
                d = d[:1000] + "…"
            lines.append(f"Description: {d}")
        return "\n".join(lines)

    task_entries: list[str] = []
    all_tasks_rows: list = []
    if task_intent or not note_groups:
        all_tasks_rows = (await db.execute(
            select(Task.id, Task.title, Task.description, Task.status, Task.due_date)
            .where(Task.workspace_id == workspace_id)
            .limit(20)
        )).all()

        # Order: question-relevant tasks first (already in task_mentions), then the rest.
        seen_in_mentions = [tm["task_id"] for tm in task_mentions]
        task_lookup = {str(t[0]): t for t in all_tasks_rows}
        ordered_ids: list[str] = []
        for tid in seen_in_mentions:
            if tid in task_lookup and tid not in ordered_ids:
                ordered_ids.append(tid)
        for tid in task_lookup:
            if tid not in ordered_ids:
                ordered_ids.append(tid)

        for tid in ordered_ids[:12]:
            _id, title, desc, status, due_date = task_lookup[tid]
            task_entries.append(_task_to_entry(title, desc, status, due_date))

        # Surface every task so the UI gets chips, not just the question-relevant ones.
        existing_ids = {tm["task_id"] for tm in task_mentions}
        for tid, title, desc, status, due_date in all_tasks_rows:
            tid_str = str(tid)
            if tid_str in existing_ids:
                continue
            task_mentions.append({
                "task_id": tid_str,
                "title": title,
                "status": status,
                "due_date": due_date.isoformat() if due_date else None,
                "description": desc,
                "score": 0.0,
            })

    notes_payload: list[dict] = []
    n_token_to_bucket: dict[str, dict] = {}
    counter = 0

    # When the user is asking about tasks, tasks become the PRIMARY context (N1, N2, …)
    # and note buckets follow. This stops the LLM from answering with note prose
    # when the user wanted a task list.
    if task_intent and task_entries:
        for entry in task_entries:
            counter += 1
            notes_payload.append({"id": f"N{counter}", "text": entry})
        for bucket in note_groups.values():
            counter += 1
            n_token = f"N{counter}"
            n_token_to_bucket[n_token] = bucket
            title = bucket["note_title"]
            body_text = "\n— — —\n".join(bucket["snippets"])
            if len(body_text) > 4000:
                body_text = body_text[:4000] + "…"
            notes_payload.append({
                "id": n_token,
                "text": f"From note titled \"{title}\":\n{body_text}",
            })
    else:
        # Default: notes first
        for bucket in note_groups.values():
            counter += 1
            n_token = f"N{counter}"
            n_token_to_bucket[n_token] = bucket
            title = bucket["note_title"]
            body_text = "\n— — —\n".join(bucket["snippets"])
            if len(body_text) > 4000:
                body_text = body_text[:4000] + "…"
            notes_payload.append({
                "id": n_token,
                "text": f"From note titled \"{title}\":\n{body_text}",
            })
        # Tasks-only fallback: no notes at all but workspace has tasks
        if not notes_payload and task_entries:
            for entry in task_entries:
                counter += 1
                notes_payload.append({"id": f"N{counter}", "text": entry})

    if not notes_payload:
        return {
            "answer": "I don't have any indexed notes or tasks yet to answer from. Try adding some notes or tasks first and wait for them to finish processing.",
            "citations": [],
            "sources": [],
            "tasks": [],
        }

    # Dedupe task chips for UI rendering
    if task_mentions:
        seen = set()
        deduped = []
        for tm in task_mentions:
            if tm["task_id"] in seen:
                continue
            seen.add(tm["task_id"])
            deduped.append(tm)
        task_mentions = deduped[:8]

    # Only append the task-bullet hint to the question when tasks AREN'T already
    # in the primary context. With task_intent, they're already [N#] blocks —
    # appending bullets would be redundant noise.
    augmented_question = body.question
    if task_mentions and not task_intent and not task_entries:
        bullets = "\n".join(
            f"- {tm['title']} (status: {tm['status']}"
            + (f", due: {tm['due_date']}" if tm.get("due_date") else "")
            + ")"
            for tm in task_mentions
        )
        augmented_question = (
            f"{body.question}\n\n"
            f"Tasks in this workspace that may relate:\n{bullets}"
        )

    try:
        result = await chat_remote(
            question=augmented_question,
            spans=notes_payload,
            history=[t.model_dump() for t in body.history],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"chat failed: {e}")

    raw_answer = result.get("answer", "") or ""
    cited_tokens = list(result.get("citations") or [])

    # Fallback: if the LLM emitted no [N#] tags but produced an answer,
    # surface the highest-scored note groups so the UI still has chips.
    if not cited_tokens and notes_payload and raw_answer.strip():
        cited_tokens = [n["id"] for n in notes_payload[:3]]

    # Build sources at note granularity, in citation order
    sources = []
    seen_notes = set()
    for tok in cited_tokens:
        bucket = n_token_to_bucket.get(tok)
        if not bucket or bucket["note_id"] in seen_notes:
            continue
        seen_notes.add(bucket["note_id"])
        preview = " ".join(bucket["snippets"]).replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:220] + "…"
        sources.append({
            "note_id": bucket["note_id"],
            "note_title": bucket["note_title"],
            "score": round(float(bucket["score"]), 3),
            "preview": preview,
            "via_graph": bucket["via_graph"],
            "via_task": bucket["via_task"],
            "relations": bucket["relations"],
            # legacy fields for any older client code:
            "span_id": bucket["span_ids"][0] if bucket["span_ids"] else bucket["note_id"],
            "relation": bucket["relations"][0] if bucket["relations"] else None,
        })

    # Strip the [N#] inline tags from the answer for clean display.
    # Citation chips below the bubble already surface the source notes.
    clean_answer = re.sub(r"\s*\[N\d+\]", "", raw_answer).strip()
    # Also remove any stray double-spaces left behind.
    clean_answer = re.sub(r"[ \t]{2,}", " ", clean_answer)

    return {
        "answer": clean_answer,
        "citations": cited_tokens,
        "sources": sources,
        "model": result.get("model"),
        "graph_expanded": len(extras),
        "task_extras": len(task_extras),
        "tasks": task_mentions,
    }
