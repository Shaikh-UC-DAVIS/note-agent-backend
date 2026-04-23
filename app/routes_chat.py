from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Note, Span, User, Workspace
from app.auth import get_current_user
from app.ml_client import chat_remote, embed_remote

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
    id_to_meta = {
        str(r[0]): {
            "note_id": str(r[2]),
            "note_title": r[3],
            "score": max(0.0, 1.0 - float(r[4] or 0)),
            "text": r[1],
        }
        for r in rows
    }

    try:
        result = await chat_remote(
            question=body.question,
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
        })

    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations") or [],
        "sources": sources,
        "model": result.get("model"),
    }
