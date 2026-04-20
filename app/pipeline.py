"""
Background orchestrator: pulls note text → calls the ML HTTP service →
persists spans/objects/links/insights → drives note.status through the
lifecycle (created → chunked → embedded → structured → ready | error).
"""
from uuid import UUID, uuid4
from typing import Dict

from sqlalchemy import select, delete

from app.database import async_session
from app.models import Note, Span, Object, ObjectMention, Link, Insight
from app.ml_client import process_note_remote


async def process_note(note_id: UUID) -> None:
    """Entrypoint for BackgroundTasks. Opens its own DB session so it can
    outlive the request that scheduled it."""
    async with async_session() as db:
        note = (await db.execute(select(Note).where(Note.id == note_id))).scalar_one_or_none()
        if note is None:
            return

        text = (note.cleaned_text or note.raw_text or "").strip()
        if not text:
            note.status = "ready"
            await db.commit()
            return

        try:
            # Wipe any prior pipeline artifacts for this note (idempotent re-runs)
            await db.execute(delete(Span).where(Span.note_id == note_id))
            await db.commit()

            result = await process_note_remote(
                text=text,
                note_id=note_id,
                workspace_id=note.workspace_id,
            )

            # 1. Spans + embeddings
            spans_payload = result.get("spans") or []
            vectors = result.get("embeddings") or []
            span_rows: list[Span] = []
            for i, s in enumerate(spans_payload):
                row = Span(
                    note_id=note_id,
                    chunk_index=s["chunk_index"],
                    start_char=s["start_char"],
                    end_char=s["end_char"],
                    text=s["text"],
                    token_count=s.get("token_count", 0),
                    embedding=vectors[i] if i < len(vectors) else None,
                )
                span_rows.append(row)
                db.add(row)
            await db.flush()
            note.status = "chunked"
            await db.commit()
            if vectors:
                note.status = "embedded"
                await db.commit()

            # 2. Objects (remap ML temp ids → UUIDs) + mentions + links
            objects_payload = result.get("objects") or []
            links_payload = result.get("links") or []
            mentions_payload = result.get("mentions") or []
            temp_to_uuid: Dict[str, UUID] = {}

            for o in objects_payload:
                new_id = uuid4()
                temp_to_uuid[o["id"]] = new_id
                db.add(Object(
                    id=new_id,
                    workspace_id=note.workspace_id,
                    type=o["type"],
                    canonical_text=o["canonical_text"],
                    confidence=o.get("confidence"),
                    status="active",
                ))
            await db.flush()

            first_span = span_rows[0].id if span_rows else None

            # Mentions: prefer explicit ones from ML; fall back to first-span pin.
            if mentions_payload and first_span is not None:
                for m in mentions_payload:
                    obj_uuid = temp_to_uuid.get(m.get("object_id"))
                    if obj_uuid is None:
                        continue
                    db.add(ObjectMention(
                        object_id=obj_uuid,
                        span_id=first_span,
                        note_id=note_id,
                        role="primary",
                        confidence=None,
                    ))
            elif first_span is not None:
                for o in objects_payload:
                    db.add(ObjectMention(
                        object_id=temp_to_uuid[o["id"]],
                        span_id=first_span,
                        note_id=note_id,
                        role="primary",
                        confidence=o.get("confidence"),
                    ))

            for l in links_payload:
                src = temp_to_uuid.get(l["source_id"])
                dst = temp_to_uuid.get(l["target_id"])
                if not (src and dst):
                    continue
                db.add(Link(
                    workspace_id=note.workspace_id,
                    src_object_id=src,
                    dst_object_id=dst,
                    type=l["type"],
                    confidence=l.get("confidence"),
                    evidence_span_id=first_span,
                ))

            if objects_payload:
                note.status = "structured"
                await db.commit()

            # 3. Insights
            contradictions = result.get("contradictions") or []
            stale = result.get("stale_threads") or []
            for c in contradictions:
                if not isinstance(c, dict):
                    continue
                db.add(Insight(
                    workspace_id=note.workspace_id,
                    type="contradiction",
                    severity=(c.get("severity") or "high").lower(),
                    status="new",
                    payload={
                        "source_text": c.get("source_text") or c.get("claim1_text"),
                        "target_text": c.get("target_text") or c.get("claim2_text"),
                    },
                ))
            for s in stale:
                db.add(Insight(
                    workspace_id=note.workspace_id,
                    type="stale_thread",
                    severity=s.get("severity", "medium"),
                    status="new",
                    payload=s.get("payload") or s,
                ))

            note.status = "ready"
            await db.commit()

        except Exception as e:
            await db.rollback()
            try:
                note.status = "error"
                note.error_message = str(e)[:500]
                await db.commit()
            except Exception:
                await db.rollback()
            print(f"[pipeline] error processing note {note_id}: {e}")
