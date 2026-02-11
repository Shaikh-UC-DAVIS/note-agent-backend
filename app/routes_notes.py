import hashlib
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User, Workspace, Note
from app.schemas import NoteCreate, NoteUpdate, NoteOut, WorkspaceCreate, WorkspaceOut
from app.auth import get_current_user

router = APIRouter(tags=["notes"])


# ── Helpers ───────────────────────────────────────────
async def _get_workspace(
    workspace_id: UUID, user: User, db: AsyncSession
) -> Workspace:
    result = await db.execute(
        select(Workspace).where(
            Workspace.id == workspace_id,
            Workspace.owner_user_id == user.id,
        )
    )
    ws = result.scalar_one_or_none()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws


# ── Workspaces ────────────────────────────────────────
@router.get("/workspaces", response_model=list[WorkspaceOut])
async def list_workspaces(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Workspace).where(Workspace.owner_user_id == user.id).order_by(Workspace.created_at)
    )
    return result.scalars().all()


@router.post("/workspaces", response_model=WorkspaceOut, status_code=201)
async def create_workspace(
    body: WorkspaceCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = Workspace(owner_user_id=user.id, name=body.name)
    db.add(ws)
    await db.commit()
    await db.refresh(ws)
    return ws


@router.patch("/workspaces/{workspace_id}", response_model=WorkspaceOut)
async def update_workspace(
    workspace_id: UUID,
    body: WorkspaceCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ws = await _get_workspace(workspace_id, user, db)
    ws.name = body.name
    await db.commit()
    await db.refresh(ws)
    return ws


@router.delete("/workspaces/{workspace_id}", status_code=204)
async def delete_workspace(
    workspace_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Prevent deleting the last workspace
    result = await db.execute(
        select(func.count()).select_from(Workspace).where(Workspace.owner_user_id == user.id)
    )
    count = result.scalar()
    if count <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete your only workspace")
    ws = await _get_workspace(workspace_id, user, db)
    await db.delete(ws)
    await db.commit()


# ── Notes CRUD ────────────────────────────────────────
@router.get("/workspaces/{workspace_id}/notes", response_model=list[NoteOut])
async def list_notes(
    workspace_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_workspace(workspace_id, user, db)
    result = await db.execute(
        select(Note)
        .where(Note.workspace_id == workspace_id)
        .order_by(Note.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return result.scalars().all()


@router.post("/workspaces/{workspace_id}/notes", response_model=NoteOut, status_code=201)
async def create_note(
    workspace_id: UUID,
    body: NoteCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_workspace(workspace_id, user, db)
    content_hash = hashlib.sha256(body.raw_text.encode()).hexdigest()
    note = Note(
        workspace_id=workspace_id,
        title=body.title,
        raw_text=body.raw_text,
        content_hash=content_hash,
    )
    db.add(note)
    await db.commit()
    await db.refresh(note)
    return note


@router.get("/workspaces/{workspace_id}/notes/{note_id}", response_model=NoteOut)
async def get_note(
    workspace_id: UUID,
    note_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_workspace(workspace_id, user, db)
    result = await db.execute(
        select(Note).where(Note.id == note_id, Note.workspace_id == workspace_id)
    )
    note = result.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.patch("/workspaces/{workspace_id}/notes/{note_id}", response_model=NoteOut)
async def update_note(
    workspace_id: UUID,
    note_id: UUID,
    body: NoteUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_workspace(workspace_id, user, db)
    result = await db.execute(
        select(Note).where(Note.id == note_id, Note.workspace_id == workspace_id)
    )
    note = result.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    if body.title is not None:
        note.title = body.title
    if body.raw_text is not None:
        note.raw_text = body.raw_text
        note.content_hash = hashlib.sha256(body.raw_text.encode()).hexdigest()

    await db.commit()
    await db.refresh(note)
    return note


@router.delete("/workspaces/{workspace_id}/notes/{note_id}", status_code=204)
async def delete_note(
    workspace_id: UUID,
    note_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_workspace(workspace_id, user, db)
    result = await db.execute(
        select(Note).where(Note.id == note_id, Note.workspace_id == workspace_id)
    )
    note = result.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    await db.delete(note)
    await db.commit()
