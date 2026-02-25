# app/crud/tasks.py
from __future__ import annotations

from datetime import date
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Task
from app.schemas import TaskCreate, TaskUpdate


async def create_task(db: AsyncSession, *, user_id: UUID, payload: TaskCreate) -> Task:
    data = payload.model_dump(exclude_unset=True)

    # never trust client-provided ownership
    data.pop("user_id", None)
    data["user_id"] = user_id

    task = Task(**data)
    db.add(task)
    await db.commit()
    await db.refresh(task)
    return task


async def get_task_or_none(
    db: AsyncSession,
    *,
    task_id: UUID,
    user_id: UUID,
    workspace_id: UUID | None = None,
) -> Task | None:
    stmt = select(Task).where(Task.id == task_id, Task.user_id == user_id)
    if workspace_id is not None:
        stmt = stmt.where(Task.workspace_id == workspace_id)

    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_tasks(
    db: AsyncSession,
    *,
    user_id: UUID,
    workspace_id: UUID | None = None,
    status: str | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[Task]:
    filters = [Task.user_id == user_id]

    if workspace_id is not None:
        filters.append(Task.workspace_id == workspace_id)

    if status is not None:
        filters.append(Task.status == status)

    if start is not None:
        filters.append(Task.due_date >= start)
    if end is not None:
        filters.append(Task.due_date <= end)

    stmt = (
        select(Task)
        .where(and_(*filters))
        .order_by(Task.due_date.asc().nullslast(), Task.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def apply_task_update(db: AsyncSession, *, task: Task, patch: TaskUpdate) -> Task:
    updates = patch.model_dump(exclude_unset=True)

    # prevent ownership tampering
    updates.pop("user_id", None)
    updates.pop("workspace_id", None)  # not in schema, but safe guardrail

    for k, v in updates.items():
        setattr(task, k, v)

    await db.commit()
    await db.refresh(task)
    return task


async def delete_task(db: AsyncSession, *, task: Task) -> None:
    await db.delete(task)
    await db.commit()