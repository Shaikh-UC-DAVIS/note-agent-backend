# app/routes_tasks.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.schemas import TaskCreate, TaskOut, TaskUpdate

# ✅ minimal change: import CRUD helpers
from app.crud.tasks import (
    create_task as crud_create_task,
    get_task_or_none as crud_get_task_or_none,
    list_tasks as crud_list_tasks,
    apply_task_update as crud_apply_task_update,
    delete_task as crud_delete_task,
)

router = APIRouter(prefix="/tasks", tags=["tasks"])


# -------------------------
# Helpers
# -------------------------
def _parse_iso_date(d: Optional[str]) -> Optional[date]:
    if d is None:
        return None
    try:
        return date.fromisoformat(d)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date '{d}'. Expected YYYY-MM-DD.",
        )


def _compute_date_range(anchor: date, period: str) -> tuple[date, date]:
    period = period.lower().strip()

    if period == "day":
        return anchor, anchor

    if period == "week":
        start = anchor - timedelta(days=anchor.weekday())  # Monday start
        end = start + timedelta(days=6)
        return start, end

    if period == "month":
        start = anchor.replace(day=1)
        if start.month == 12:
            next_month = start.replace(year=start.year + 1, month=1, day=1)
        else:
            next_month = start.replace(month=start.month + 1, day=1)
        end = next_month - timedelta(days=1)
        return start, end

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid period. Use one of: day, week, month.",
    )


async def _get_task_or_404(
    db: AsyncSession, task_id: UUID, user: User, workspace_id: Optional[UUID] = None
):
    # ✅ minimal change: delegate to CRUD
    task = await crud_get_task_or_none(
        db,
        task_id=task_id,
        user_id=user.id,
        workspace_id=workspace_id,
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


# -------------------------
# Routes
# -------------------------
@router.post("/", response_model=TaskOut, status_code=status.HTTP_201_CREATED)
async def create_task(
    payload: TaskCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # ✅ minimal change: delegate to CRUD
    return await crud_create_task(db, user_id=user.id, payload=payload)


@router.get("/", response_model=list[TaskOut])
async def list_tasks(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    workspace_id: Optional[UUID] = Query(default=None),
    status_: Optional[str] = Query(default=None, alias="status"),
    start_date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    anchor_date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    period: Optional[str] = Query(default=None, description="day|week|month"),
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
):
    # Determine date window
    if period is not None:
        if anchor_date is None:
            raise HTTPException(
                status_code=400,
                detail="anchor_date is required when period is provided.",
            )
        a = _parse_iso_date(anchor_date)
        assert a is not None
        start, end = _compute_date_range(a, period)
    else:
        start = _parse_iso_date(start_date)
        end = _parse_iso_date(end_date)

    # ✅ minimal change: delegate to CRUD
    return await crud_list_tasks(
        db,
        user_id=user.id,
        workspace_id=workspace_id,
        status=status_,
        start=start,
        end=end,
        limit=limit,
        offset=offset,
    )


@router.get("/{task_id}", response_model=TaskOut)
async def get_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    workspace_id: Optional[UUID] = Query(default=None),
):
    return await _get_task_or_404(db, task_id, user, workspace_id=workspace_id)


@router.patch("/{task_id}", response_model=TaskOut)
async def update_task(
    task_id: UUID,
    payload: TaskUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    workspace_id: Optional[UUID] = Query(default=None),
):
    task = await _get_task_or_404(db, task_id, user, workspace_id=workspace_id)

    # ✅ minimal change: delegate to CRUD (includes anti-tamper guards)
    return await crud_apply_task_update(db, task=task, patch=payload)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    workspace_id: Optional[UUID] = Query(default=None),
):
    task = await _get_task_or_404(db, task_id, user, workspace_id=workspace_id)

    # ✅ minimal change: delegate to CRUD
    await crud_delete_task(db, task=task)
    return None