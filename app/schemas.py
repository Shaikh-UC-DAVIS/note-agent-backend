from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field


# ── Auth ──────────────────────────────────────────────
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    first_name: str | None = Field(default=None, max_length=120)
    last_name: str | None = Field(default=None, max_length=120)


class UserOut(BaseModel):
    id: UUID
    email: str
    first_name: str | None = None
    last_name: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Workspace ─────────────────────────────────────────
class WorkspaceCreate(BaseModel):
    name: str = Field(default="Default", max_length=255)


class WorkspaceOut(BaseModel):
    id: UUID
    name: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Note ──────────────────────────────────────────────
class NoteCreate(BaseModel):
    title: str = Field(max_length=500)
    raw_text: str = ""


class NoteUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    raw_text: str | None = None


class NoteOut(BaseModel):
    id: UUID
    workspace_id: UUID
    title: str
    raw_text: str
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ── Task ──────────────────────────────────────────────
from datetime import date  # add at top if you prefer, but this works here too

class TaskCreate(BaseModel):
    workspace_id: UUID
    title: str = Field(max_length=500)

    # optional metadata
    description: str | None = None
    status: str = Field(default="todo")  # adjust if you use other statuses
    due_date: date | None = None

    # optional links (only if your DB/model has these columns)
    user_id: UUID | None = None
    note_id: UUID | None = None


class TaskUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    description: str | None = None
    status: str | None = None
    due_date: date | None = None

    user_id: UUID | None = None
    note_id: UUID | None = None


class TaskOut(BaseModel):
    id: UUID
    workspace_id: UUID
    title: str
    description: str | None
    status: str
    due_date: date | None

    user_id: UUID | None
    note_id: UUID | None

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
