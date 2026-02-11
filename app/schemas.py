from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field


# ── Auth ──────────────────────────────────────────────
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class UserOut(BaseModel):
    id: UUID
    email: str
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
