# app/models.py
from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

from sqlalchemy import Date, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Text,
    Date,
    DateTime,
    ForeignKey,
    Enum as SAEnum,
    JSON,
    Integer,
    Float,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, declarative_base
from pgvector.sqlalchemy import Vector


def utcnow():
    return datetime.now(timezone.utc)


EMBEDDING_DIM = 384


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(320), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    first_name = Column(String(120), nullable=True)
    last_name = Column(String(120), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    workspaces = relationship(
        "Workspace",
        back_populates="owner",
        cascade="all, delete-orphan",
    )


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name = Column(String(255), nullable=False, default="Default")
    settings_json = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    owner = relationship("User", back_populates="workspaces")
    notes = relationship("Note", back_populates="workspace", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="workspace", cascade="all, delete-orphan")


NOTE_STATUS = SAEnum(
    "created",
    "extracted",
    "chunked",
    "embedded",
    "structured",
    "resolved",
    "ready",
    "error",
    name="note_status",
    create_type=True,
)

OBJECT_TYPE = SAEnum(
    "Idea", "Claim", "Assumption", "Question",
    "Task", "Evidence", "Definition",
    name="object_type",
    create_type=True,
)

LINK_TYPE = SAEnum(
    "Supports", "Contradicts", "Refines",
    "DependsOn", "SameAs", "Causes",
    name="link_type",
    create_type=True,
)

INSIGHT_TYPE = SAEnum(
    "contradiction", "stale_thread", "consolidation_opportunity",
    name="insight_type",
    create_type=True,
)


class Note(Base):
    __tablename__ = "notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    title = Column(String(500), nullable=False)
    raw_text = Column(Text, nullable=False, default="")
    cleaned_text = Column(Text, nullable=True)
    content_hash = Column(String(64), nullable=True)
    status = Column(NOTE_STATUS, nullable=False, server_default="created")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    workspace = relationship("Workspace", back_populates="notes")
    spans = relationship("Span", back_populates="note", cascade="all, delete-orphan")



class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Required by schema
    workspace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)

    # Optional metadata
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # You can enforce allowed values via CHECK/Enum later; for now store as string
    status: Mapped[str] = mapped_column(String(50), nullable=False, server_default="todo", index=True)

    due_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)

    # Optional links
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    note_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("notes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    workspace = relationship("Workspace", back_populates="tasks")


# ── ML tables ──────────────────────────────────────────────────────────────
class Span(Base):
    __tablename__ = "spans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    note_id = Column(UUID(as_uuid=True), ForeignKey("notes.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    note = relationship("Note", back_populates="spans")


class Object(Base):
    __tablename__ = "objects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(OBJECT_TYPE, nullable=False)
    canonical_text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    status = Column(String(64), nullable=False, default="active")
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class ObjectMention(Base):
    __tablename__ = "object_mentions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    object_id = Column(UUID(as_uuid=True), ForeignKey("objects.id", ondelete="CASCADE"), nullable=False)
    span_id = Column(UUID(as_uuid=True), ForeignKey("spans.id", ondelete="CASCADE"), nullable=False)
    note_id = Column(UUID(as_uuid=True), ForeignKey("notes.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(64), nullable=False, default="primary")
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Link(Base):
    __tablename__ = "links"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    src_object_id = Column(UUID(as_uuid=True), ForeignKey("objects.id", ondelete="CASCADE"), nullable=False)
    dst_object_id = Column(UUID(as_uuid=True), ForeignKey("objects.id", ondelete="CASCADE"), nullable=False)
    type = Column(LINK_TYPE, nullable=False)
    confidence = Column(Float, nullable=True)
    evidence_span_id = Column(UUID(as_uuid=True), ForeignKey("spans.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Insight(Base):
    __tablename__ = "insights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(INSIGHT_TYPE, nullable=False)
    severity = Column(String(16), nullable=True)
    status = Column(String(32), nullable=False, default="new")
    payload = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)