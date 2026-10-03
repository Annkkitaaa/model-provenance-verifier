"""SQLite persistence for comparison runs.

A run is versioned: it records the probe set id and the exact model
revisions (Hub commit SHAs, not just repo names) that produced its score, so
the run can be reproduced later even if a model repo's "main" branch moves on.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DEFAULT_DATABASE_URL = "sqlite:///./runs.db"


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_a: Mapped[str] = mapped_column(String, nullable=False)
    model_b: Mapped[str] = mapped_column(String, nullable=False)
    model_a_revision: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model_b_revision: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    probe_set_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vocab_compatible: Mapped[Optional[bool]] = mapped_column(nullable=True)
    method: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mean_cosine_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_kl_divergence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_topk_jaccard: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


def make_session_factory(database_url: str | None = None) -> tuple[object, sessionmaker]:
    """Creates an engine and session factory for the given database URL.

    Tests pass their own in-memory URL; the app uses PROVENANCE_DATABASE_URL
    (or the sqlite file default) so a real run persists across restarts.
    """
    url = database_url or os.environ.get("PROVENANCE_DATABASE_URL", DEFAULT_DATABASE_URL)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return engine, SessionLocal
