from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RunCreateRequest(BaseModel):
    model_a: str
    model_b: str


class RunResponse(BaseModel):
    id: int
    model_a: str
    model_b: str
    model_a_revision: str | None
    model_b_revision: str | None
    probe_set_id: str | None
    status: str
    score: float | None
    vocab_compatible: bool | None
    method: str | None
    mean_cosine_similarity: float | None
    mean_kl_divergence: float | None
    mean_topk_jaccard: float | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None

    model_config = ConfigDict(from_attributes=True)
