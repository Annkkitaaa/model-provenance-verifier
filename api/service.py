"""Executes a comparison run and writes the result back to its row.

Kept separate from the FastAPI routes so it can run as a background task
(POST /runs returns immediately with status "pending") and so tests can
monkeypatch compute_output_distribution_signal to avoid downloading real
models just to test the run lifecycle.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import model_info

from provenance.signals.output_distribution import compute_output_distribution_signal

from api.db import Run


def _resolve_revision(model_id: str) -> str | None:
    """Returns the exact Hub commit SHA for a model id, or None if it can't
    be resolved (e.g. offline, or the id is invalid). A run is still useful
    without this; it just can't be reproduced against a moving "main" branch
    with full confidence.
    """
    try:
        return model_info(model_id).sha
    except Exception:
        return None


def execute_run(session_factory: Callable, run_id: int) -> None:
    session = session_factory()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return

        run.status = "running"
        session.commit()

        try:
            model_a_revision = _resolve_revision(run.model_a)
            model_b_revision = _resolve_revision(run.model_b)
            result = compute_output_distribution_signal(run.model_a, run.model_b)

            run.model_a_revision = model_a_revision
            run.model_b_revision = model_b_revision
            run.probe_set_id = result.probe_set_id
            run.score = result.score
            run.vocab_compatible = result.vocab_compatible
            run.method = result.method
            run.mean_cosine_similarity = result.mean_cosine_similarity
            run.mean_kl_divergence = result.mean_kl_divergence
            run.mean_topk_jaccard = result.mean_topk_jaccard
            run.status = "completed"
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)

        run.completed_at = datetime.now(timezone.utc)
        session.commit()
    finally:
        session.close()
