"""FastAPI backend: submit a model comparison, check its status, list past runs.

POST /runs creates a row with status "pending" and returns immediately; the
comparison itself (which can take anywhere from seconds to several minutes
depending on model size) runs as a background task and updates the row to
"completed" or "failed" when done. This is a demo-scale choice: a single
in-process background task, not a real job queue, so a run is lost if the
server restarts mid-run.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.db import Run, make_session_factory
from api.schemas import RunCreateRequest, RunResponse
from api.service import execute_run

app = FastAPI(title="Model Provenance Verifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine, SessionLocal = make_session_factory()


@app.post("/runs", response_model=RunResponse, status_code=201)
def create_run(payload: RunCreateRequest, background_tasks: BackgroundTasks) -> RunResponse:
    session = SessionLocal()
    try:
        run = Run(model_a=payload.model_a, model_b=payload.model_b, status="pending")
        session.add(run)
        session.commit()
        session.refresh(run)
        response = RunResponse.model_validate(run)
    finally:
        session.close()

    background_tasks.add_task(execute_run, SessionLocal, response.id)
    return response


@app.get("/runs", response_model=list[RunResponse])
def list_runs() -> list[RunResponse]:
    session = SessionLocal()
    try:
        runs = session.query(Run).order_by(Run.id.desc()).all()
        return [RunResponse.model_validate(r) for r in runs]
    finally:
        session.close()


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: int) -> RunResponse:
    session = SessionLocal()
    try:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return RunResponse.model_validate(run)
    finally:
        session.close()
