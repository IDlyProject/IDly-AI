from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from .db import Base, engine, get_db
from .schemas import CreateJobRequest, CreateJobResponse, JobResultResponse, JobStatusResponse
from .service import create_job, get_job

app = FastAPI(title="Mail Analysis API", version="0.1.0")

Base.metadata.create_all(bind=engine)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", response_model=CreateJobResponse)
def create_analysis_job(payload: CreateJobRequest, db: Session = Depends(get_db)) -> CreateJobResponse:
    job = create_job(db=db, mbox_path=payload.mbox_path, keywords=payload.keywords)
    return CreateJobResponse(job_id=job.id, status=job.status)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_analysis_job(job_id: str, db: Session = Depends(get_db)) -> JobStatusResponse:
    job = get_job(db=db, job_id=job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        mbox_path=job.mbox_path,
        request_keywords=job.request_keywords,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        updated_at=job.updated_at,
    )


@app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_analysis_result(job_id: str, db: Session = Depends(get_db)) -> JobResultResponse:
    job = get_job(db=db, job_id=job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResultResponse(
        id=job.id,
        status=job.status,
        result=job.result_json,
    )
