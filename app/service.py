from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from analysis_pipeline import run_analysis

from .models import AnalysisJob, JobStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def normalize_for_json(value):
    if isinstance(value, dict):
        return {str(k): normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_for_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def create_job(db: Session, mbox_path: str, keywords: list[str]) -> AnalysisJob:
    resolved = ensure_path(mbox_path)
    job = AnalysisJob(
        status=JobStatus.queued.value,
        progress=0,
        request_keywords=keywords,
        mbox_path=str(resolved),
        updated_at=utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> AnalysisJob | None:
    return db.get(AnalysisJob, job_id)


def claim_next_job(db: Session) -> AnalysisJob | None:
    queued_job = db.scalars(
        select(AnalysisJob)
        .where(AnalysisJob.status == JobStatus.queued.value)
        .order_by(AnalysisJob.created_at.asc())
        .limit(1)
    ).first()

    if not queued_job:
        return None

    rows = (
        db.query(AnalysisJob)
        .filter(
            AnalysisJob.id == queued_job.id,
            AnalysisJob.status == JobStatus.queued.value,
        )
        .update(
            {
                AnalysisJob.status: JobStatus.running.value,
                AnalysisJob.progress: 5,
                AnalysisJob.started_at: utcnow(),
                AnalysisJob.updated_at: utcnow(),
            },
            synchronize_session=False,
        )
    )
    db.commit()

    if rows == 0:
        return None

    return db.get(AnalysisJob, queued_job.id)


def run_job(db: Session, job: AnalysisJob) -> None:
    try:
        db.query(AnalysisJob).filter(AnalysisJob.id == job.id).update(
            {
                AnalysisJob.progress: 20,
                AnalysisJob.updated_at: utcnow(),
            },
            synchronize_session=False,
        )
        db.commit()

        result = run_analysis(mbox_path=job.mbox_path, keywords=job.request_keywords)
        serialized = normalize_for_json(result)

        db.query(AnalysisJob).filter(AnalysisJob.id == job.id).update(
            {
                AnalysisJob.status: JobStatus.succeeded.value,
                AnalysisJob.progress: 100,
                AnalysisJob.result_json: serialized,
                AnalysisJob.error_message: None,
                AnalysisJob.finished_at: utcnow(),
                AnalysisJob.updated_at: utcnow(),
            },
            synchronize_session=False,
        )
        db.commit()
    except Exception as exc:
        db.query(AnalysisJob).filter(AnalysisJob.id == job.id).update(
            {
                AnalysisJob.status: JobStatus.failed.value,
                AnalysisJob.progress: 100,
                AnalysisJob.error_message: str(exc),
                AnalysisJob.finished_at: utcnow(),
                AnalysisJob.updated_at: utcnow(),
            },
            synchronize_session=False,
        )
        db.commit()
