from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CreateJobRequest(BaseModel):
    mbox_path: str = Field(..., description="Absolute or workspace-relative path to mbox file")
    keywords: list[str] = Field(..., min_length=1, description="Keywords used for filtering")


class CreateJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    id: str
    status: str
    progress: int
    mbox_path: str
    request_keywords: list[str]
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime


class JobResultResponse(BaseModel):
    id: str
    status: str
    result: dict | None
