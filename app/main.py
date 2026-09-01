from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile

from .db import BASE_DIR, Base, SessionLocal, engine
from .models import Job, JobStatus
from .schemas import JobCreatedResponse, JobStatusResponse
from .service import create_job, get_job, run_job

logger = logging.getLogger(__name__)

app = FastAPI(title="Mail Analysis API", version="0.3.0")
DEFAULT_ANALYSIS_KEYWORDS = ["보안"]

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# Render 프리티어 cold start 방지: 14분마다 자기 자신에게 핑
SELF_URL = os.getenv("RENDER_EXTERNAL_URL", "")

_INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "")


def verify_internal_secret(x_internal_secret: str = Header(default="")) -> None:
    """IDly-Back 전용 내부 인증. INTERNAL_SECRET 환경변수가 설정된 경우에만 검증한다."""
    if not _INTERNAL_SECRET:
        return  # 환경변수 미설정 시 개발 편의를 위해 통과
    if x_internal_secret != _INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def _keep_alive() -> None:
    if not SELF_URL:
        return
    await asyncio.sleep(60)  # 시작 후 1분 대기
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await client.get(f"{SELF_URL}/health", timeout=10)
            except Exception as e:
                logger.debug(f"Keep-alive ping failed: {e}")
            await asyncio.sleep(14 * 60)  # 14분마다


@app.on_event("startup")
async def startup_event() -> None:
    Base.metadata.create_all(bind=engine)

    # 서버 재시작 시 running 상태로 멈춘 잡을 failed로 리셋.
    # 인메모리 태스크가 사라졌으므로 영구 미완료 상태를 방지한다.
    db = SessionLocal()
    try:
        stuck = db.query(Job).filter(Job.status == JobStatus.running.value).all()
        for job in stuck:
            job.status = JobStatus.failed.value
            job.error_message = "Server restarted while job was running"
        if stuck:
            db.commit()
            logger.warning(f"[startup] {len(stuck)} stuck running job(s) reset to failed")
    finally:
        db.close()

    asyncio.create_task(_keep_alive())


async def save_uploaded_mbox(file: UploadFile) -> tuple[str, Path, int]:
    original_name = Path(file.filename or "").name
    if not original_name or not original_name.lower().endswith(".mbox"):
        raise HTTPException(status_code=400, detail="Only .mbox files are allowed")

    safe_stem = Path(original_name).stem.replace(" ", "_")
    stored_name = f"{safe_stem}_{uuid.uuid4().hex}.mbox"
    destination = UPLOAD_DIR / stored_name

    total_size = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
                    )
                handle.write(chunk)
    except HTTPException:
        if destination.exists():
            destination.unlink()
        raise
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=500, detail="Failed to save uploaded file") from exc
    finally:
        await file.close()

    return original_name, destination, total_size


def _run_job_sync(job_id: str) -> None:
    """asyncio.to_thread를 통해 별도 스레드에서 실행된다."""
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            logger.error(f"Job {job_id} disappeared before background execution started")
            return
        run_job(db, job)
    finally:
        db.close()


async def _run_job_background(job_id: str) -> None:
    try:
        await asyncio.to_thread(_run_job_sync, job_id)
    except Exception:
        logger.exception(f"Unhandled error while running job {job_id} in the background")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/analyze",
    response_model=JobCreatedResponse,
    status_code=202,
    dependencies=[Depends(verify_internal_secret)],
)
async def analyze_mbox(file: UploadFile = File(...)) -> JobCreatedResponse:
    """.mbox를 업로드하면 분석 잡을 큐에 넣고 즉시 job_id를 반환한다 (202 Accepted).
    X-Internal-Secret 헤더 인증 필요."""
    _, destination, _ = await save_uploaded_mbox(file)

    db = SessionLocal()
    try:
        job = create_job(db, mbox_path=str(destination), keywords=DEFAULT_ANALYSIS_KEYWORDS)
        job_id = job.id
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=500, detail="Failed to queue analysis job") from exc
    finally:
        db.close()

    asyncio.create_task(_run_job_background(job_id))

    return JobCreatedResponse(job_id=job_id, status=JobStatus.queued.value)


@app.get(
    "/analyze/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(verify_internal_secret)],
)
def get_analysis_status(job_id: str) -> JobStatusResponse:
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status == JobStatus.succeeded.value:
            accounts = (job.result_json or {}).get("accounts", [])
            return JobStatusResponse(
                job_id=job.id,
                status=job.status,
                progress=job.progress,
                accounts=accounts,
            )

        if job.status == JobStatus.failed.value:
            return JobStatusResponse(
                job_id=job.id,
                status=job.status,
                progress=job.progress,
                error=job.error_message,
            )

        return JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
        )
    finally:
        db.close()
