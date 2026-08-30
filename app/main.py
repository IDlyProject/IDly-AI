from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile

from .db import BASE_DIR, Base, SessionLocal, engine
from .models import JobStatus
from .schemas import JobCreatedResponse, JobStatusResponse
from .service import create_job, get_job, run_job

logger = logging.getLogger(__name__)

app = FastAPI(title="Mail Analysis API", version="0.3.0")
DEFAULT_ANALYSIS_KEYWORDS = ["보안"]

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 업로드 용량 제한 없음: /analyze는 비동기 job이라 처리 시간이 길어져도 HTTP 요청이
# 걸려 있지 않으므로 게이트웨이 타임아웃 위험이 없다. 대신 파일이 클수록 백그라운드
# 잡 처리 시간(및 메모리 사용량)이 그만큼 늘어난다.

# Render 프리티어 cold start 방지: 14분마다 자기 자신에게 핑
SELF_URL = os.getenv("RENDER_EXTERNAL_URL", "")


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
    # 잡 큐 테이블 생성. worker.py를 별도 프로세스로 안 띄워도 이 웹 서비스 단독으로
    # /analyze -> 백그라운드 잡 실행까지 전부 처리할 수 있어야 하므로 여기서도 생성해둔다.
    Base.metadata.create_all(bind=engine)
    # Keep-alive 태스크
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
                handle.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=500, detail="Failed to save uploaded file") from exc
    finally:
        await file.close()

    return original_name, destination, total_size


def _run_job_sync(job_id: str) -> None:
    """asyncio.to_thread를 통해 별도 스레드에서 실행된다. 요청을 처리하던 세션을 그대로
    재사용하면 스레드 세이프하지 않으므로, 이 잡 전용 DB 세션을 새로 열어 끝까지 그 안에서만
    사용한다."""
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
    """NLP(의미 기반 스코어링) -> 분석(TFIDF/클러스터링/이상탐지) 전 단계를 서버가 자동으로
    이어서 실행한다. 클라이언트는 결과를 기다리지 않고 즉시 job_id를 받으며, 이 태스크는
    HTTP 요청/응답 사이클과 완전히 분리돼 있어 Render 게이트웨이 타임아웃에 걸려 502가
    나는 경우가 없다. (다만 이 프로세스 자체가 재시작되면 실행 중이던 잡은 이어받지
    못하고 running 상태로 멈춘다 - 별도 워커 프로세스로 분리하기 전까지의 알려진 한계.)"""
    try:
        await asyncio.to_thread(_run_job_sync, job_id)
    except Exception:
        logger.exception(f"Unhandled error while running job {job_id} in the background")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=JobCreatedResponse, status_code=202)
async def analyze_mbox(file: UploadFile = File(...)) -> JobCreatedResponse:
    """.mbox를 업로드하면 분석 잡을 큐에 넣고 즉시 job_id를 반환한다 (202 Accepted).
    실제 NLP 스코어링 + 분석은 백그라운드에서 서버가 자동으로 이어서 실행하며,
    진행 상황과 최종 결과는 GET /analyze/{job_id}로 폴링해서 확인한다."""
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


@app.get("/analyze/{job_id}", response_model=JobStatusResponse)
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
