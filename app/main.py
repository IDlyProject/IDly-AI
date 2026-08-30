from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile

from analysis_pipeline import run_analysis, _preload_korean_sentence_model

from .db import BASE_DIR
from .schemas import (
    AnalyzeMboxResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Mail Analysis API", version="0.2.0")
DEFAULT_ANALYSIS_KEYWORDS = ["보안"]

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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

async def _preload_model_background() -> None:
    """백그라운드에서 모델 프리로드 시도 (실패해도 무시)"""
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_preload_korean_sentence_model),
            timeout=120  # 2분 타임아웃
        )
        logger.info("Korean sentence model loaded successfully")
    except asyncio.TimeoutError:
        logger.warning("Korean sentence model load timed out, will use lexicon-based scoring")
    except Exception as e:
        logger.warning(f"Failed to load Korean sentence model: {e}, will use lexicon-based scoring")

@app.on_event("startup")
async def startup_event() -> None:
    # Keep-alive 태스크
    asyncio.create_task(_keep_alive())
    # 모델 프리로드 (실패해도 서버는 정상 작동)
    asyncio.create_task(_preload_model_background())


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
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=500, detail="Failed to save uploaded file") from exc
    finally:
        await file.close()

    return original_name, destination, total_size


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeMboxResponse)
async def analyze_mbox(file: UploadFile = File(...)) -> AnalyzeMboxResponse:
    _, destination, _ = await save_uploaded_mbox(file)

    try:
        result = await asyncio.to_thread(
            run_analysis,
            mbox_path=destination,
            keywords=DEFAULT_ANALYSIS_KEYWORDS,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        if destination.exists():
            destination.unlink()

    return AnalyzeMboxResponse(accounts=result.get("accounts", []))
