from __future__ import annotations

from pydantic import BaseModel


class ProblemMailItem(BaseModel):
    subject: str
    date: str
    body: str
    matched_keywords: str


class AccountAnalysisItem(BaseModel):
    account_id: str
    account: str
    security_score: float
    security_level: str
    interpretation: str
    problem_mails: list[ProblemMailItem]


class AnalyzeMboxResponse(BaseModel):
    accounts: list[AccountAnalysisItem]


class JobCreatedResponse(BaseModel):
    """POST /analyze 응답. 잡을 큐에 넣었다는 확인만 즉시 돌려주고,
    실제 NLP + 분석은 서버가 백그라운드에서 자동으로 이어서 실행한다."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """GET /analyze/{job_id} 응답. status가 'succeeded'가 되면 accounts가,
    'failed'가 되면 error가 채워진다."""

    job_id: str
    status: str
    progress: int
    accounts: list[AccountAnalysisItem] | None = None
    error: str | None = None
