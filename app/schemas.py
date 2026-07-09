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
