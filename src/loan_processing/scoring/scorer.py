"""Rules-based credit/risk scoring on a FICO-like 300-850 scale.

This is a transparent baseline (debt-to-income, income level, bank balance
cushion). It is deliberately isolated behind `compute_score` so it can later
be swapped for, or blended with, a trained model without touching the rest
of the pipeline.
"""
from __future__ import annotations

from loan_processing.models import DocumentType, ExtractedFields, ScoreResult

_BASE_SCORE = 650
_MIN_SCORE = 300
_MAX_SCORE = 850


def _to_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def _fields_for(extracted: list[ExtractedFields], doc_type: DocumentType) -> dict[str, str]:
    for doc in extracted:
        if doc.doc_type == doc_type:
            return doc.fields
    return {}


def compute_score(extracted: list[ExtractedFields]) -> ScoreResult:
    application = _fields_for(extracted, DocumentType.APPLICATION_FORM)
    tax_form = _fields_for(extracted, DocumentType.TAX_FORM)
    pay_stub = _fields_for(extracted, DocumentType.PAY_STUB)
    bank_statement = _fields_for(extracted, DocumentType.BANK_STATEMENT)

    annual_income = _to_float(tax_form.get("annual_income"))
    if annual_income is None:
        gross_pay = _to_float(pay_stub.get("gross_pay"))
        annual_income = gross_pay * 12 if gross_pay is not None else None

    monthly_debt = _to_float(application.get("monthly_debt")) or 0.0
    closing_balance = _to_float(bank_statement.get("closing_balance"))

    score = float(_BASE_SCORE)
    factors: dict[str, float] = {}

    if annual_income is not None:
        monthly_income = annual_income / 12
        factors["monthly_income"] = monthly_income
        if monthly_income > 0:
            dti = monthly_debt / monthly_income
            factors["debt_to_income_ratio"] = round(dti, 4)
            if dti <= 0.2:
                score += 80
            elif dti <= 0.36:
                score += 30
            elif dti <= 0.5:
                score -= 40
            else:
                score -= 100

        if monthly_income >= 8000:
            score += 40
        elif monthly_income >= 4000:
            score += 15
        elif monthly_income < 2000:
            score -= 30
    else:
        factors["debt_to_income_ratio"] = -1.0
        score -= 60  # unverifiable income is a strong negative signal

    if closing_balance is not None:
        factors["closing_balance"] = closing_balance
        if closing_balance >= 10000:
            score += 30
        elif closing_balance < 500:
            score -= 30

    score = max(_MIN_SCORE, min(_MAX_SCORE, round(score)))

    if score >= 720:
        risk_tier = "low"
    elif score >= 620:
        risk_tier = "medium"
    else:
        risk_tier = "high"

    return ScoreResult(score=score, risk_tier=risk_tier, factors=factors)
