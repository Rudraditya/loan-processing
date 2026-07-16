"""Applies underwriting rules to a score + validation result to reach a decision."""
from __future__ import annotations

from loan_processing.models import Decision, DecisionOutcome, ScoreResult, ValidationResult

_AUTO_APPROVE_SCORE = 700
_AUTO_DENY_SCORE = 580


def decide(score_result: ScoreResult, validation_result: ValidationResult) -> Decision:
    reasons: list[str] = []

    if not validation_result.is_valid:
        reasons.extend(issue.message for issue in validation_result.issues if issue.severity == "error")
        return Decision(outcome=DecisionOutcome.MANUAL_REVIEW, reasons=reasons)

    if score_result.score >= _AUTO_APPROVE_SCORE:
        reasons.append(f"Score {score_result.score} meets auto-approval threshold ({_AUTO_APPROVE_SCORE}).")
        return Decision(outcome=DecisionOutcome.APPROVED, reasons=reasons)

    if score_result.score <= _AUTO_DENY_SCORE:
        reasons.append(f"Score {score_result.score} is at or below auto-denial threshold ({_AUTO_DENY_SCORE}).")
        return Decision(outcome=DecisionOutcome.DENIED, reasons=reasons)

    reasons.append(
        f"Score {score_result.score} falls between denial ({_AUTO_DENY_SCORE}) and "
        f"approval ({_AUTO_APPROVE_SCORE}) thresholds."
    )
    return Decision(outcome=DecisionOutcome.MANUAL_REVIEW, reasons=reasons)
