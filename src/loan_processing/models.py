"""Domain models shared across every pipeline stage."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PAY_STUB = "pay_stub"
    BANK_STATEMENT = "bank_statement"
    TAX_FORM = "tax_form"
    APPLICATION_FORM = "application_form"
    UNKNOWN = "unknown"


class UploadedDocument(BaseModel):
    filename: str
    doc_type: DocumentType
    raw_text: str


class ExtractedFields(BaseModel):
    doc_type: DocumentType
    source_filename: str
    fields: dict[str, str] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    field: str
    message: str
    severity: str  # "error" | "warning"


class ValidationResult(BaseModel):
    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class ScoreResult(BaseModel):
    score: int  # 300-850, FICO-like scale
    risk_tier: str  # "low" | "medium" | "high"
    factors: dict[str, float] = Field(default_factory=dict)


class DecisionOutcome(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    MANUAL_REVIEW = "manual_review"


class Decision(BaseModel):
    outcome: DecisionOutcome
    reasons: list[str] = Field(default_factory=list)


class LoanApplicationResult(BaseModel):
    application_id: str
    extracted: list[ExtractedFields]
    validation: ValidationResult
    score: ScoreResult
    decision: Decision
