"""Schema for the synthetic loan applicant dataset.

`full_name` and `employer_name` aren't in the required column list but are
included now because document_simulation (next phase) needs identity/employer
text to render onto mock pay stubs and bank statements.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class EmploymentType(str, Enum):
    SALARIED = "Salaried"
    SELF_EMPLOYED = "Self-Employed"
    BUSINESS_OWNER = "Business Owner"


class ApplicantRecord(BaseModel):
    applicant_id: str
    full_name: str
    employer_name: str
    age: int = Field(ge=21, le=65)
    employment_type: EmploymentType
    monthly_net_income: float = Field(gt=0)
    gross_income: float = Field(gt=0)
    total_deductions: float = Field(ge=0)
    total_existing_emis: float = Field(ge=0)
    is_first_loan: int = Field(ge=0, le=1)
    cibil_score: int | None = Field(default=None, ge=300, le=900)
    requested_loan_amount: float = Field(gt=0)
    requested_tenure_months: int = Field(gt=0)
    average_monthly_bank_balance: float = Field(ge=0)
    number_of_bounced_transactions_last_6m: int = Field(ge=0)

    @model_validator(mode="after")
    def _cibil_score_matches_first_loan_status(self) -> "ApplicantRecord":
        # New-to-credit (NTC) applicants have no credit history to score,
        # so CIBIL_Score must be genuinely absent for them rather than
        # defaulted to a fabricated value; conversely, returning borrowers
        # must have a real score.
        if self.is_first_loan == 1 and self.cibil_score is not None:
            raise ValueError("cibil_score must be null for first-time (NTC) applicants")
        if self.is_first_loan == 0 and self.cibil_score is None:
            raise ValueError("cibil_score is required for applicants with prior credit history")
        return self

    @model_validator(mode="after")
    def _net_income_reconciles_with_gross_and_deductions(self) -> "ApplicantRecord":
        if abs((self.gross_income - self.total_deductions) - self.monthly_net_income) > 0.02:
            raise ValueError("monthly_net_income must equal gross_income - total_deductions")
        return self
