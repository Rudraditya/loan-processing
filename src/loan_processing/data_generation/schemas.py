"""Schema for the synthetic loan applicant dataset.

`full_name` and `employer_name` aren't in the required column list but are
included now because document_simulation (next phase) needs identity/employer
text to render onto mock pay stubs and bank statements.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EmploymentType(str, Enum):
    SALARIED = "Salaried"
    SELF_EMPLOYED = "Self-Employed"


class ApplicantRecord(BaseModel):
    applicant_id: str
    full_name: str
    employer_name: str
    age: int = Field(ge=21, le=65)
    employment_type: EmploymentType
    monthly_net_income: float = Field(gt=0)
    total_existing_emis: float = Field(ge=0)
    cibil_score: int = Field(ge=300, le=900)
    requested_loan_amount: float = Field(gt=0)
    requested_tenure_months: int = Field(gt=0)
    average_monthly_bank_balance: float = Field(ge=0)
    number_of_bounced_transactions_last_6m: int = Field(ge=0)
