"""Combines salary-slip and bank-statement extraction into one structured
record, and builds a feature DataFrame whose column names, order, and
one-hot encoding exactly match what the saved XGBoost model expects.

Reuses `risk_modeling.train.FEATURE_COLUMNS` / `build_features` directly
rather than re-deriving the schema here, so the two can never drift apart.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from loan_processing.extraction.bank_statement_parser import parse_bank_statement
from loan_processing.extraction.salary_slip_parser import parse_salary_slip
from loan_processing.risk_modeling.train import FEATURE_COLUMNS, build_features

# The salary slip and bank statement can't supply these: Requested_Loan_Amount
# / Requested_Tenure_Months / and Is_First_Loan come from the loan
# application form (no extractor for that yet), and the mock bank statement
# doesn't simulate bounced-transaction line items. CIBIL_Score is not on this
# list because it isn't a model input at all (see risk_modeling.train) - a
# credit-bureau pull was deliberately excluded from the feature set.
FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS = [
    "Is_First_Loan",
    "Requested_Loan_Amount",
    "Requested_Tenure_Months",
    "Number_of_Bounced_Transactions_Last_6M",
]


def build_applicant_record(
    salary_slip_path: str | Path,
    bank_statement_path: str | Path,
    is_first_loan: bool | None = None,
) -> dict:
    """`is_first_loan` stands in for the loan-application-form input the real
    frontend collects (there's no form extractor yet). Leave it None to get
    the honest "unknown from these documents" behavior; pass True/False to
    simulate that form field being supplied, e.g. for the CLI demo.
    """
    salary_fields = parse_salary_slip(salary_slip_path)
    bank_fields = parse_bank_statement(bank_statement_path)

    bank_applicant_id = bank_fields.get("Applicant_ID")
    if bank_applicant_id not in (None, salary_fields["Applicant_ID"]):
        raise ValueError(
            "Applicant ID mismatch between documents: salary slip says "
            f"{salary_fields['Applicant_ID']!r}, bank statement says {bank_applicant_id!r}"
        )

    record: dict = {
        "Applicant_ID": salary_fields["Applicant_ID"],
        "Age": salary_fields["Age"],
        "Employment_Type": salary_fields["Employment_Type"],
        "Monthly_Net_Income": salary_fields["Monthly_Net_Income"],
        "Total_Existing_EMIs": bank_fields["Total_Existing_EMIs"],
        "Average_Monthly_Bank_Balance": bank_fields["Average_Monthly_Bank_Balance"],
    }
    for field in FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS:
        record[field] = math.nan

    if is_first_loan is not None:
        record["Is_First_Loan"] = 1 if is_first_loan else 0

    return record


def build_model_input(record: dict) -> pd.DataFrame:
    """Single-row DataFrame with columns/order matching the saved XGBoost model."""
    raw_df = pd.DataFrame([record])
    features = build_features(raw_df)
    return features[FEATURE_COLUMNS]
