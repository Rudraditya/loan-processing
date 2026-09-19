"""Combines salary-slip and bank-statement extraction into one structured
record, and builds a feature DataFrame whose column names, order, and
one-hot encoding exactly match what the saved XGBoost model expects.

Reuses `risk_modeling.train.FEATURE_COLUMNS` / `build_features` directly
rather than re-deriving the schema here, so the two can never drift apart.
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd

from loan_processing.extraction.bank_statement_parser import expand_cash_flow_trend, parse_bank_statement
from loan_processing.extraction.salary_slip_parser import parse_salary_slip
from loan_processing.risk_modeling.train import FEATURE_COLUMNS, build_features


def calculate_age(date_of_birth: date, as_of: date | None = None) -> int:
    """Whole years between `date_of_birth` and `as_of` (today by default) -
    the standard birthday-aware definition, not a naive days/365 division.
    Shared by agentic_api/main.py (web form input) and app.py (CLI input),
    so "age as of today" means the same thing in both places.
    """
    as_of = as_of or date.today()
    years = as_of.year - date_of_birth.year
    if (as_of.month, as_of.day) < (date_of_birth.month, date_of_birth.day):
        years -= 1
    return years

# The salary slip and bank statement can't supply these: Requested_Loan_Amount
# / Requested_Tenure_Months / and Is_First_Loan come from the loan
# application form (no extractor for that yet). CIBIL_Score is not on this
# list because it isn't a model input at all (see risk_modeling.train) - a
# credit-bureau pull was deliberately excluded from the feature set.
# Number_of_Bounced_Transactions_Last_6M *used* to be on this list - the mock
# bank statement didn't simulate bounced-transaction line items at all - but
# document_simulation/bank_statement_xlsx.py now renders them and
# bank_statement_parser.py counts them back out, so it's a real
# document-derived field below instead.
FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS = [
    "Is_First_Loan",
    "Requested_Loan_Amount",
    "Requested_Tenure_Months",
]


def build_applicant_record(
    salary_slip_path: str | Path,
    bank_statement_path: str | Path,
    applicant_id: str,
    age: int,
    employment_type: str,
    is_first_loan: bool | None = None,
) -> dict:
    """`applicant_id` is a serial number the caller assigns, not a value
    recovered from the documents - the regex parsers don't extract it (see
    salary_slip_parser.py), since a real-world salary slip/bank statement
    has no machine-readable applicant identifier the way this project's own
    mock documents do. `age` is likewise caller-supplied rather than
    document-extracted - callers should compute it from a date of birth via
    `calculate_age()` above rather than trust a possibly-stale printed
    figure. `employment_type` is a loan-application-form field too now -
    unlike `age`/`applicant_id`, it used to be document-extracted, but a
    self-reported document label isn't a reliable substitute for what the
    applicant actually selects on the form, and the model now supports a
    third category ("Business Owner") no document label could offer, so
    it's always caller-supplied. `is_first_loan` stands in for the
    loan-application-form input the real frontend collects (there's no form
    extractor yet). Leave it None to get the honest "unknown from these
    documents" behavior; pass True/False to simulate that form field being
    supplied, e.g. for the CLI demo.
    """
    salary_fields = parse_salary_slip(salary_slip_path)
    bank_fields = parse_bank_statement(bank_statement_path)

    record: dict = {
        "Applicant_ID": applicant_id,
        "Age": age,
        "Employment_Type": employment_type,
        "Monthly_Net_Income": salary_fields["Monthly_Net_Income"],
        "Gross_Income": salary_fields["Gross_Income"],
        "Total_Deductions": salary_fields["Total_Deductions"],
        "Total_Existing_EMIs": bank_fields["Total_Existing_EMIs"],
        "Average_Monthly_Bank_Balance": bank_fields["Average_Monthly_Bank_Balance"],
        "Number_of_Bounced_Transactions_Last_6M": bank_fields["Number_of_Bounced_Transactions_Last_6M"],
        **expand_cash_flow_trend(bank_fields.get("Cash_Flow_Trend")),
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
