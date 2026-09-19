#!/usr/bin/env python
"""AI-powered Loan Assessment Agent — end-to-end CLI.

Usage:
    python app.py <Applicant_ID> <Date_Of_Birth YYYY-MM-DD>

Locates the applicant's mock salary-slip PDF and bank-statement XLSX under
data/mock_documents/, extracts structured financial fields, scores them with
the saved Logistic Regression credit-risk model, and prints a Customer Risk
Analysis Report with a final processing recommendation. Age is calculated
from the supplied date of birth as of today, not extracted from the
documents.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from loan_processing.extraction.record_builder import (
    FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS,
    build_applicant_record,
    build_model_input,
    calculate_age,
)

REPO_ROOT = Path(__file__).resolve().parent
MOCK_DOCS_DIR = REPO_ROOT / "data" / "mock_documents"
MODELS_DIR = REPO_ROOT / "models"

# Illustrative underwriting sanity bounds for the report's consistency
# checks — not calibrated policy limits.
MIN_PLAUSIBLE_MONTHLY_INCOME = 10_000.0
HIGH_EMI_TO_INCOME_RATIO = 0.50
LOW_BALANCE_TO_INCOME_RATIO = 0.25  # balance under a quarter month's income looks thin

_MISSING_FIELD_REASONS = {
    "Is_First_Loan": "requires the loan application form",
    "Requested_Loan_Amount": "requires the loan application form",
    "Requested_Tenure_Months": "requires the loan application form",
}

# The model doesn't use a credit-bureau score (CIBIL_Score) as an input for
# any applicant - these are the alternative-data factors it relies on
# instead, called out explicitly in the NTC section since a first-time
# borrower also has no repayment history behind them.
NTC_ALTERNATIVE_DATA_FACTORS = ["Monthly_Net_Income", "Average_Monthly_Bank_Balance"]


def locate_documents(applicant_id: str) -> tuple[Path, Path]:
    salary_path = MOCK_DOCS_DIR / f"{applicant_id}_salary.pdf"
    bank_path = MOCK_DOCS_DIR / f"{applicant_id}_bank.xlsx"
    missing = [p for p in (salary_path, bank_path) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Could not find document(s) for applicant '{applicant_id}' in {MOCK_DOCS_DIR}: {names}"
        )
    return salary_path, bank_path


def load_model_artifacts() -> tuple:
    # The pipeline bundles preprocessing (one-hot encoding Employment_Type,
    # median-imputing + scaling the numeric features) and the classifier
    # into one fitted sklearn Pipeline - see risk_modeling/train.py.
    pipeline = joblib.load(MODELS_DIR / "logistic_regression_pipeline.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    return pipeline, feature_columns


def score_applicant(model_input: pd.DataFrame, pipeline, feature_columns: list[str]) -> tuple[str, float]:
    assert list(model_input.columns) == feature_columns, (
        "Feature order from the extraction layer doesn't match the saved model's "
        "feature_columns.pkl — the two have drifted out of sync."
    )
    probability_of_default = float(pipeline.predict_proba(model_input)[0, 1])
    predicted_class = "Default" if probability_of_default >= 0.5 else "No Default"
    return predicted_class, probability_of_default


def run_consistency_checks(record: dict) -> list[str]:
    checks: list[str] = []

    income = record["Monthly_Net_Income"]
    emi = record["Total_Existing_EMIs"]
    balance = record["Average_Monthly_Bank_Balance"]

    if income < MIN_PLAUSIBLE_MONTHLY_INCOME:
        checks.append(
            f"[FLAG] Monthly net income (Rs. {income:,.2f}) is below the plausible minimum "
            f"of Rs. {MIN_PLAUSIBLE_MONTHLY_INCOME:,.2f} - verify the salary slip is genuine."
        )
    else:
        checks.append(f"[OK] Monthly net income (Rs. {income:,.2f}) is within a plausible range.")

    emi_ratio = emi / income if income else float("inf")
    if emi_ratio > HIGH_EMI_TO_INCOME_RATIO:
        checks.append(
            f"[FLAG] EMI-to-income ratio ({emi_ratio:.1%}) exceeds the "
            f"{HIGH_EMI_TO_INCOME_RATIO:.0%} high-burden threshold."
        )
    else:
        checks.append(f"[OK] EMI-to-income ratio ({emi_ratio:.1%}) is within a serviceable range.")

    if income and balance / income < LOW_BALANCE_TO_INCOME_RATIO:
        checks.append(
            f"[FLAG] Average bank balance (Rs. {balance:,.2f}) is thin relative to monthly "
            f"income (Rs. {income:,.2f}) - under {LOW_BALANCE_TO_INCOME_RATIO:.0%} of a month's income."
        )
    else:
        checks.append(f"[OK] Average bank balance (Rs. {balance:,.2f}) looks healthy relative to income.")

    for field in FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS:
        value = record.get(field)
        if isinstance(value, float) and math.isnan(value):
            checks.append(f"[MISSING] {field} not available from these documents - {_MISSING_FIELD_REASONS[field]}.")

    return checks


def build_report(
    applicant_id: str,
    record: dict,
    checks: list[str],
    predicted_class: str,
    probability_of_default: float,
    total_features: int,
) -> str:
    missing_required_fields = [
        f
        for f in FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS
        if isinstance(record.get(f), float) and math.isnan(record[f])
    ]
    has_data_flags = any(line.startswith("[FLAG]") for line in checks)

    income = record["Monthly_Net_Income"]
    emi = record["Total_Existing_EMIs"]
    balance = record["Average_Monthly_Bank_Balance"]
    cash_flow = income - emi

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("CUSTOMER RISK ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Applicant ID     : {applicant_id}")
    lines.append(f"Report Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("1. EXTRACTED FINANCIAL SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Employment Type              : {record['Employment_Type']}")
    lines.append(f"Age                          : {record['Age']}")
    lines.append(f"Monthly Net Income           : Rs. {income:,.2f}")
    lines.append(f"Total Existing EMIs          : Rs. {emi:,.2f}")
    lines.append(f"Estimated Monthly Cash Flow  : Rs. {cash_flow:,.2f}  (Income - EMIs)")
    lines.append(f"Average Monthly Bank Balance : Rs. {balance:,.2f}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("2. DATA QUALITY / CONSISTENCY CHECKS")
    lines.append("-" * 70)
    lines.extend(checks)

    lines.append("")
    lines.append("-" * 70)
    lines.append("3. AI RISK PROFILE PREDICTION (Logistic Regression)")
    lines.append("-" * 70)
    lines.append(f"Predicted Class         : {predicted_class}")
    lines.append(f"Default Probability     : {probability_of_default:.2%}")
    if missing_required_fields:
        lines.append(
            f"Confidence Note         : computed with {len(missing_required_fields)} of "
            f"{total_features} model features missing ({', '.join(missing_required_fields)}). "
            "Logistic Regression can't accept missing values directly, so each was substituted "
            "with its training-set median before scoring - this prediction should be treated as "
            "a preliminary signal only, not a final assessment."
        )

    is_ntc = record.get("Is_First_Loan") == 1
    if is_ntc:
        lines.append("")
        lines.append("-" * 70)
        lines.append("4. NEW-TO-CREDIT (NTC) APPLICANT NOTE")
        lines.append("-" * 70)
        lines.append(
            "This applicant is taking out their first loan. This model does not use a credit "
            "bureau score (CIBIL_Score) as an input for any applicant, so this assessment isn't "
            "weighted any differently on that front than a returning borrower's - it rests "
            f"entirely on alternative data: {', '.join(NTC_ALTERNATIVE_DATA_FACTORS)}. A "
            "first-time borrower does, however, have no repayment track record behind those "
            "numbers yet, so treat this prediction as directional; an underwriter should "
            "corroborate with other NTC-appropriate signals (e.g. utility/rent payment history) "
            "before final decisioning."
        )

    lines.append("")
    lines.append("-" * 70)
    lines.append(f"{5 if is_ntc else 4}. PROCESSING RECOMMENDATION")
    lines.append("-" * 70)
    if missing_required_fields:
        lines.append(">>> FLAG FOR MANUAL REVIEW <<<")
        lines.append(
            "Reason: required underwriting inputs are missing "
            f"({', '.join(missing_required_fields)}). Route to an underwriter to supplement "
            "with the applicant's loan application form before final decisioning."
        )
    elif has_data_flags:
        lines.append(">>> FLAG FOR MANUAL REVIEW <<<")
        lines.append("Reason: one or more data consistency checks raised a flag (see Section 2).")
    elif predicted_class == "Default":
        lines.append(">>> FLAG FOR MANUAL REVIEW <<<")
        lines.append(f"Reason: AI model predicts elevated default risk ({probability_of_default:.2%}).")
    else:
        lines.append(">>> PROCEED TO UNDERWRITING <<<")
        lines.append("Reason: no data quality flags raised and AI model indicates low default risk.")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-powered Loan Assessment Agent")
    parser.add_argument("applicant_id", help="Applicant_ID to process, e.g. APP01732")
    parser.add_argument(
        "date_of_birth",
        help=(
            "Applicant's date of birth, YYYY-MM-DD. Age is calculated from this as of "
            "today rather than extracted from the documents (a printed age goes stale "
            "the moment it ages past the document's issue date)."
        ),
    )
    parser.add_argument(
        "employment_status",
        choices=["Salaried", "Self-Employed", "Business Owner"],
        help=(
            "Stands in for the loan-application-form field - no longer extracted from "
            "the salary slip (a self-reported document label isn't a reliable substitute "
            "for what the applicant actually selects on the form, and a document can't "
            "offer a category like 'Business Owner' beyond whatever two values happen to "
            "be printed on it)."
        ),
    )
    parser.add_argument(
        "--first-loan",
        choices=["yes", "no"],
        default=None,
        help=(
            "Stands in for the loan-application-form field (there's no form extractor yet). "
            "Omit to leave Is_First_Loan as unknown/NaN, same as the other application-form-only "
            "fields."
        ),
    )
    args = parser.parse_args()

    try:
        date_of_birth = datetime.strptime(args.date_of_birth, "%Y-%m-%d").date()
    except ValueError as exc:
        parser.error(f"date_of_birth must be in YYYY-MM-DD format: {exc}")

    salary_path, bank_path = locate_documents(args.applicant_id)
    print(f"Located documents: {salary_path.name}, {bank_path.name}\n")

    age = calculate_age(date_of_birth)
    is_first_loan = None if args.first_loan is None else args.first_loan == "yes"
    record = build_applicant_record(
        salary_path, bank_path, args.applicant_id, age, args.employment_status, is_first_loan=is_first_loan
    )
    model_input = build_model_input(record)

    pipeline, feature_columns = load_model_artifacts()
    predicted_class, probability_of_default = score_applicant(model_input, pipeline, feature_columns)

    checks = run_consistency_checks(record)
    report = build_report(
        args.applicant_id, record, checks, predicted_class, probability_of_default, len(feature_columns)
    )

    print(report)


if __name__ == "__main__":
    main()
