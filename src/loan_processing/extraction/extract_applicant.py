"""Runs full document extraction for one applicant's mock document set and
prints the resulting structured record and model-input DataFrame.

Cross-checks the extracted values against the synthetic ground truth in
data/raw/applicants.csv — available only because we generated the mock
documents ourselves; a real pipeline wouldn't have this, but it's a useful
self-check on extraction fidelity here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from loan_processing.extraction.record_builder import (
    FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS,
    build_applicant_record,
    build_model_input,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MOCK_DOCS_DIR = REPO_ROOT / "data" / "mock_documents"
APPLICANTS_CSV = REPO_ROOT / "data" / "raw" / "applicants.csv"

_CROSS_CHECK_FIELDS = [
    "Age",
    "Employment_Type",
    "Monthly_Net_Income",
    "Total_Existing_EMIs",
    "Average_Monthly_Bank_Balance",
]


def extract_for_applicant(applicant_id: str) -> dict:
    salary_path = MOCK_DOCS_DIR / f"{applicant_id}_salary.pdf"
    bank_path = MOCK_DOCS_DIR / f"{applicant_id}_bank.xlsx"
    return build_applicant_record(salary_path, bank_path)


def main() -> None:
    if len(sys.argv) > 1:
        applicant_id = sys.argv[1]
    else:
        applicant_id = sorted(MOCK_DOCS_DIR.glob("*_salary.pdf"))[0].stem.removesuffix("_salary")

    print(f"Extracting documents for {applicant_id}...\n")
    record = extract_for_applicant(applicant_id)

    print("\nStructured record:")
    for key, value in record.items():
        flag = "  <- not available from salary slip / bank statement" if key in FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS else ""
        print(f"  {key}: {value}{flag}")

    model_input = build_model_input(record)
    print("\nModel-input DataFrame (columns/order match the saved XGBoost model):")
    print(model_input.to_string(index=False))
    print("\nDtypes:")
    print(model_input.dtypes.to_string())

    if APPLICANTS_CSV.exists():
        ground_truth = pd.read_csv(APPLICANTS_CSV).set_index("Applicant_ID")
        if applicant_id in ground_truth.index:
            truth = ground_truth.loc[applicant_id]
            print(f"\nCross-check against synthetic ground truth for {applicant_id}:")
            for field in _CROSS_CHECK_FIELDS:
                print(f"  {field}: extracted={record[field]!r}  true={truth[field]!r}")


if __name__ == "__main__":
    main()
