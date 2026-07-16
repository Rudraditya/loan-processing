"""Samples applicants from the generated dataset and renders a mock salary
slip (PDF) and bank statement (XLSX) for each — test inputs for the
document extraction / verification layer.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from loan_processing.document_simulation.bank_statement_xlsx import render_bank_statement
from loan_processing.document_simulation.salary_slip_pdf import render_salary_slip

REPO_ROOT = Path(__file__).resolve().parents[3]
APPLICANTS_CSV = REPO_ROOT / "data" / "raw" / "applicants.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "mock_documents"

SAMPLE_SIZE = 5
SAMPLE_SEED = 42


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    applicants = pd.read_csv(APPLICANTS_CSV)
    sample = applicants.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED)

    for _, applicant in sample.iterrows():
        applicant_id = applicant["Applicant_ID"]

        salary_path = OUTPUT_DIR / f"{applicant_id}_salary.pdf"
        salary_path.write_bytes(render_salary_slip(applicant))

        bank_path = OUTPUT_DIR / f"{applicant_id}_bank.xlsx"
        bank_path.write_bytes(render_bank_statement(applicant))

        print(f"{applicant_id}: wrote {salary_path.name}, {bank_path.name}")

    print(f"\nGenerated documents for {len(sample)} applicants in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
