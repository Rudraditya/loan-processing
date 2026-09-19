"""Persists HITL review decisions to a local Excel summary sheet.

Append-only log of every applicant that reached a final HITL decision
(Quick Approve, or Approve/Reject from the Document Inspection Drawer),
written to data/applicants_summary.xlsx at the repo root - the same
directory convention as the rest of the pipeline's committed artifacts
(see CLAUDE.md's data_generation/document_simulation/risk_modeling stages).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# excel_manager.py lives at <repo_root>/src/loan_processing/agentic_api/ -
# same depth as main.py, which derives REPO_ROOT the same way.
REPO_ROOT = Path(__file__).resolve().parents[3]
EXCEL_PATH = REPO_ROOT / "data" / "applicants_summary.xlsx"

SUMMARY_COLUMNS = [
    "Applicant_ID",
    "Employment_Type",
    "Monthly_Net_Income",
    "Requested_Loan_Amount",
    "Default_Probability",
    "Classification_Verdict",
    "Final_Status",
]


def append_to_excel(applicant_data: dict, excel_path: Path = EXCEL_PATH) -> None:
    """Appends one applicant row to the Excel summary, creating the file
    (and its parent directory) on first use. Re-reads and overwrites the
    whole file each call rather than a true incremental append, since xlsx
    has no native append mode - fine at this log's expected scale.
    """
    row = {column: applicant_data.get(column) for column in SUMMARY_COLUMNS}

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    if excel_path.exists():
        existing = pd.read_excel(excel_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row], columns=SUMMARY_COLUMNS)

    updated.to_excel(excel_path, index=False)
