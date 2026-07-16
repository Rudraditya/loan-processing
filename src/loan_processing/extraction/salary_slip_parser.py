"""Parses a mock salary-slip PDF back into structured fields.

Regexes are anchored to the exact labels salary_slip_pdf.py writes
(`Applicant ID:`, `Age:`, `Employment Type:`, `Net Pay:`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

_PATTERNS = {
    "applicant_id": r"Applicant ID:\s*(\S+)",
    "age": r"Age:\s*(\d+)",
    "employment_type": r"Employment Type:\s*(.+)",
    "monthly_net_income": r"Net Pay:\s*Rs\.\s*([\d,]+\.\d{2})",
}


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def parse_salary_slip(pdf_path: str | Path) -> dict:
    text = extract_text_from_pdf(pdf_path)

    fields: dict[str, str] = {}
    for name, pattern in _PATTERNS.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Could not find '{name}' in salary slip: {pdf_path}")
        fields[name] = match.group(1).strip()

    return {
        "Applicant_ID": fields["applicant_id"],
        "Age": int(fields["age"]),
        "Employment_Type": fields["employment_type"],
        "Monthly_Net_Income": float(fields["monthly_net_income"].replace(",", "")),
    }
