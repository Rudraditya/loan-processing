"""Parses a salary-slip PDF back into structured fields.

Regexes are anchored to the labels salary_slip_pdf.py writes (`Employee
Name:`, `Net Pay:`, `Gross Pay:`, `Tax Deductions:`, each with an "Rs."
currency prefix) plus a second accepted vocabulary (`Gross Income:`, `Total
Deductions:`, no currency prefix required) covering a differently-formatted
real-world salary slip seen in practice - matching is case-insensitive so
an ALL-CAPS layout ("EMPLOYEE NAME:") works the same as this project's own
mixed-case mock format. This is intentionally still a **fixed, small**
vocabulary, not a general-purpose parser: it's meant to absorb label
variations actually observed, not every conceivable format. A salary slip
using label text outside both vocabularies is expected to fall through to
the Gemini extraction agent (gemini_extractor.py), which reads the
document's visual layout rather than matching literal strings.

Applicant_ID and Age are deliberately not recovered here: Applicant_ID is
just a serial number printed on the document for human reference, and Age
is unreliable to trust from a static document (it goes stale the moment it
ages past the document's issue date) - both are caller-supplied instead
(see agentic_api/main.py's `applicant_id`/`date_of_birth` form fields and
record_builder.build_applicant_record's `applicant_id`/`age` parameters;
Age is derived from `date_of_birth` as-of today's date, not typed in
directly). Employment_Type is likewise no longer extracted here - it's a
loan-application-form field now (see agentic_api/main.py's
`employment_status` form field), not something read off the document,
since a real applicant's own self-reported document label isn't a reliable
substitute for what they actually select on the form.
"""
from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

_PATTERNS = {
    "full_name": r"Employee Name:\s*(.+)",
    "monthly_net_income": r"Net Pay:\s*(?:Rs\.\s*)?([\d,]+\.\d{2})",
    "gross_income": r"Gross (?:Pay|Income):\s*(?:Rs\.\s*)?([\d,]+\.\d{2})",
    "total_deductions": r"(?:Tax|Total) Deductions:\s*(?:Rs\.\s*)?([\d,]+\.\d{2})",
}


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def parse_salary_slip(pdf_path: str | Path) -> dict:
    text = extract_text_from_pdf(pdf_path)

    fields: dict[str, str] = {}
    for name, pattern in _PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not find '{name}' in salary slip: {pdf_path}")
        fields[name] = match.group(1).strip()

    return {
        "Full_Name": fields["full_name"],
        "Monthly_Net_Income": float(fields["monthly_net_income"].replace(",", "")),
        "Gross_Income": float(fields["gross_income"].replace(",", "")),
        "Total_Deductions": float(fields["total_deductions"].replace(",", "")),
    }
