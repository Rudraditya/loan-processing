"""Renders a LoanApplicationResult as a multi-sheet Excel workbook."""
from __future__ import annotations

import io

import pandas as pd

from loan_processing.models import LoanApplicationResult


def write_excel(result: LoanApplicationResult) -> io.BytesIO:
    buffer = io.BytesIO()

    summary_df = pd.DataFrame(
        [
            {
                "application_id": result.application_id,
                "decision": result.decision.outcome.value,
                "reasons": "; ".join(result.decision.reasons),
                "score": result.score.score,
                "risk_tier": result.score.risk_tier,
                "is_valid": result.validation.is_valid,
            }
        ]
    )

    extracted_rows = [
        {"document": doc.source_filename, "doc_type": doc.doc_type.value, "field": field, "value": value}
        for doc in result.extracted
        for field, value in doc.fields.items()
    ]
    extracted_df = pd.DataFrame(extracted_rows, columns=["document", "doc_type", "field", "value"])

    issues_rows = [
        {"severity": issue.severity, "field": issue.field, "message": issue.message}
        for issue in result.validation.issues
    ]
    issues_df = pd.DataFrame(issues_rows, columns=["severity", "field", "message"])

    factors_rows = [{"factor": name, "value": value} for name, value in result.score.factors.items()]
    factors_df = pd.DataFrame(factors_rows, columns=["factor", "value"])

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        extracted_df.to_excel(writer, sheet_name="Extracted Data", index=False)
        issues_df.to_excel(writer, sheet_name="Validation Issues", index=False)
        factors_df.to_excel(writer, sheet_name="Score Factors", index=False)

    buffer.seek(0)
    return buffer
