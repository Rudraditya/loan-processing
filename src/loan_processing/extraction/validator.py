"""Deterministic post-extraction consistency checks.

These are soft signals only - appended to a record's discrepancy_flags for
the caller to see, never a hard rejection. This mirrors agentic_api/main.py's
existing hard-vs-soft convention: an applicant-name mismatch is treated as
likely identity fraud and hard-rejects (HTTP 422), but an extraction-quality
signal like an arithmetic mismatch between two extracted figures does not -
the record still proceeds, same as the existing bank-balance consistency
check (Bank_Consistency_Verified / Consistency_Flag).
"""
from __future__ import annotations

import math


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def compute_discrepancy_flags(record: dict) -> list[str]:
    flags: list[str] = []

    gross = record.get("Gross_Income")
    deductions = record.get("Total_Deductions")
    net = record.get("Monthly_Net_Income")
    if not any(is_missing(v) for v in (gross, deductions, net)):
        expected_net = gross - deductions
        tolerance = max(1.0, 0.005 * net)
        if abs(expected_net - net) > tolerance:
            flags.append(
                f"Gross Income (Rs. {gross:,.2f}) minus Total Deductions (Rs. {deductions:,.2f}) "
                f"doesn't reconcile with Monthly Net Income (Rs. {net:,.2f})."
            )

    trend = record.get("Cash_Flow_Trend")
    if trend is not None and len(trend) != 6:
        flags.append(f"Cash flow trend has {len(trend)} month(s) of data instead of the expected 6.")

    return flags
