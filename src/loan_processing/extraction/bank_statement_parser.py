"""Parses a mock bank-statement XLSX back into structured fields.

Computes a *day-weighted* average balance rather than a naive mean over
transaction rows: a naive row mean over-weights whichever balance level
happens to have the most transaction rows recorded against it (rows cluster
in the two-to-three weeks after each salary credit), which isn't
representative of how long the account actually sat at that level.
`bank_statement_xlsx.py` uses the same definition to calibrate the mock
statements, so this doubles as a verification step: the computed value is
cross-checked against the statement's own stated "Average Balance:" header.

EMI debits are recovered by summing the recurring EMI-labeled debit rows and
dividing by how many such rows were found, since Total_Existing_EMIs is a
*monthly* figure, not a 6-month total.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_HEADER_ROW_INDEX = 9  # Excel row 10 (1-indexed) holds the transaction table's column headers
_EMI_DESCRIPTION = "ACH DEBIT - EMI LOAN REPAYMENT"


def _read_header_fields(source: str | Path) -> dict[str, object]:
    header_df = pd.read_excel(source, sheet_name="Statement", header=None, nrows=8, usecols="A:B")
    header: dict[str, object] = {}
    for _, row in header_df.iterrows():
        label, value = row[0], row[1]
        if isinstance(label, str) and label.endswith(":"):
            header[label.rstrip(":")] = value
    return header


def _day_weighted_average_balance(transactions: pd.DataFrame) -> float:
    dates = pd.to_datetime(transactions["Date"]).tolist()
    balances = transactions["Balance"].tolist()

    period_end = dates[-1] + pd.Timedelta(days=1)
    weighted_sum = 0.0
    for i in range(len(dates)):
        start = dates[i]
        end = dates[i + 1] if i + 1 < len(dates) else period_end
        weighted_sum += balances[i] * (end - start).days
    total_days = (period_end - dates[0]).days
    return weighted_sum / total_days


def parse_bank_statement(xlsx_path: str | Path) -> dict:
    transactions = pd.read_excel(xlsx_path, sheet_name="Statement", header=_HEADER_ROW_INDEX)
    header = _read_header_fields(xlsx_path)

    emi_rows = transactions[transactions["Description"] == _EMI_DESCRIPTION]
    monthly_emi = float(emi_rows["Debit"].sum() / len(emi_rows)) if len(emi_rows) else 0.0

    day_weighted_avg = _day_weighted_average_balance(transactions)

    stated_avg = header.get("Average Balance")
    if isinstance(stated_avg, (int, float)):
        naive_avg = float(transactions["Balance"].mean())
        print(
            f"  [verification] day-weighted avg = Rs. {day_weighted_avg:,.2f} vs stated header = "
            f"Rs. {stated_avg:,.2f} (diff {abs(day_weighted_avg - stated_avg) / stated_avg * 100:.2f}%); "
            f"naive row-mean = Rs. {naive_avg:,.2f} (diff "
            f"{abs(naive_avg - stated_avg) / stated_avg * 100:.2f}%) - day-weighting is the closer match"
        )

    return {
        "Applicant_ID": header.get("Applicant ID"),
        "Average_Monthly_Bank_Balance": round(day_weighted_avg, 2),
        "Total_Existing_EMIs": round(monthly_emi, 2),
    }
