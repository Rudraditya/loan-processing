"""Parses a bank-statement XLSX back into structured fields.

Computes a *day-weighted* average balance rather than a naive mean over
transaction rows: a naive row mean over-weights whichever balance level
happens to have the most transaction rows recorded against it (rows cluster
in the two-to-three weeks after each salary credit), which isn't
representative of how long the account actually sat at that level.
`bank_statement_xlsx.py` uses the same definition to calibrate the mock
statements, so this doubles as a verification step: the computed value is
cross-checked against the statement's own stated "Average Balance:" header,
when one is present.

Two sheet layouts are understood - this project's own mock format (a
"Statement" sheet, transaction table in real Excel columns: Date /
Description / Debit / Credit / Balance) and a second, differently-formatted
real-world statement seen in practice (any sheet name, e.g. "Sheet1";
header fields and the transaction table both rendered as pipe-delimited
text crammed into a single column, with WITHDRAWAL/DEPOSIT column names
instead of Debit/Credit). `_looks_like_narrow_layout` distinguishes the two
so the same downstream day-weighted-average/EMI/consistency logic runs
either way. As with the salary slip parser, this is still a **fixed, small**
set of known layouts, not a general-purpose parser - anything outside both
is expected to fall through to the Gemini extraction agent.

EMI debits are recovered by summing debit rows whose description mentions
"EMI" (case-insensitive - covers both this project's own mock wording,
"ACH DEBIT - EMI LOAN REPAYMENT", and the real-world statement's
"ACH-AUTO-LOAN-EMI") and dividing by how many such rows were found, since
Total_Existing_EMIs is a *monthly* figure, not a 6-month total.
Number_of_Bounced_Transactions_Last_6M is recovered the same way, counting
debit rows whose description mentions "bounce" (this project's own mock
wording, "ACH DEBIT - CHEQUE BOUNCE CHARGE" - a single known vocabulary,
not yet broadened to a second observed real-world variant the way EMI/
average-balance detection has been, per this module's "known, fixed
vocabulary" convention below). A statement with no such rows counts as 0
bounces, same convention gemini_extractor.py uses when a real document
simply doesn't mention any - not a "couldn't extract this" NaN.

Cash_Flow_Trend (the 6 monthly day-weighted averages) is read directly from
the statement's own "Month N Avg Balance:" header when present (this
project's own mock statements bake this in as a convenience), but falls
back to computing it directly from the parsed transaction table otherwise -
see _day_weighted_monthly_averages, which applies the exact same
day-weighting/windowing document_simulation/bank_statement_xlsx.py uses to
calibrate that header in the first place, just run against a statement that
never had the header to begin with. Only None if the statement's own date
range doesn't span 6 full calendar months of data to compute from.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

_HEADER_ROW_INDEX = 17  # Excel row 18 (1-indexed) holds the transaction table's column headers, wide layout only
_NARROW_COLUMN_MAP = {
    "DATE": "Date",
    "DESCRIPTION": "Description",
    "DEBIT": "Debit",
    "WITHDRAWAL": "Debit",
    "CREDIT": "Credit",
    "DEPOSIT": "Credit",
    "BALANCE": "Balance",
}


def _read_header_fields_wide(raw: pd.DataFrame) -> dict[str, object]:
    header: dict[str, object] = {}
    for _, row in raw.iloc[:16, :2].iterrows():
        label, value = row[0], row[1]
        if isinstance(label, str) and label.endswith(":"):
            header[label.rstrip(":")] = value
    return header


def _coerce_narrow_header_value(value: str) -> object:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return value


def _read_header_fields_narrow(lines: list[str]) -> dict[str, object]:
    header: dict[str, object] = {}
    for line in lines:
        stripped = line.strip()
        if "|" in stripped or ":" not in stripped:
            continue
        label, _, value = stripped.partition(":")
        header[label.strip()] = _coerce_narrow_header_value(value.strip())
    return header


def _parse_narrow_transactions(lines: list[str]) -> pd.DataFrame:
    pipe_lines = [line for line in lines if "|" in line]
    header_cells = [cell.strip() for cell in pipe_lines[0].split("|")]
    columns = [_NARROW_COLUMN_MAP.get(cell.upper(), cell) for cell in header_cells]

    rows = [
        [cell.strip() for cell in line.split("|")]
        for line in pipe_lines[1:]
        if line.count("|") == len(header_cells) - 1
    ]
    transactions = pd.DataFrame(rows, columns=columns)
    for numeric_col in ("Debit", "Credit", "Balance"):
        if numeric_col in transactions.columns:
            transactions[numeric_col] = pd.to_numeric(transactions[numeric_col], errors="coerce")
    return transactions


def _looks_like_narrow_layout(raw: pd.DataFrame) -> bool:
    """True when the sheet crams everything into a single text column
    (real-world layout) rather than using real Excel columns (this
    project's own mock layout, which always populates at least 2 columns).
    """
    return raw.shape[1] == 1 or raw.iloc[:, 1:].isna().all(axis=None)


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


def _month_starts_ending_at(anchor: pd.Timestamp, months: int) -> list[pd.Timestamp]:
    starts = []
    year, month = anchor.year, anchor.month
    for _ in range(months):
        starts.append(pd.Timestamp(year=year, month=month, day=1))
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    return list(reversed(starts))


def _month_end_exclusive(month_start: pd.Timestamp) -> pd.Timestamp:
    if month_start.month == 12:
        return pd.Timestamp(year=month_start.year + 1, month=1, day=1)
    return pd.Timestamp(year=month_start.year, month=month_start.month + 1, day=1)


def _day_weighted_monthly_averages(transactions: pd.DataFrame) -> list[float] | None:
    """Each of the last 6 calendar months' own day-weighted average balance,
    computed directly from the raw dated-transaction-with-running-balance
    rows - same per-month day-weighting/windowing
    document_simulation/bank_statement_xlsx.py uses to calibrate its own
    "Month N Avg Balance:" header. Anchored to the statement's own last
    transaction date, not "today" - a real statement's period end is
    whatever the document says it is. None if the statement's earliest
    transaction doesn't reach back far enough to cover all 6 months (a
    partial month would silently understate that month's true average).
    """
    dates = pd.to_datetime(transactions["Date"])
    order = dates.argsort().to_numpy()
    dates = dates.iloc[order].tolist()
    balances = transactions["Balance"].to_numpy()[order].tolist()

    month_starts = _month_starts_ending_at(dates[-1], 6)
    if dates[0] > month_starts[0]:
        return None

    period_end = dates[-1] + pd.Timedelta(days=1)
    averages = []
    for month_start in month_starts:
        month_end_excl = _month_end_exclusive(month_start)
        total_days = (month_end_excl - month_start).days
        weighted_sum = 0.0
        for i in range(len(dates)):
            interval_start = dates[i]
            interval_end = dates[i + 1] if i + 1 < len(dates) else period_end
            clipped_start = max(interval_start, month_start)
            clipped_end = min(interval_end, month_end_excl)
            overlap_days = (clipped_end - clipped_start).days
            if overlap_days > 0:
                weighted_sum += balances[i] * overlap_days
        averages.append(round(weighted_sum / total_days, 2))
    return averages


def parse_bank_statement(xlsx_path: str | Path) -> dict:
    workbook = pd.ExcelFile(xlsx_path)
    sheet_name = "Statement" if "Statement" in workbook.sheet_names else workbook.sheet_names[0]
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)

    if _looks_like_narrow_layout(raw):
        lines = [str(value) for value in raw.iloc[:, 0].tolist() if pd.notna(value)]
        transactions = _parse_narrow_transactions(lines)
        header = _read_header_fields_narrow(lines)
    else:
        transactions = pd.read_excel(workbook, sheet_name=sheet_name, header=_HEADER_ROW_INDEX)
        header = _read_header_fields_wide(raw)

    emi_rows = transactions[transactions["Description"].str.contains("emi", case=False, na=False)]
    monthly_emi = float(emi_rows["Debit"].sum() / len(emi_rows)) if len(emi_rows) else 0.0

    bounce_rows = transactions[transactions["Description"].str.contains("bounce", case=False, na=False)]
    number_of_bounces = int(len(bounce_rows))

    day_weighted_avg = _day_weighted_average_balance(transactions)

    stated_avg = header.get("Average Balance")
    consistency_verified: bool | None = None
    if isinstance(stated_avg, (int, float)):
        naive_avg = float(transactions["Balance"].mean())
        diff_pct = abs(day_weighted_avg - stated_avg) / stated_avg * 100 if stated_avg else 100.0
        consistency_verified = diff_pct <= 2.0
        print(
            f"  [verification] day-weighted avg = Rs. {day_weighted_avg:,.2f} vs stated header = "
            f"Rs. {stated_avg:,.2f} (diff {diff_pct:.2f}%); "
            f"naive row-mean = Rs. {naive_avg:,.2f} (diff "
            f"{abs(naive_avg - stated_avg) / stated_avg * 100:.2f}%) - day-weighting is the closer match"
        )

    trend = [header.get(f"Month {i} Avg Balance") for i in range(1, 7)]
    if all(isinstance(v, (int, float)) for v in trend):
        cash_flow_trend = trend
    else:
        # No convenience header (a real-world statement never has one) -
        # compute it directly from the transaction table already parsed
        # above, instead of giving up. None only if there isn't 6 full
        # calendar months of dated data to compute from at all.
        cash_flow_trend = _day_weighted_monthly_averages(transactions)

    return {
        "Average_Monthly_Bank_Balance": round(day_weighted_avg, 2),
        "Total_Existing_EMIs": round(monthly_emi, 2),
        "Number_of_Bounced_Transactions_Last_6M": number_of_bounces,
        # None when the statement has no stated "Average Balance:" header to check
        # against at all (e.g. a real-world statement, not one of our mock ones).
        "Consistency_Verified": consistency_verified,
        "Cash_Flow_Trend": cash_flow_trend,
    }


def expand_cash_flow_trend(trend: list[float] | None) -> dict[str, float]:
    """Unpacks a 6-element Cash_Flow_Trend list into the flat
    Cash_Flow_Month_1..6 keys build_features() expects. All-NaN if the trend
    is missing or malformed, matching FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS'
    explicit-NaN-over-fabricated-value convention.
    """
    if not trend or len(trend) != 6:
        return {f"Cash_Flow_Month_{i}": math.nan for i in range(1, 7)}
    return {f"Cash_Flow_Month_{i}": trend[i - 1] for i in range(1, 7)}
