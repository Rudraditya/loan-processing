"""Renders a mock bank-statement XLSX for a synthetic applicant record.

Simulates 6 months of transactions: a monthly salary/business-income credit,
a monthly EMI debit equal to the applicant's total EMI figure, several
smaller randomized discretionary debits, and the applicant's own
Number_of_Bounced_Transactions_Last_6M count as "CHEQUE BOUNCE CHARGE" debit
rows scattered across the 6 months - bank_statement_parser.py counts these
back out by the same "bounce" substring. The account's opening balance is
then solved for so that the statement's *day-weighted* average balance
(the standard definition of an average monthly balance — a plain average
over transaction events would overweight the days right after each salary
credit, since income lands early in the month and spending trickles out
over the following weeks) lands exactly on the applicant's stored
Average_Monthly_Bank_Balance.
"""
from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

MONTHS_OF_HISTORY = 6
_DISCRETIONARY_CATEGORIES = [
    "POS DEBIT - GROCERY STORE",
    "UPI DEBIT - UTILITY BILL",
    "ATM WITHDRAWAL",
    "UPI DEBIT - ONLINE SHOPPING",
    "AUTO-DEBIT - SUBSCRIPTION",
    "POS DEBIT - RESTAURANT",
    "UPI DEBIT - FUEL STATION",
    "NEFT DEBIT - RENT",
]


def _month_starts(anchor: date, months: int) -> list[date]:
    starts = []
    year, month = anchor.year, anchor.month
    for _ in range(months):
        starts.append(date(year, month, 1))
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    return list(reversed(starts))


def _month_end(month_start: date) -> date:
    if month_start.month == 12:
        next_month_start = date(month_start.year + 1, 1, 1)
    else:
        next_month_start = date(month_start.year, month_start.month + 1, 1)
    return next_month_start - timedelta(days=1)


def _build_relative_transactions(applicant: pd.Series, seed: int) -> tuple[list[dict], date]:
    """Builds transactions against a relative balance starting at 0; returns (rows, period_end).

    Interest is calculated off the applicant's target average balance rather
    than the running relative balance: the relative balance is just a
    bookkeeping intermediate (pre-calibration-offset) and can dip negative,
    which would otherwise produce a nonsensical negative "interest credit".
    """
    rng = np.random.default_rng(seed)
    income = float(applicant["Monthly_Net_Income"])
    emi = float(applicant["Total_Existing_EMIs"])
    target_avg = float(applicant["Average_Monthly_Bank_Balance"])
    employer = applicant["Employer_Name"]
    is_salaried = applicant["Employment_Type"] == "Salaried"
    number_of_bounces = int(applicant["Number_of_Bounced_Transactions_Last_6M"])

    balance = 0.0
    rows: list[dict] = []
    month_starts = _month_starts(date.today().replace(day=1), MONTHS_OF_HISTORY)
    # Which of the 6 months (by index) each bounced-transaction charge lands
    # in - drawn once up front so a count above 6 can still land more than
    # one charge in the same month. "bounce" is the fixed vocabulary
    # bank_statement_parser.py's regex counts on - see its docstring.
    bounce_month_indices = rng.integers(0, MONTHS_OF_HISTORY, size=number_of_bounces)

    # A synthetic opening-balance row dated at the very start of the history
    # window - without it, the days between month_starts[0] and the first
    # real transaction have no row to inherit a balance from at all, which
    # would silently dilute that first month's day-weighted average (there's
    # nothing "before" it to carry forward, unlike every later month, which
    # inherits from the previous month's last transaction).
    rows.append(
        {
            "Date": month_starts[0],
            "Description": "OPENING BALANCE",
            "Debit": None,
            "Credit": None,
            "Balance": balance,
        }
    )

    for month_index, month_start in enumerate(month_starts):
        income_day = month_start + timedelta(days=int(rng.integers(0, 3)))
        balance += income
        income_label = "SALARY" if is_salaried else "BUSINESS INCOME"
        rows.append(
            {
                "Date": income_day,
                "Description": f"NEFT CREDIT - {employer} {income_label}",
                "Debit": None,
                "Credit": round(income, 2),
                "Balance": balance,
            }
        )

        if emi > 0:
            emi_day = month_start + timedelta(days=int(rng.integers(4, 8)))
            balance -= emi
            rows.append(
                {
                    "Date": emi_day,
                    "Description": "ACH DEBIT - EMI LOAN REPAYMENT",
                    "Debit": round(emi, 2),
                    "Credit": None,
                    "Balance": balance,
                }
            )

        disposable = max(income - emi, 0.0)
        discretionary_target = disposable * rng.uniform(0.90, 1.10)
        n_discretionary = int(rng.integers(5, 9))
        shares = rng.dirichlet(np.ones(n_discretionary)) * discretionary_target
        day_offsets = sorted(rng.integers(8, 27, size=n_discretionary))
        categories = rng.choice(_DISCRETIONARY_CATEGORIES, size=n_discretionary, replace=False)

        for amount, day_offset, category in zip(shares, day_offsets, categories):
            balance -= amount
            rows.append(
                {
                    "Date": month_start + timedelta(days=int(day_offset)),
                    "Description": category,
                    "Debit": round(float(amount), 2),
                    "Credit": None,
                    "Balance": balance,
                }
            )

        bounces_this_month = int(np.sum(bounce_month_indices == month_index))
        for _ in range(bounces_this_month):
            bounce_fee = round(float(rng.uniform(300, 750)), 2)  # typical Indian bank return/bounce charge
            balance -= bounce_fee
            rows.append(
                {
                    "Date": month_start + timedelta(days=int(rng.integers(1, 26))),
                    "Description": "ACH DEBIT - CHEQUE BOUNCE CHARGE",
                    "Debit": bounce_fee,
                    "Credit": None,
                    "Balance": balance,
                }
            )

        interest_credit = round(target_avg * 0.0025, 2)  # ~3% p.a. savings interest, credited monthly
        balance += interest_credit
        rows.append(
            {
                "Date": month_start + timedelta(days=28),
                "Description": "SAVINGS INTEREST CREDIT",
                "Debit": None,
                "Credit": interest_credit,
                "Balance": balance,
            }
        )

    rows.sort(key=lambda r: r["Date"])
    return rows, _month_end(month_starts[-1])


def _day_weighted_average(rows: list[dict], period_end: date) -> float:
    weighted_sum = 0.0
    for i, row in enumerate(rows):
        start_d = row["Date"]
        end_d = rows[i + 1]["Date"] if i + 1 < len(rows) else period_end + timedelta(days=1)
        weighted_sum += row["Balance"] * (end_d - start_d).days
    total_days = (period_end + timedelta(days=1) - rows[0]["Date"]).days
    return weighted_sum / total_days


def _day_weighted_average_window(
    rows: list[dict], period_end: date, window_start: date, window_end: date
) -> float:
    """Same day-weighting as `_day_weighted_average`, but restricted to a
    single [window_start, window_end] slice of the full transaction history -
    used to measure one month's own average out of the whole 6-month ledger.
    """
    window_end_excl = window_end + timedelta(days=1)
    total_days = (window_end_excl - window_start).days
    weighted_sum = 0.0
    for i, row in enumerate(rows):
        interval_start = row["Date"]
        interval_end = rows[i + 1]["Date"] if i + 1 < len(rows) else period_end + timedelta(days=1)
        clipped_start = max(interval_start, window_start)
        clipped_end = min(interval_end, window_end_excl)
        overlap_days = (clipped_end - clipped_start).days
        if overlap_days > 0:
            weighted_sum += row["Balance"] * overlap_days
    return weighted_sum / total_days


def _simulate_transactions(applicant: pd.Series, seed: int) -> tuple[list[dict], date]:
    """Builds the 6 months of transactions, then calibrates each month's own
    day-weighted average onto its own Cash_Flow_Month_N target - a
    generalization of the single-global-offset trick this function used to
    do (one target, one shift) into 6 sequential passes (one target, one
    shift, per month). Each month's shift is applied to that month's rows
    *and* every later row, since Balance is a running total and the shift
    must persist forward - otherwise the ledger would show an unexplained
    jump at the month boundary.
    """
    rows, period_end = _build_relative_transactions(applicant, seed)
    month_starts = _month_starts(date.today().replace(day=1), MONTHS_OF_HISTORY)

    for i, month_start in enumerate(month_starts, start=1):
        month_end = _month_end(month_start)
        target = float(applicant[f"Cash_Flow_Month_{i}"])
        current_avg = _day_weighted_average_window(rows, period_end, month_start, month_end)
        needed = target - current_avg
        for row in rows:
            if row["Date"] >= month_start:
                row["Balance"] = round(row["Balance"] + needed, 2)

    return rows, period_end


def render_bank_statement(applicant: pd.Series, seed: int | None = None) -> bytes:
    seed = seed if seed is not None else abs(hash(applicant["Applicant_ID"])) % (2**32)
    rows, period_end = _simulate_transactions(applicant, seed)

    closing_balance = rows[-1]["Balance"]
    realized_average_balance = round(_day_weighted_average(rows, period_end), 2)

    wb = Workbook()
    ws = wb.active
    ws.title = "Statement"
    bold = Font(bold=True)

    ws["A1"] = "MOCK NATIONAL BANK"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = "Statement of Account"

    header_fields = [
        ("A4", "Account Holder:", "B4", applicant["Full_Name"]),
        ("A5", "Applicant ID:", "B5", applicant["Applicant_ID"]),
        ("A6", "Statement Period:", "B6", f"{rows[0]['Date'].isoformat()} to {rows[-1]['Date'].isoformat()}"),
        ("A7", "Closing Balance:", "B7", closing_balance),
        ("A8", "Average Balance:", "B8", realized_average_balance),
    ]
    for label_cell, label, value_cell, value in header_fields:
        ws[label_cell] = label
        ws[label_cell].font = bold
        ws[value_cell] = value

    ws["A10"] = "Monthly Cash Flow Trend"
    ws["A10"].font = bold
    month_starts = _month_starts(date.today().replace(day=1), MONTHS_OF_HISTORY)
    for i, month_start in enumerate(month_starts, start=1):
        month_end = _month_end(month_start)
        realized_month_avg = round(
            _day_weighted_average_window(rows, period_end, month_start, month_end), 2
        )
        label_row = 10 + i
        ws[f"A{label_row}"] = f"Month {i} Avg Balance:"
        ws[f"A{label_row}"].font = bold
        ws[f"B{label_row}"] = realized_month_avg

    header_row = 18
    for col, heading in enumerate(["Date", "Description", "Debit", "Credit", "Balance"], start=1):
        cell = ws.cell(row=header_row, column=col, value=heading)
        cell.font = bold

    for offset, row in enumerate(rows, start=1):
        r = header_row + offset
        ws.cell(row=r, column=1, value=row["Date"])
        ws.cell(row=r, column=2, value=row["Description"])
        ws.cell(row=r, column=3, value=row["Debit"])
        ws.cell(row=r, column=4, value=row["Credit"])
        ws.cell(row=r, column=5, value=row["Balance"])

    for col_idx, width in enumerate([14, 34, 12, 12, 14], start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
