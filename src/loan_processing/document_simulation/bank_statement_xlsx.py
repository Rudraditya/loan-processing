"""Renders a mock bank-statement XLSX for a synthetic applicant record.

Simulates 6 months of transactions: a monthly salary/business-income credit,
a monthly EMI debit equal to the applicant's total EMI figure, and several
smaller randomized discretionary debits. The account's opening balance is
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

    balance = 0.0
    rows: list[dict] = []
    month_starts = _month_starts(date.today().replace(day=1), MONTHS_OF_HISTORY)

    for month_start in month_starts:
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


def _simulate_transactions(applicant: pd.Series, seed: int) -> tuple[list[dict], date]:
    rows, period_end = _build_relative_transactions(applicant, seed)

    target_avg = float(applicant["Average_Monthly_Bank_Balance"])
    offset = target_avg - _day_weighted_average(rows, period_end)
    for row in rows:
        row["Balance"] = round(row["Balance"] + offset, 2)

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

    header_row = 10
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
