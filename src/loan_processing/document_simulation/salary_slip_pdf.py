"""Renders a mock salary-slip PDF for a synthetic applicant record.

Labels are chosen to double as a forward-compatible fixture for the
extraction layer (`Employer:`, `Gross Pay:`, `Net Pay:`, `Pay Period:` mirror
`extraction/extractor.py`'s existing PAY_STUB field patterns).
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

TAX_DEDUCTION_RATE = 0.22


def render_salary_slip(applicant: pd.Series, statement_month: date | None = None) -> bytes:
    statement_month = statement_month or date.today().replace(day=1)
    net_pay = float(applicant["Monthly_Net_Income"])
    gross_pay = net_pay / (1 - TAX_DEDUCTION_RATE)
    tax_deduction = gross_pay - net_pay

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    left = 20 * mm
    y = height - 25 * mm

    def line(text: str, size: int = 10, gap: float = 7 * mm, bold: bool = False) -> None:
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(left, y, text)
        y -= gap

    is_salaried = applicant["Employment_Type"] == "Salaried"

    line(str(applicant["Employer_Name"]), size=14, bold=True)
    line("Payroll Department | HR-Compliance Office", size=9)
    line("-" * 95, size=9, gap=6 * mm)
    line("SALARY SLIP" if is_salaried else "INCOME STATEMENT", size=12, bold=True)
    line(f"Pay Period: {statement_month.strftime('%B %Y')} (Monthly)")
    line("")
    line(f"Applicant ID: {applicant['Applicant_ID']}")
    line(f"Employee Name: {applicant['Full_Name']}")
    line(f"Age: {int(applicant['Age'])}")
    line(f"Employment Type: {applicant['Employment_Type']}")
    line(f"Employee Code: EMP-{abs(hash(applicant['Applicant_ID'])) % 90000 + 10000}")
    line("")
    line("Earnings", size=11, bold=True)
    line(f"Gross Pay: Rs. {gross_pay:,.2f}")
    line(f"Tax Deductions: Rs. {tax_deduction:,.2f}")
    line(f"Net Pay: Rs. {net_pay:,.2f}")
    line("")
    line(
        "This is a system-generated document issued for the sole purpose of income",
        size=8,
        gap=5 * mm,
    )
    line(
        "verification during the loan application process. Figures are computed on a",
        size=8,
        gap=5 * mm,
    )
    line("monthly basis and are subject to statutory deductions as per applicable tax law.", size=8)

    c.showPage()
    c.save()
    return buffer.getvalue()
