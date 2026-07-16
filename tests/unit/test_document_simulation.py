import io

import openpyxl
import pandas as pd
import pdfplumber
import pytest

from loan_processing.document_simulation.bank_statement_xlsx import render_bank_statement
from loan_processing.document_simulation.salary_slip_pdf import render_salary_slip


@pytest.fixture
def applicant() -> pd.Series:
    return pd.Series(
        {
            "Applicant_ID": "APP00001",
            "Full_Name": "Jane Doe",
            "Employer_Name": "Acme Corp",
            "Age": 34,
            "Employment_Type": "Salaried",
            "Monthly_Net_Income": 75000.0,
            "Total_Existing_EMIs": 12000.0,
            "CIBIL_Score": 720,
            "Requested_Loan_Amount": 500000.0,
            "Requested_Tenure_Months": 36,
            "Average_Monthly_Bank_Balance": 150000.0,
            "Number_of_Bounced_Transactions_Last_6M": 0,
        }
    )


def test_salary_slip_contains_key_applicant_fields(applicant):
    pdf_bytes = render_salary_slip(applicant)

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    assert applicant["Applicant_ID"] in text
    assert str(int(applicant["Age"])) in text
    assert applicant["Employment_Type"] in text
    assert applicant["Employer_Name"] in text
    assert "Tax Deductions" in text
    assert f"{applicant['Monthly_Net_Income']:,.2f}" in text


def test_bank_statement_has_six_months_and_emi_debits(applicant):
    xlsx_bytes = render_bank_statement(applicant, seed=1)

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    ws = wb["Statement"]

    assert ws["B5"].value == applicant["Applicant_ID"]

    rows = list(ws.iter_rows(min_row=11, values_only=True))
    emi_rows = [r for r in rows if r[1] == "ACH DEBIT - EMI LOAN REPAYMENT"]
    salary_rows = [r for r in rows if "SALARY" in (r[1] or "")]

    assert len(emi_rows) == 6
    assert len(salary_rows) == 6
    assert all(r[2] == pytest.approx(applicant["Total_Existing_EMIs"]) for r in emi_rows)
    assert all(r[3] == pytest.approx(applicant["Monthly_Net_Income"]) for r in salary_rows)


def test_bank_statement_realized_average_is_close_to_target(applicant):
    xlsx_bytes = render_bank_statement(applicant, seed=2)

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    ws = wb["Statement"]
    realized_average = ws["B8"].value

    target = applicant["Average_Monthly_Bank_Balance"]
    assert abs(realized_average - target) / target < 0.01


def test_bank_statement_debit_and_credit_amounts_are_never_negative(applicant):
    xlsx_bytes = render_bank_statement(applicant, seed=3)

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    ws = wb["Statement"]
    rows = list(ws.iter_rows(min_row=11, values_only=True))

    for _, _, debit, credit, _ in rows:
        assert debit is None or debit >= 0
        assert credit is None or credit >= 0
