import io
import math
from datetime import date

import pandas as pd
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from loan_processing.document_simulation.bank_statement_xlsx import render_bank_statement
from loan_processing.document_simulation.salary_slip_pdf import render_salary_slip
from loan_processing.extraction.bank_statement_parser import parse_bank_statement
from loan_processing.extraction.record_builder import (
    FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS,
    build_applicant_record,
    build_model_input,
    calculate_age,
)
from loan_processing.extraction.salary_slip_parser import parse_salary_slip
from loan_processing.risk_modeling.train import FEATURE_COLUMNS


def test_calculate_age_before_birthday_this_year():
    # Born 12 Apr; "today" is 20 Mar - birthday hasn't happened yet this year.
    assert calculate_age(date(2000, 4, 12), as_of=date(2026, 3, 20)) == 25


def test_calculate_age_on_or_after_birthday_this_year():
    # Same birth date; "today" is on/after 12 Apr - birthday already happened.
    assert calculate_age(date(2000, 4, 12), as_of=date(2026, 4, 12)) == 26
    assert calculate_age(date(2000, 4, 12), as_of=date(2026, 7, 20)) == 26


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
            "Gross_Income": 96153.85,
            "Total_Deductions": 21153.85,
            "Total_Existing_EMIs": 12000.0,
            "CIBIL_Score": 720,
            "Requested_Loan_Amount": 500000.0,
            "Requested_Tenure_Months": 36,
            "Average_Monthly_Bank_Balance": 150000.0,
            "Number_of_Bounced_Transactions_Last_6M": 0,
            "Cash_Flow_Month_1": 145000.0,
            "Cash_Flow_Month_2": 147000.0,
            "Cash_Flow_Month_3": 149000.0,
            "Cash_Flow_Month_4": 151000.0,
            "Cash_Flow_Month_5": 153000.0,
            "Cash_Flow_Month_6": 155000.0,
        }
    )


@pytest.fixture
def salary_slip_path(tmp_path, applicant):
    path = tmp_path / f"{applicant['Applicant_ID']}_salary.pdf"
    path.write_bytes(render_salary_slip(applicant))
    return path


@pytest.fixture
def bank_statement_path(tmp_path, applicant):
    path = tmp_path / f"{applicant['Applicant_ID']}_bank.xlsx"
    path.write_bytes(render_bank_statement(applicant, seed=1))
    return path


def test_parse_salary_slip_recovers_original_fields(salary_slip_path, applicant):
    parsed = parse_salary_slip(salary_slip_path)

    assert parsed["Full_Name"] == applicant["Full_Name"]
    assert "Age" not in parsed  # caller-supplied now, not document-extracted - see salary_slip_parser.py
    assert "Employment_Type" not in parsed  # caller-supplied form field now, not document-extracted
    assert parsed["Monthly_Net_Income"] == pytest.approx(applicant["Monthly_Net_Income"])
    assert parsed["Gross_Income"] == pytest.approx(applicant["Gross_Income"])
    assert parsed["Total_Deductions"] == pytest.approx(applicant["Total_Deductions"])


def test_parse_bank_statement_recovers_emi_and_average_balance(bank_statement_path, applicant):
    parsed = parse_bank_statement(bank_statement_path)

    assert parsed["Total_Existing_EMIs"] == pytest.approx(applicant["Total_Existing_EMIs"], rel=0.01)
    assert parsed["Average_Monthly_Bank_Balance"] == pytest.approx(
        applicant["Average_Monthly_Bank_Balance"], rel=0.01
    )
    assert len(parsed["Cash_Flow_Trend"]) == 6
    for i, value in enumerate(parsed["Cash_Flow_Trend"], start=1):
        assert value == pytest.approx(applicant[f"Cash_Flow_Month_{i}"], rel=0.03)


@pytest.fixture
def alternate_format_salary_slip_path(tmp_path) -> object:
    """A real-world salary slip using ALL-CAPS labels, no 'Rs.' currency
    prefix, and different label wording ('Gross Income'/'Total Deductions'
    instead of 'Gross Pay'/'Tax Deductions') - the second accepted
    vocabulary salary_slip_parser.py's _PATTERNS was broadened to cover.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.drawString(50, 800, "COMPANY NAME: Tata Power")
    c.drawString(50, 780, "EMPLOYEE NAME: Rahul Verma")
    c.drawString(50, 760, "GROSS INCOME: 162500.00")
    c.drawString(50, 740, "TOTAL DEDUCTIONS: 27500.00")
    c.drawString(50, 720, "NET PAY: 135000.00")
    c.showPage()
    c.save()

    path = tmp_path / "alternate_salary.pdf"
    path.write_bytes(buffer.getvalue())
    return path


def test_parse_salary_slip_accepts_alternate_all_caps_format(alternate_format_salary_slip_path):
    parsed = parse_salary_slip(alternate_format_salary_slip_path)

    assert parsed["Full_Name"] == "Rahul Verma"
    assert parsed["Monthly_Net_Income"] == pytest.approx(135000.0)
    assert parsed["Gross_Income"] == pytest.approx(162500.0)
    assert parsed["Total_Deductions"] == pytest.approx(27500.0)


@pytest.fixture
def alternate_format_bank_statement_path(tmp_path) -> object:
    """A real-world bank statement rendered as pipe-delimited text crammed
    into a single column of a sheet not named "Statement" (WITHDRAWAL/
    DEPOSIT instead of Debit/Credit) - the second accepted layout
    bank_statement_parser.py's parse_bank_statement() was broadened to
    cover, on top of this project's own mock (multi-column, "Statement"
    sheet) layout.
    """
    lines = [
        "BANK NAME: State Bank of India",
        "ACCOUNT NAME: Rahul Verma",
        "ACCOUNT NUMBER: 30998273645",
        "STATEMENT START: 2026-01-01",
        "STATEMENT END: 2026-02-28",
        "OPENING BALANCE: 35000.00",
        None,
        "DATE       | REF_NO     | DESCRIPTION                  | WITHDRAWAL | DEPOSIT   | BALANCE",
        "-" * 91,
        "2026-01-01 | TXN-001    | B/F OPENING BALANCE          | 0.00       | 0.00      | 35000.00",
        "2026-01-02 | SAL-011    | NEFT-SALARY-DEC              | 0.00       | 135000.00 | 170000.00",
        "2026-01-05 | ACH-031    | ACH-AUTO-LOAN-EMI            | 10000.00   | 0.00      | 160000.00",
        "2026-01-10 | UPI-041    | UPI-GROCERIES-DMART          | 8000.00    | 0.00      | 152000.00",
    ]
    path = tmp_path / "alternate_bank.xlsx"
    pd.DataFrame({"col": lines}).to_excel(path, sheet_name="Sheet1", index=False, header=False)
    return path


def test_parse_bank_statement_accepts_alternate_pipe_delimited_format(alternate_format_bank_statement_path):
    parsed = parse_bank_statement(alternate_format_bank_statement_path)

    assert parsed["Total_Existing_EMIs"] == pytest.approx(10000.0)
    assert parsed["Average_Monthly_Bank_Balance"] > 0
    # This layout has no stated "Average Balance:"/"Month N Avg Balance:"
    # header fields to cross-check against or read a trend from - None here
    # is the correct, documented behavior for a statement lacking them, not
    # a parsing failure.
    assert parsed["Consistency_Verified"] is None
    assert parsed["Cash_Flow_Trend"] is None


def test_build_applicant_record_combines_both_documents(salary_slip_path, bank_statement_path, applicant):
    record = build_applicant_record(
        salary_slip_path,
        bank_statement_path,
        applicant["Applicant_ID"],
        applicant["Age"],
        applicant["Employment_Type"],
    )

    # Applicant_ID and Age are caller-assigned/supplied, not recovered from
    # the documents - they should come back exactly as passed in.
    assert record["Applicant_ID"] == applicant["Applicant_ID"]
    assert record["Age"] == applicant["Age"]
    assert record["Employment_Type"] == applicant["Employment_Type"]
    for field in FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS:
        assert math.isnan(record[field])
    assert not math.isnan(record["Gross_Income"])
    assert not math.isnan(record["Total_Deductions"])
    for i in range(1, 7):
        assert not math.isnan(record[f"Cash_Flow_Month_{i}"])
    # Number_of_Bounced_Transactions_Last_6M used to be permanently NaN too
    # (the mock statement never rendered bounce line items) - it's now a
    # real document-derived field, no longer in
    # FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS. The fixture applicant has 0
    # bounces, so the recovered count should be exactly 0, not NaN.
    assert record["Number_of_Bounced_Transactions_Last_6M"] == 0


def test_build_model_input_matches_saved_model_schema(salary_slip_path, bank_statement_path, applicant):
    record = build_applicant_record(
        salary_slip_path,
        bank_statement_path,
        applicant["Applicant_ID"],
        applicant["Age"],
        applicant["Employment_Type"],
    )
    model_input = build_model_input(record)

    assert list(model_input.columns) == FEATURE_COLUMNS
    assert len(model_input) == 1
    # Employment_Type is passed through as a raw string now - the
    # ColumnTransformer/OneHotEncoder in risk_modeling.train handles the
    # encoding, not build_features().
    assert model_input["Employment_Type"].iloc[0] == "Salaried"


def test_parse_bank_statement_counts_bounced_transactions(applicant, tmp_path):
    """document_simulation now renders Number_of_Bounced_Transactions_Last_6M
    as "CHEQUE BOUNCE CHARGE" debit rows scattered across the 6 months -
    bank_statement_parser.py should count them back out exactly."""
    bounced_applicant = applicant.copy()
    bounced_applicant["Number_of_Bounced_Transactions_Last_6M"] = 3

    path = tmp_path / "bounced_bank.xlsx"
    path.write_bytes(render_bank_statement(bounced_applicant, seed=7))

    parsed = parse_bank_statement(path)

    assert parsed["Number_of_Bounced_Transactions_Last_6M"] == 3


def test_parse_bank_statement_zero_bounces_counts_as_zero_not_missing(bank_statement_path):
    """The applicant fixture has 0 bounces - the recovered count should be a
    real 0, not None/NaN, same convention gemini_extractor.py uses when a
    document simply has no bounces to report."""
    parsed = parse_bank_statement(bank_statement_path)

    assert parsed["Number_of_Bounced_Transactions_Last_6M"] == 0


@pytest.fixture
def six_month_no_header_bank_statement_path(tmp_path) -> object:
    """A real-world-style (narrow, pipe-delimited) statement spanning 6 full
    calendar months with a distinct, constant balance per month and no
    "Month N Avg Balance:" header - exercises the Cash_Flow_Trend fallback
    (_day_weighted_monthly_averages), which must compute the same figures
    directly from the transaction table instead of returning None.
    """
    lines = [
        "BANK NAME: State Bank of India",
        "ACCOUNT NAME: Rahul Verma",
        "ACCOUNT NUMBER: 30998273645",
        "STATEMENT START: 2026-01-01",
        "STATEMENT END: 2026-06-30",
        "OPENING BALANCE: 100000.00",
        None,
        "DATE       | REF_NO     | DESCRIPTION                  | WITHDRAWAL | DEPOSIT   | BALANCE",
        "-" * 91,
        "2026-01-01 | TXN-001    | B/F OPENING BALANCE          | 0.00       | 0.00      | 100000.00",
        "2026-02-01 | TXN-002    | NEFT CREDIT                  | 0.00       | 10000.00  | 110000.00",
        "2026-03-01 | TXN-003    | NEFT CREDIT                  | 0.00       | 10000.00  | 120000.00",
        "2026-04-01 | TXN-004    | NEFT CREDIT                  | 0.00       | 10000.00  | 130000.00",
        "2026-05-01 | TXN-005    | NEFT CREDIT                  | 0.00       | 10000.00  | 140000.00",
        "2026-06-01 | TXN-006    | NEFT CREDIT                  | 0.00       | 10000.00  | 150000.00",
        "2026-06-30 | TXN-007    | CLOSING CARRY                | 0.00       | 0.00      | 150000.00",
    ]
    path = tmp_path / "six_month_bank.xlsx"
    pd.DataFrame({"col": lines}).to_excel(path, sheet_name="Sheet1", index=False, header=False)
    return path


def test_cash_flow_trend_falls_back_to_computed_monthly_averages(six_month_no_header_bank_statement_path):
    parsed = parse_bank_statement(six_month_no_header_bank_statement_path)

    assert parsed["Cash_Flow_Trend"] is not None
    assert len(parsed["Cash_Flow_Trend"]) == 6
    expected = [100000.0, 110000.0, 120000.0, 130000.0, 140000.0, 150000.0]
    for actual, target in zip(parsed["Cash_Flow_Trend"], expected):
        assert actual == pytest.approx(target, rel=0.01)
