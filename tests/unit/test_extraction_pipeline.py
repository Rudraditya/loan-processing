import math

import pandas as pd
import pytest

from loan_processing.document_simulation.bank_statement_xlsx import render_bank_statement
from loan_processing.document_simulation.salary_slip_pdf import render_salary_slip
from loan_processing.extraction.bank_statement_parser import parse_bank_statement
from loan_processing.extraction.record_builder import (
    FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS,
    build_applicant_record,
    build_model_input,
)
from loan_processing.extraction.salary_slip_parser import parse_salary_slip
from loan_processing.risk_modeling.train import FEATURE_COLUMNS


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

    assert parsed["Applicant_ID"] == applicant["Applicant_ID"]
    assert parsed["Age"] == applicant["Age"]
    assert parsed["Employment_Type"] == applicant["Employment_Type"]
    assert parsed["Monthly_Net_Income"] == pytest.approx(applicant["Monthly_Net_Income"])


def test_parse_bank_statement_recovers_emi_and_average_balance(bank_statement_path, applicant):
    parsed = parse_bank_statement(bank_statement_path)

    assert parsed["Total_Existing_EMIs"] == pytest.approx(applicant["Total_Existing_EMIs"], rel=0.01)
    assert parsed["Average_Monthly_Bank_Balance"] == pytest.approx(
        applicant["Average_Monthly_Bank_Balance"], rel=0.01
    )


def test_build_applicant_record_combines_both_documents(salary_slip_path, bank_statement_path, applicant):
    record = build_applicant_record(salary_slip_path, bank_statement_path)

    assert record["Applicant_ID"] == applicant["Applicant_ID"]
    assert record["Age"] == applicant["Age"]
    assert record["Employment_Type"] == applicant["Employment_Type"]
    for field in FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS:
        assert math.isnan(record[field])


def test_build_model_input_matches_saved_model_schema(salary_slip_path, bank_statement_path):
    record = build_applicant_record(salary_slip_path, bank_statement_path)
    model_input = build_model_input(record)

    assert list(model_input.columns) == FEATURE_COLUMNS
    assert len(model_input) == 1
    assert model_input["Is_Self_Employed"].iloc[0] == 0


def test_mismatched_applicant_ids_raise(salary_slip_path, bank_statement_path, applicant, tmp_path):
    other_applicant = applicant.copy()
    other_applicant["Applicant_ID"] = "APP99999"
    mismatched_bank_path = tmp_path / "mismatched_bank.xlsx"
    mismatched_bank_path.write_bytes(render_bank_statement(other_applicant, seed=1))

    with pytest.raises(ValueError, match="Applicant ID mismatch"):
        build_applicant_record(salary_slip_path, mismatched_bank_path)
