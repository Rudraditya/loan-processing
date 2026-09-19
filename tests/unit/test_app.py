import math

import pytest

from app import build_report, run_consistency_checks


@pytest.fixture
def complete_record() -> dict:
    return {
        "Applicant_ID": "APP00001",
        "Age": 34,
        "Employment_Type": "Salaried",
        "Monthly_Net_Income": 75000.0,
        "Total_Existing_EMIs": 12000.0,
        "Average_Monthly_Bank_Balance": 150000.0,
        "Is_First_Loan": 0,
        "Requested_Loan_Amount": 500000.0,
        "Requested_Tenure_Months": 36,
        "Number_of_Bounced_Transactions_Last_6M": 0,
    }


@pytest.fixture
def incomplete_record(complete_record) -> dict:
    record = dict(complete_record)
    for field in [
        "Is_First_Loan",
        "Requested_Loan_Amount",
        "Requested_Tenure_Months",
    ]:
        record[field] = math.nan
    return record


@pytest.fixture
def ntc_record(complete_record) -> dict:
    record = dict(complete_record)
    record["Is_First_Loan"] = 1
    return record


def test_consistency_checks_flag_missing_fields(incomplete_record):
    checks = run_consistency_checks(incomplete_record)

    missing_lines = [c for c in checks if c.startswith("[MISSING]")]
    assert len(missing_lines) == 3


def test_report_includes_ntc_note_for_first_time_borrowers(ntc_record):
    checks = run_consistency_checks(ntc_record)
    report = build_report("APP00001", ntc_record, checks, "No Default", 0.05, total_features=11)

    assert "NEW-TO-CREDIT (NTC) APPLICANT NOTE" in report
    assert "Monthly_Net_Income" in report
    assert "Average_Monthly_Bank_Balance" in report


def test_report_omits_ntc_note_for_returning_borrowers(complete_record):
    checks = run_consistency_checks(complete_record)
    report = build_report("APP00001", complete_record, checks, "No Default", 0.05, total_features=11)

    assert "NEW-TO-CREDIT (NTC) APPLICANT NOTE" not in report


def test_consistency_checks_flag_high_emi_ratio(complete_record):
    record = dict(complete_record)
    record["Total_Existing_EMIs"] = 50000.0  # 66% of income

    checks = run_consistency_checks(record)

    assert any(c.startswith("[FLAG]") and "EMI-to-income" in c for c in checks)


def test_report_recommends_manual_review_when_required_fields_missing(incomplete_record):
    checks = run_consistency_checks(incomplete_record)
    report = build_report("APP00001", incomplete_record, checks, "No Default", 0.05, total_features=11)

    assert ">>> FLAG FOR MANUAL REVIEW <<<" in report
    assert "required underwriting inputs are missing" in report


def test_report_recommends_proceeding_when_complete_and_low_risk(complete_record):
    checks = run_consistency_checks(complete_record)
    report = build_report("APP00001", complete_record, checks, "No Default", 0.05, total_features=11)

    assert ">>> PROCEED TO UNDERWRITING <<<" in report


def test_report_recommends_manual_review_when_model_predicts_default(complete_record):
    checks = run_consistency_checks(complete_record)
    report = build_report("APP00001", complete_record, checks, "Default", 0.85, total_features=11)

    assert ">>> FLAG FOR MANUAL REVIEW <<<" in report
    assert "elevated default risk" in report
