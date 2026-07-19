"""Verifies prepare_secondary_data.py's mapping/NTC/cleaning logic against a
small, hand-built fixture matching the *assumed* Kaggle schema - not a real
downloaded dataset. This only proves the script's own logic is correct; it
is not a benchmark and produces no numbers meaningful outside this test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prepare_secondary_data import (
    FINAL_COLUMNS,
    clean_and_cap,
    engineer_is_first_loan,
    inject_synthetic_cibil_score,
    load_and_map,
)


@pytest.fixture
def raw_kaggle_csv(tmp_path):
    df = pd.DataFrame(
        {
            "person_age": [25, 40, 35, 999, 30],  # 999 is a deliberately corrupted row
            "person_income": [600_000, 1_200_000, 900_000, 500_000, np.nan],  # annual
            "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT", "RENT"],
            "person_emp_length": [2.0, 10.0, 5.0, 1.0, 0.0],
            "loan_amnt": [100_000, 200_000, 150_000, 50_000, 80_000],
            "loan_percent_income": [0.2, 0.15, 0.18, 0.3, 0.25],
            "loan_status": [0, 0, 1, 0, 0],
            "cb_person_default_on_file": ["N", "N", "Y", "N", "N"],
            "cb_person_cred_hist_length": [3, 12, 0, 0, 5],  # two NTC rows (hist length 0)
        }
    )
    path = tmp_path / "raw_kaggle_credit_data.csv"
    df.to_csv(path, index=False)
    return path


def test_load_and_map_raises_clearly_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_and_map(tmp_path / "nope.csv")


def test_load_and_map_raises_clearly_on_schema_mismatch(tmp_path):
    path = tmp_path / "wrong_schema.csv"
    pd.DataFrame({"totally_different_column": [1, 2, 3]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing expected column"):
        load_and_map(path)


def test_full_pipeline_produces_expected_schema_and_ntc_flagging(raw_kaggle_csv):
    mapped = load_and_map(raw_kaggle_csv)
    engineered = engineer_is_first_loan(mapped)
    cleaned = clean_and_cap(engineered)

    assert list(cleaned.columns[: len(FINAL_COLUMNS)]) != []  # sanity: didn't come back empty
    for column in FINAL_COLUMNS:
        assert column in cleaned.columns

    # The corrupted-age row (999) and the missing-income row should be dropped.
    assert len(cleaned) == 3
    assert cleaned["Age"].between(18, 100).all()

    # CIBIL_Score has no real source in this schema - NaN for every row.
    assert cleaned["CIBIL_Score"].isna().all()

    # Is_First_Loan == 1 exactly where credit-history length was 0, and those
    # rows carry no fabricated CIBIL_Score.
    ntc_rows = cleaned[cleaned["Is_First_Loan"] == 1]
    assert len(ntc_rows) >= 1
    assert ntc_rows["CIBIL_Score"].isna().all()

    # Fields this dataset structurally can't supply stay explicitly NaN,
    # not silently defaulted to 0 or fabricated.
    assert cleaned["Average_Monthly_Bank_Balance"].isna().all()
    assert cleaned["Number_of_Bounced_Transactions_Last_6M"].isna().all()

    # The real loan_status outcome is carried through as Default, not
    # replaced by a synthetic label.
    assert set(cleaned["Default"].unique()) <= {0, 1}

    # build_features() needs these even though the task description didn't
    # list them; loan_amnt supplies the former for real, the latter has no
    # source in this dataset and is NaN.
    assert (cleaned["Requested_Loan_Amount"] > 0).all()
    assert cleaned["Requested_Tenure_Months"].isna().all()


def test_inject_synthetic_cibil_score_fabricates_plausible_values_and_matching_ntc():
    df = pd.DataFrame(
        {
            "Applicant_ID": [f"ID{i}" for i in range(2000)],
            "CIBIL_Score": [np.nan] * 2000,
            "Is_First_Loan": [0] * 2000,
        }
    )

    result = inject_synthetic_cibil_score(df, ntc_rate=0.15, seed=1)

    # Every value is fabricated by design, but must still land in the valid range.
    populated = result.loc[result["Is_First_Loan"] == 0, "CIBIL_Score"]
    assert populated.between(300, 900).all()
    assert populated.notna().all()

    # NTC rate should land close to the requested 15% on a large enough sample.
    assert abs(result["Is_First_Loan"].mean() - 0.15) < 0.03

    # The invariant enforced elsewhere in this codebase must hold: CIBIL_Score
    # is NaN exactly where Is_First_Loan == 1, and nowhere else.
    ntc_rows = result[result["Is_First_Loan"] == 1]
    non_ntc_rows = result[result["Is_First_Loan"] == 0]
    assert ntc_rows["CIBIL_Score"].isna().all()
    assert non_ntc_rows["CIBIL_Score"].notna().all()


def test_inject_synthetic_cibil_score_is_reproducible_with_same_seed():
    df = pd.DataFrame({"CIBIL_Score": [np.nan] * 500, "Is_First_Loan": [0] * 500})

    first = inject_synthetic_cibil_score(df, seed=7)
    second = inject_synthetic_cibil_score(df, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_clean_and_cap_winsorizes_extreme_income_outlier():
    df = pd.DataFrame(
        {
            "Applicant_ID": [f"ID{i}" for i in range(100)],
            "Age": [30] * 100,
            "Employment_Type": ["Salaried"] * 100,
            "Monthly_Net_Income": [50_000.0] * 99 + [50_000_000.0],  # one wild outlier
            "Total_Existing_EMIs": [10_000.0] * 100,
            "CIBIL_Score": [np.nan] * 100,
            "Is_First_Loan": [0] * 100,
            "Average_Monthly_Bank_Balance": [np.nan] * 100,
            "Number_of_Bounced_Transactions_Last_6M": [np.nan] * 100,
        }
    )

    capped = clean_and_cap(df)

    assert capped["Monthly_Net_Income"].max() < 50_000_000.0
