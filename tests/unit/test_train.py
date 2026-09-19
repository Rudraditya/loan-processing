"""Direct unit tests for build_features()'s engineered ratio/slope formulas.

There was previously no dedicated test file for train.py at all - FEATURE_COLUMNS/
build_features() were only indirectly exercised via test_extraction_pipeline.py's
schema-match assertion. These tests check the formulas themselves against a
small hand-built fixture, independent of the full synthetic dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_processing.risk_modeling.emi import estimated_monthly_installment
from loan_processing.risk_modeling.train import FEATURE_COLUMNS, _build_preprocessor, build_features


def _base_row(**overrides) -> dict:
    row = {
        "Age": 30,
        "Employment_Type": "Salaried",
        "Monthly_Net_Income": 75000.0,
        "Total_Existing_EMIs": 12000.0,
        "Is_First_Loan": 0,
        "Requested_Loan_Amount": 500000.0,
        "Requested_Tenure_Months": 36,
        "Average_Monthly_Bank_Balance": 150000.0,
        "Number_of_Bounced_Transactions_Last_6M": 0,
        "Gross_Income": 96153.85,
        "Total_Deductions": 21153.85,
        "Cash_Flow_Month_1": 145000.0,
        "Cash_Flow_Month_2": 147000.0,
        "Cash_Flow_Month_3": 149000.0,
        "Cash_Flow_Month_4": 151000.0,
        "Cash_Flow_Month_5": 153000.0,
        "Cash_Flow_Month_6": 155000.0,
    }
    row.update(overrides)
    return row


def test_feature_columns_are_produced_in_order():
    df = pd.DataFrame([_base_row()])
    features = build_features(df)

    assert list(features.columns) == FEATURE_COLUMNS


def test_deductions_to_gross_ratio_formula():
    df = pd.DataFrame([_base_row(Gross_Income=100000.0, Total_Deductions=25000.0)])
    features = build_features(df)

    assert features["Deductions_to_Gross_Ratio"].iloc[0] == 0.25


def test_cash_flow_trend_slope_is_positive_for_rising_trend():
    df = pd.DataFrame([_base_row()])  # Cash_Flow_Month_1..6 rise by 2000 each month
    features = build_features(df)

    assert features["Cash_Flow_Trend_Slope"].iloc[0] > 0


def test_cash_flow_trend_slope_is_negative_for_declining_trend():
    row = _base_row(
        Cash_Flow_Month_1=160000.0,
        Cash_Flow_Month_2=157000.0,
        Cash_Flow_Month_3=154000.0,
        Cash_Flow_Month_4=151000.0,
        Cash_Flow_Month_5=148000.0,
        Cash_Flow_Month_6=145000.0,
    )
    df = pd.DataFrame([row])
    features = build_features(df)

    assert features["Cash_Flow_Trend_Slope"].iloc[0] < 0


def test_cash_flow_trend_slope_is_zero_for_flat_trend():
    row = _base_row(**{f"Cash_Flow_Month_{i}": 150000.0 for i in range(1, 7)})
    df = pd.DataFrame([row])
    features = build_features(df)

    assert features["Cash_Flow_Trend_Slope"].iloc[0] == 0


def test_cash_flow_trend_slope_is_nan_when_any_month_missing():
    row = _base_row(Cash_Flow_Month_3=np.nan)
    df = pd.DataFrame([row])
    features = build_features(df)

    assert np.isnan(features["Cash_Flow_Trend_Slope"].iloc[0])


def test_deductions_to_gross_ratio_is_nan_when_gross_income_missing():
    df = pd.DataFrame([_base_row(Gross_Income=np.nan)])
    features = build_features(df)

    assert np.isnan(features["Deductions_to_Gross_Ratio"].iloc[0])


def test_requested_emi_to_income_ratio_matches_amortization_formula():
    df = pd.DataFrame([_base_row(Requested_Loan_Amount=900000.0, Requested_Tenure_Months=36, Monthly_Net_Income=50000.0)])
    features = build_features(df)

    expected_installment = estimated_monthly_installment(900000.0, 36)
    assert features["Requested_EMI_to_Income_Ratio"].iloc[0] == pytest.approx(expected_installment / 50000.0)


def test_requested_emi_to_income_ratio_is_higher_for_shorter_tenure():
    """Same loan amount and income, only tenure differs - a short tenure's
    own installment should burden income far more than a long tenure's, which
    Loan_to_Annual_Income_Ratio alone can't distinguish (it ignores tenure).
    """
    short_tenure = build_features(pd.DataFrame([_base_row(Requested_Tenure_Months=12)]))
    long_tenure = build_features(pd.DataFrame([_base_row(Requested_Tenure_Months=120)]))

    assert (
        short_tenure["Requested_EMI_to_Income_Ratio"].iloc[0] > long_tenure["Requested_EMI_to_Income_Ratio"].iloc[0]
    )


def test_requested_emi_to_income_ratio_is_nan_when_tenure_missing():
    df = pd.DataFrame([_base_row(Requested_Tenure_Months=np.nan)])
    features = build_features(df)

    assert np.isnan(features["Requested_EMI_to_Income_Ratio"].iloc[0])


def _training_frame(*employment_types: str) -> pd.DataFrame:
    return build_features(pd.DataFrame([_base_row(Employment_Type=t) for t in employment_types]))


def test_preprocessor_one_hot_encodes_employment_type_into_binary_columns():
    X = _training_frame("Salaried", "Self-Employed", "Business Owner")
    preprocessor = _build_preprocessor(FEATURE_COLUMNS, impute_numeric=False)

    transformed = preprocessor.fit_transform(X)
    names = list(preprocessor.get_feature_names_out())

    cat_columns = [n for n in names if n.startswith("cat__")]
    # 3 categories seen at fit time -> 3 one-hot columns, each row a single 1.
    assert len(cat_columns) == 3
    cat_indices = [names.index(c) for c in cat_columns]
    for row in transformed[:, cat_indices]:
        assert row.sum() == 1
        assert set(row) <= {0.0, 1.0}


def test_preprocessor_handles_unseen_category_gracefully():
    """Defense-in-depth: even though Business Owner is now a real training
    category, handle_unknown='ignore' must still degrade gracefully (rather
    than raising) if some future category the encoder never saw at fit time
    shows up at scoring time.
    """
    X_train = _training_frame("Salaried", "Self-Employed")
    preprocessor = _build_preprocessor(FEATURE_COLUMNS, impute_numeric=False)
    preprocessor.fit(X_train)

    X_unseen = _training_frame("Business Owner")
    transformed = preprocessor.transform(X_unseen)
    names = list(preprocessor.get_feature_names_out())
    cat_indices = [i for i, n in enumerate(names) if n.startswith("cat__")]

    # Never-seen category -> all one-hot columns zero-filled, not an exception.
    assert transformed[:, cat_indices].sum() == 0


def test_preprocessor_scales_numeric_and_imputes_only_when_requested():
    X = _training_frame("Salaried", "Salaried")
    X.loc[0, "Average_Monthly_Bank_Balance"] = np.nan

    xgb_preprocessor = _build_preprocessor(FEATURE_COLUMNS, impute_numeric=False)
    transformed_xgb = xgb_preprocessor.fit_transform(X)
    names = list(xgb_preprocessor.get_feature_names_out())
    balance_idx = names.index("num__Average_Monthly_Bank_Balance")
    assert np.isnan(transformed_xgb[0, balance_idx])  # XGBoost path preserves NaN

    lr_preprocessor = _build_preprocessor(FEATURE_COLUMNS, impute_numeric=True)
    transformed_lr = lr_preprocessor.fit_transform(X)
    assert not np.isnan(transformed_lr[0, balance_idx])  # LR path median-imputes


def test_full_pipeline_transforms_raw_row_and_predicts():
    """End-to-end: a fitted Pipeline (preprocessor + classifier) takes a raw
    DataFrame - Employment_Type as a string - and returns a valid probability
    in one call, matching how agentic_api/main.py and app.py score live
    requests.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X = _training_frame("Salaried", "Self-Employed", "Business Owner", "Salaried", "Business Owner")
    y = pd.Series([0, 1, 0, 1, 0])

    preprocessor = _build_preprocessor(FEATURE_COLUMNS, impute_numeric=True)
    X_transformed = preprocessor.fit_transform(X)
    classifier = LogisticRegression(max_iter=1000).fit(X_transformed, y)
    pipeline = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])

    probability = pipeline.predict_proba(_training_frame("Business Owner"))[0, 1]
    assert 0.0 <= probability <= 1.0
