"""Unit tests for train_kaggle_native.py's cleaning/feature-building/
preprocessing logic and an end-to-end smoke test of main(), against a small
hand-built fixture - independent of the real (large, uncommitted-in-spirit)
data/raw_kaggle_credit_data.csv, same convention as test_prepare_secondary_data.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_processing.risk_modeling.train_kaggle_native import (
    FEATURE_COLUMNS,
    MAX_PLAUSIBLE_AGE,
    MAX_PLAUSIBLE_EMP_LENGTH_YEARS,
    _build_preprocessor,
    build_features,
    clean_kaggle_data,
    main,
)


def _base_row(**overrides) -> dict:
    row = {
        "person_age": 30,
        "person_income": 60000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 11.5,
        "loan_status": 0,
        "loan_percent_income": 0.17,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4,
    }
    row.update(overrides)
    return row


def test_clean_kaggle_data_drops_exact_duplicates():
    df = pd.DataFrame([_base_row(), _base_row()])
    cleaned = clean_kaggle_data(df)

    assert len(cleaned) == 1


def test_clean_kaggle_data_filters_implausible_age():
    df = pd.DataFrame([_base_row(), _base_row(person_age=MAX_PLAUSIBLE_AGE + 1)])
    cleaned = clean_kaggle_data(df)

    assert len(cleaned) == 1
    assert cleaned["person_age"].iloc[0] <= MAX_PLAUSIBLE_AGE


def test_clean_kaggle_data_filters_implausible_emp_length():
    df = pd.DataFrame([_base_row(), _base_row(person_emp_length=MAX_PLAUSIBLE_EMP_LENGTH_YEARS + 1)])
    cleaned = clean_kaggle_data(df)

    assert len(cleaned) == 1


def test_clean_kaggle_data_keeps_missing_emp_length_rows():
    """A NaN emp_length must not be treated as an outlier by the > threshold
    check - it's a missing value, handled later by imputation, not a bad
    value to be filtered out."""
    df = pd.DataFrame([_base_row(person_emp_length=np.nan)])
    cleaned = clean_kaggle_data(df)

    assert len(cleaned) == 1


def test_build_features_selects_expected_columns_in_order():
    df = pd.DataFrame([_base_row()])
    features = build_features(df)

    assert list(features.columns) == FEATURE_COLUMNS


def test_preprocessor_ordinal_encodes_loan_grade_preserving_rank():
    df = build_features(pd.DataFrame([_base_row(loan_grade=g) for g in ["A", "D", "G"]]))
    preprocessor = _build_preprocessor()

    transformed = preprocessor.fit_transform(df)
    names = list(preprocessor.get_feature_names_out())
    grade_idx = names.index("ordinal__loan_grade")

    assert list(transformed[:, grade_idx]) == [0.0, 3.0, 6.0]


def test_preprocessor_one_hot_encodes_home_ownership():
    df = build_features(pd.DataFrame([_base_row(person_home_ownership=v) for v in ["RENT", "OWN", "MORTGAGE"]]))
    preprocessor = _build_preprocessor()

    transformed = preprocessor.fit_transform(df)
    names = list(preprocessor.get_feature_names_out())
    ownership_indices = [i for i, n in enumerate(names) if n.startswith("onehot__person_home_ownership")]

    assert len(ownership_indices) == 3
    for row in transformed[:, ownership_indices]:
        assert row.sum() == 1


def test_preprocessor_imputes_missing_numeric_values():
    df = build_features(pd.DataFrame([_base_row(), _base_row(person_emp_length=np.nan)]))
    preprocessor = _build_preprocessor()

    transformed = preprocessor.fit_transform(df)
    names = list(preprocessor.get_feature_names_out())
    emp_idx = names.index("numeric__person_emp_length")

    assert not np.isnan(transformed[:, emp_idx]).any()


def _synthetic_kaggle_frame(n_per_class: int = 30, seed: int = 42) -> pd.DataFrame:
    """Small but SMOTE-sized (needs >= 6 minority-class training rows)
    synthetic fixture mimicking the real column distributions - not the real
    Kaggle file. Class separation is loose so classification_report can be
    computed without conditioning on a specific accuracy value.
    """
    rng = np.random.default_rng(seed)
    grades = ["A", "B", "C", "D", "E", "F", "G"]
    ownerships = ["RENT", "OWN", "MORTGAGE", "OTHER"]
    intents = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"]

    rows = []
    for _ in range(n_per_class):
        rows.append(
            _base_row(
                person_age=int(rng.integers(21, 60)),
                person_income=float(rng.integers(20000, 80000)),
                person_home_ownership=rng.choice(ownerships),
                person_emp_length=float(rng.integers(0, 20)),
                loan_intent=rng.choice(intents),
                loan_grade=rng.choice(grades[:3]),
                loan_amnt=float(rng.integers(1000, 10000)),
                loan_int_rate=float(rng.uniform(6, 12)),
                loan_status=0,
                loan_percent_income=float(rng.uniform(0.05, 0.2)),
                cb_person_default_on_file="N",
                cb_person_cred_hist_length=int(rng.integers(1, 10)),
            )
        )
    for _ in range(n_per_class):
        rows.append(
            _base_row(
                person_age=int(rng.integers(21, 60)),
                person_income=float(rng.integers(15000, 40000)),
                person_home_ownership=rng.choice(ownerships),
                person_emp_length=float(rng.integers(0, 5)),
                loan_intent=rng.choice(intents),
                loan_grade=rng.choice(grades[4:]),
                loan_amnt=float(rng.integers(8000, 35000)),
                loan_int_rate=float(rng.uniform(14, 22)),
                loan_status=1,
                loan_percent_income=float(rng.uniform(0.4, 0.7)),
                cb_person_default_on_file="Y",
                cb_person_cred_hist_length=int(rng.integers(0, 3)),
            )
        )
    return pd.DataFrame(rows)


def test_main_end_to_end_saves_pipeline_and_runs_without_error(tmp_path):
    df = _synthetic_kaggle_frame()
    data_path = tmp_path / "fixture_kaggle.csv"
    df.to_csv(data_path, index=False)
    models_dir = tmp_path / "models"
    figures_dir = tmp_path / "figures"

    main(data_path=data_path, models_dir=models_dir, figures_dir=figures_dir)

    assert (models_dir / "logistic_regression_pipeline.pkl").exists()
    assert (models_dir / "feature_columns.pkl").exists()
    assert (figures_dir / "confusion_matrix_logistic_regression.png").exists()
    assert (figures_dir / "roc_curve.png").exists()


def test_main_saved_pipeline_predicts_valid_probabilities(tmp_path):
    import joblib

    df = _synthetic_kaggle_frame()
    data_path = tmp_path / "fixture_kaggle.csv"
    df.to_csv(data_path, index=False)
    models_dir = tmp_path / "models"
    figures_dir = tmp_path / "figures"

    main(data_path=data_path, models_dir=models_dir, figures_dir=figures_dir)

    pipeline = joblib.load(models_dir / "logistic_regression_pipeline.pkl")
    sample = build_features(pd.DataFrame([_base_row()]))
    proba = pipeline.predict_proba(sample)[0, 1]

    assert 0.0 <= proba <= 1.0
