"""Sandbox-only: trains a Logistic Regression credit-risk classifier on the
Kaggle "Credit Risk Dataset" using ITS OWN native columns (person_age,
loan_grade, loan_intent, ...) as features, rather than mapping the dataset
onto this project's synthetic FEATURE_COLUMNS schema the way
prepare_secondary_data.py does. That existing mapping only carries 5 of
Kaggle's 12 raw columns forward and leaves 6 of the synthetic schema's 13
features NaN for every row; this script instead asks "how well does a model
do with the real signal Kaggle actually has" (loan_grade, loan_intent,
home_ownership, interest rate, etc.) as a separate benchmark point.

Not wired into app.py/agentic_api - this never becomes the production model.
Defaults write to models/experiments/kaggle_native/ and
reports/figures/experiments/kaggle_native/ so a bare run can never collide
with production artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from loan_processing.risk_modeling.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_KAGGLE_PATH = REPO_ROOT / "data" / "raw_kaggle_credit_data.csv"
MODELS_DIR = REPO_ROOT / "models" / "experiments" / "kaggle_native"
FIGURES_DIR = REPO_ROOT / "reports" / "figures" / "experiments" / "kaggle_native"

TARGET_COLUMN = "loan_status"

# person_age up to 144 and person_emp_length up to 123 appear in the raw
# file - both physically impossible (max realistic career length, max
# realistic applicant age). Filtered out in clean_kaggle_data() rather than
# left to distort StandardScaler's mean/std and SMOTE's neighbor distances.
MAX_PLAUSIBLE_AGE = 90
MAX_PLAUSIBLE_EMP_LENGTH_YEARS = 65

ONEHOT_FEATURES = ["person_home_ownership", "loan_intent"]
# loan_grade (A-G) and cb_person_default_on_file (N/Y) both carry a natural
# order - ordinal-encoded to a single numeric column each rather than
# one-hot, preserving that ordering instead of discarding it.
ORDINAL_FEATURES = ["loan_grade", "cb_person_default_on_file"]
ORDINAL_CATEGORIES = [["A", "B", "C", "D", "E", "F", "G"], ["N", "Y"]]
NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
FEATURE_COLUMNS = ONEHOT_FEATURES + ORDINAL_FEATURES + NUMERIC_FEATURES


def clean_kaggle_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drops exact-duplicate rows and physically-impossible age/employment-
    length outliers before any split, so they can't leak into scaling
    statistics or SMOTE's neighbor computation. Removes ~0.5% of rows on the
    full dataset - see CLAUDE.md's Kaggle-native benchmarking entry."""
    cleaned = df.drop_duplicates()
    cleaned = cleaned[cleaned["person_age"] <= MAX_PLAUSIBLE_AGE]
    cleaned = cleaned[~(cleaned["person_emp_length"] > MAX_PLAUSIBLE_EMP_LENGTH_YEARS)]
    return cleaned.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()


def _build_preprocessor() -> ColumnTransformer:
    """Logistic Regression can't accept NaN, so - same convention as
    train.py's LR path - numeric columns are median-imputed before scaling.
    handle_unknown='ignore'/'use_encoded_value' on the categorical/ordinal
    encoders means an unseen category at scoring time degrades gracefully
    instead of raising, matching train.py's ColumnTransformer convention."""
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT_FEATURES),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES, handle_unknown="use_encoded_value", unknown_value=-1
                ),
                ORDINAL_FEATURES,
            ),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )


def main(
    data_path: Path | None = None,
    models_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> None:
    data_path = data_path or RAW_KAGGLE_PATH
    models_dir = models_dir or MODELS_DIR
    figures_dir = figures_dir or FIGURES_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path)
    cleaned = clean_kaggle_data(raw)
    print(f"Rows before cleaning: {len(raw)}, after dedup/outlier filter: {len(cleaned)}")
    print(f"Observed default rate: {cleaned[TARGET_COLUMN].mean():.2%}")

    X = build_features(cleaned)
    y = cleaned[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = _build_preprocessor()
    feature_names = list(preprocessor.fit(X_train).get_feature_names_out())
    X_train_scaled = pd.DataFrame(preprocessor.transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(preprocessor.transform(X_test), columns=feature_names, index=X_test.index)

    print(f"\nTraining class balance before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"Training class balance after SMOTE:  {y_train_res.value_counts().to_dict()}")

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print("\n===== Logistic Regression (Kaggle-native features) =====")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
    print(pd.Series(metrics).round(4).to_string())

    plot_confusion_matrix(y_test, y_pred, "Logistic Regression (Kaggle-native)", figures_dir / "confusion_matrix_logistic_regression.png")
    plot_roc_curves({"Logistic Regression (Kaggle-native)": (y_test, y_proba)}, figures_dir / "roc_curve.png")

    pipeline = Pipeline([("preprocess", preprocessor), ("classifier", model)])
    joblib.dump(pipeline, models_dir / "logistic_regression_pipeline.pkl")
    joblib.dump(FEATURE_COLUMNS, models_dir / "feature_columns.pkl")

    print(f"\nSaved model artifacts to {models_dir}")
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sandbox: train Logistic Regression on Kaggle's own native feature columns"
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Defaults to data/raw_kaggle_credit_data.csv")
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Defaults to models/experiments/kaggle_native (never production)"
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Defaults to reports/figures/experiments/kaggle_native (never production)",
    )
    args = parser.parse_args()
    main(data_path=args.data_path, models_dir=args.models_dir, figures_dir=args.figures_dir)
