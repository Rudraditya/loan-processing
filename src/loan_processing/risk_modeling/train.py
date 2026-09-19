"""Trains and evaluates baseline (Logistic Regression) and XGBoost credit-risk
classifiers on the synthetic applicant dataset, balances the training set with
SMOTE, and serializes each model as one bundled sklearn Pipeline (preprocessing
+ classifier) for the document-extraction integration phase.

CIBIL_Score (a credit-bureau score) is deliberately excluded from
FEATURE_COLUMNS - the model relies entirely on income/EMI/balance/repayment
behavior instead of a bureau pull. XGBoost is still configured with
missing=np.nan and never imputes, since extraction-derived records (and
some secondary datasets) still leave several *other* fields (e.g.
Requested_Loan_Amount, Number_of_Bounced_Transactions_Last_6M) genuinely
missing. SMOTE cannot interpolate a feature that doesn't exist for a row, so
it's applied only to the complete-case rows (see `_smote_resample_allow_nan`);
incomplete rows are kept in the training set as-is, unresampled. Logistic
Regression, unlike XGBoost, cannot accept NaN at all (neither to fit nor to
predict), so - for that baseline comparison model only - missing values are
median-imputed; the production model (XGBoost) never sees an imputed value.

Employment_Type is the one raw-categorical feature (`Salaried`/`Self-Employed`/
`Business Owner`) - `_build_preprocessor()` one-hot encodes it via
`OneHotEncoder(handle_unknown="ignore")` inside a `ColumnTransformer`, so an
unseen category at scoring time degrades gracefully (zero-filled dummies)
rather than raising. Each model's preprocessor and classifier are fit
separately (SMOTE needs to run on the already-encoded/scaled array, between
the two - imblearn's own auto-SMOTE Pipeline step doesn't have a clean hook
for `_smote_resample_allow_nan`'s "skip incomplete rows" logic), then
assembled into a `sklearn.pipeline.Pipeline` from the already-fitted pieces
purely for serialization/scoring convenience - `pipeline.predict_proba(raw_df)`
runs the whole preprocess-then-predict sequence in one call.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from loan_processing.risk_modeling.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from loan_processing.risk_modeling.emi import estimated_monthly_installment
from loan_processing.risk_modeling.labeling import add_default_label

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "applicants.csv"
LABELED_DATA_PATH = REPO_ROOT / "data" / "processed" / "applicants_labeled.csv"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

FEATURE_COLUMNS = [
    "Age",
    "Employment_Type",
    "Monthly_Net_Income",
    "Total_Existing_EMIs",
    "Is_First_Loan",
    "Requested_Loan_Amount",
    "Requested_Tenure_Months",
    "Average_Monthly_Bank_Balance",
    "Number_of_Bounced_Transactions_Last_6M",
    "EMI_to_Income_Ratio",
    "Loan_to_Annual_Income_Ratio",
    "Deductions_to_Gross_Ratio",
    "Cash_Flow_Trend_Slope",
    "Requested_EMI_to_Income_Ratio",
]

# Employment_Type is the only raw-categorical feature - build_features()
# passes it through as a string; the ColumnTransformer preprocessor built by
# _build_preprocessor() below one-hot encodes it. Every other FEATURE_COLUMNS
# entry is numeric and gets scaled (optionally median-imputed first).
CATEGORICAL_FEATURES = ["Employment_Type"]

_CASH_FLOW_MONTH_COLUMNS = [f"Cash_Flow_Month_{i}" for i in range(1, 7)]


def _cash_flow_trend_slope(df: pd.DataFrame) -> pd.Series:
    """Centered OLS slope of the 6 monthly cash-flow figures (x = 1..6).
    Explicitly NaN if any of the 6 months is missing for a row, rather than
    silently averaging over fewer points - same "explicit NaN over a
    fabricated partial value" convention as FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS.
    """
    x = np.arange(1, 7)
    x_centered = x - x.mean()
    denom = (x_centered**2).sum()
    y = df[_CASH_FLOW_MONTH_COLUMNS].to_numpy(dtype=float)
    any_missing = np.isnan(y).any(axis=1)
    y_mean = np.where(any_missing, np.nan, np.nanmean(y, axis=1))[:, None]
    slope = ((y - y_mean) * x_centered).sum(axis=1) / denom
    return pd.Series(np.where(any_missing, np.nan, slope), index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["Age"] = df["Age"]
    features["Employment_Type"] = df["Employment_Type"]
    features["Monthly_Net_Income"] = df["Monthly_Net_Income"]
    features["Total_Existing_EMIs"] = df["Total_Existing_EMIs"]
    features["Is_First_Loan"] = df["Is_First_Loan"]
    features["Requested_Loan_Amount"] = df["Requested_Loan_Amount"]
    features["Requested_Tenure_Months"] = df["Requested_Tenure_Months"]
    features["Average_Monthly_Bank_Balance"] = df["Average_Monthly_Bank_Balance"]
    features["Number_of_Bounced_Transactions_Last_6M"] = df["Number_of_Bounced_Transactions_Last_6M"]
    features["EMI_to_Income_Ratio"] = df["Total_Existing_EMIs"] / df["Monthly_Net_Income"]
    features["Loan_to_Annual_Income_Ratio"] = df["Requested_Loan_Amount"] / (df["Monthly_Net_Income"] * 12)
    features["Deductions_to_Gross_Ratio"] = df["Total_Deductions"] / df["Gross_Income"]
    features["Cash_Flow_Trend_Slope"] = _cash_flow_trend_slope(df)
    # The *requested* loan's own estimated installment (at emi.py's assumed
    # rate) against income - unlike Loan_to_Annual_Income_Ratio, this is
    # tenure-sensitive: the same loan amount compressed into a short tenure
    # produces a much higher ratio here than spread over a long one, which a
    # coarse loan/annual-income ratio can't distinguish. NaN wherever
    # Requested_Tenure_Months is itself NaN (see emi.py).
    installment = estimated_monthly_installment(df["Requested_Loan_Amount"], df["Requested_Tenure_Months"])
    features["Requested_EMI_to_Income_Ratio"] = installment / df["Monthly_Net_Income"]
    return features[FEATURE_COLUMNS]


def _build_preprocessor(feature_columns: list[str], impute_numeric: bool) -> ColumnTransformer:
    """Builds the ColumnTransformer that turns build_features()'s raw output
    (Employment_Type as a string, everything else numeric) into a fully
    numeric array: one-hot encodes the categorical column(s) - gracefully
    zero-filling any category never seen at fit time, via handle_unknown -
    and scales the numeric columns (optionally median-imputing them first,
    for the Logistic Regression path only - XGBoost keeps true NaN via its
    own missing=np.nan handling, so its preprocessor must not impute).

    Computed from `feature_columns` (this run's actual post-exclusion column
    list, e.g. after --exclude-features), not the module-level FEATURE_COLUMNS
    constant directly - so excluding Employment_Type (or, in principle, every
    numeric column) from a given run doesn't leave a stale, mismatched
    transformer.
    """
    categorical = [c for c in feature_columns if c in CATEGORICAL_FEATURES]
    numeric = [c for c in feature_columns if c not in CATEGORICAL_FEATURES]

    numeric_steps: list[tuple] = [("scale", StandardScaler())]
    if impute_numeric:
        numeric_steps.insert(0, ("impute", SimpleImputer(strategy="median")))

    transformers = []
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    return ColumnTransformer(transformers)


def _smote_resample_allow_nan(
    X: pd.DataFrame, y: pd.Series, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTE can't synthesize a feature that's genuinely absent for a given
    row - it requires finite input to compute neighbor distances. Oversample
    only the complete-case rows; rows with a missing value are kept as-is,
    unresampled, alongside the oversampled result.

    Columns that are NaN for the *entire* training set (e.g.
    Average_Monthly_Bank_Balance / etc. when training on a secondary dataset
    that never supplies them at all) are excluded from the completeness
    check — every row is equally "incomplete" w.r.t. such a column, so
    treating that as disqualifying would leave zero complete-case rows for
    SMOTE to learn from. SMOTE runs on the remaining columns only; the
    always-missing ones are reattached as NaN on the synthesized rows too.
    """
    always_missing_cols = X.columns[X.isna().all()]
    checkable_cols = X.columns.difference(always_missing_cols)

    is_complete = ~X[checkable_cols].isna().any(axis=1) if len(checkable_cols) else pd.Series(False, index=X.index)
    X_complete, y_complete = X[is_complete], y[is_complete]
    X_missing, y_missing = X[~is_complete], y[~is_complete]

    if len(X_complete) == 0:
        # Nothing to oversample from - hand the input back unresampled
        # rather than crashing SMOTE on an empty array.
        return X.reset_index(drop=True), y.reset_index(drop=True)

    smote = SMOTE(random_state=seed)
    X_res_arr, y_res_arr = smote.fit_resample(X_complete[checkable_cols], y_complete)
    X_res = pd.DataFrame(np.asarray(X_res_arr), columns=checkable_cols)
    for col in always_missing_cols:
        X_res[col] = np.nan
    X_res = X_res[X.columns]
    y_res = pd.Series(np.asarray(y_res_arr), name=y.name)

    X_out = pd.concat([X_res, X_missing.reset_index(drop=True)], ignore_index=True)
    y_out = pd.concat([y_res, y_missing.reset_index(drop=True)], ignore_index=True)
    return X_out, y_out


def main(
    data_path: Path | None = None,
    exclude_features: list[str] | None = None,
    models_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> None:
    """`data_path` defaults to the original synthetic dataset. Pass
    `data/final_training_data.csv` (once it actually exists - see
    prepare_secondary_data.py) to train on the secondary dataset instead.
    If the input already has a `Default` column, it's used as the real
    label as-is; add_default_label() is only invoked as a fallback for
    data with no ground-truth outcome, like the synthetic applicants.csv.

    `exclude_features` drops columns from this run's own X only - it does
    NOT touch the module-level FEATURE_COLUMNS/build_features() contract
    that record_builder.py depends on, since a feature (e.g.
    Number_of_Bounced_Transactions_Last_6M) can be genuinely strong for the
    synthetic dataset and shouldn't be crippled globally just because a
    *different* dataset can't supply it.

    `models_dir`/`figures_dir` default to the standard MODELS_DIR/FIGURES_DIR
    (i.e. "production"). Pass a different directory for an experimental run
    so it doesn't silently overwrite the production model artifacts.
    """
    data_path = data_path or RAW_DATA_PATH
    models_dir = models_dir or MODELS_DIR
    figures_dir = figures_dir or FIGURES_DIR

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    LABELED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path)
    if "Default" in raw.columns:
        print(f"'{data_path.name}' already has a real Default column - using it as-is, "
              "not synthesizing a label over real feature data.")
        labeled = raw
    else:
        labeled = add_default_label(raw)
    labeled.to_csv(LABELED_DATA_PATH, index=False)
    print(f"Labeled dataset saved to {LABELED_DATA_PATH}")
    print(f"Observed default rate: {labeled['Default'].mean():.2%}")
    print(f"New-to-credit (NTC) share: {labeled['Is_First_Loan'].mean():.2%}")

    X = build_features(labeled)
    if exclude_features:
        unknown = [f for f in exclude_features if f not in X.columns]
        if unknown:
            raise ValueError(f"Cannot exclude unknown feature(s): {unknown}")
        print(f"Excluding feature(s) from this run only: {exclude_features}")
        X = X.drop(columns=exclude_features)
    feature_columns = list(X.columns)
    y = labeled["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- XGBoost path: true NaN preserved throughout, no imputation ---
    # The preprocessor one-hot encodes Employment_Type and scales the numeric
    # columns; StandardScaler passes NaN straight through untouched, same as
    # before - XGBoost's own missing=np.nan handles it downstream.
    xgb_preprocessor = _build_preprocessor(feature_columns, impute_numeric=False)
    xgb_feature_names = list(xgb_preprocessor.fit(X_train).get_feature_names_out())
    X_train_scaled = pd.DataFrame(xgb_preprocessor.transform(X_train), columns=xgb_feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(xgb_preprocessor.transform(X_test), columns=xgb_feature_names, index=X_test.index)

    print(f"\nTraining class balance before SMOTE: {y_train.value_counts().to_dict()}")
    X_train_res, y_train_res = _smote_resample_allow_nan(X_train_scaled, y_train, seed=42)
    print(f"Training class balance after SMOTE:  {y_train_res.value_counts().to_dict()}")

    # --- Logistic Regression path: median-imputed for this baseline only ---
    # Impute every numeric column with any missingness, not just CIBIL_Score -
    # the original synthetic dataset only ever left that one column NaN, but a
    # secondary dataset can leave several columns entirely missing. Logistic
    # Regression, unlike XGBoost, can't accept NaN under any circumstance.
    # SimpleImputer(strategy="median") is the sklearn-native equivalent of the
    # old manual `X_train.median().fillna(0.0)` step, now inside the pipeline.
    lr_preprocessor = _build_preprocessor(feature_columns, impute_numeric=True)
    lr_feature_names = list(lr_preprocessor.fit(X_train).get_feature_names_out())
    X_train_lr_scaled = pd.DataFrame(
        lr_preprocessor.transform(X_train), columns=lr_feature_names, index=X_train.index
    )
    X_test_lr_scaled = pd.DataFrame(lr_preprocessor.transform(X_test), columns=lr_feature_names, index=X_test.index)

    smote_lr = SMOTE(random_state=42)
    X_train_lr_res, y_train_lr_res = smote_lr.fit_resample(X_train_lr_scaled, y_train)

    model_runs = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "preprocessor": lr_preprocessor,
            "X_train": X_train_lr_res,
            "y_train": y_train_lr_res,
            "X_test": X_test_lr_scaled,
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                missing=np.nan,
                random_state=42,
            ),
            "preprocessor": xgb_preprocessor,
            "X_train": X_train_res,
            "y_train": y_train_res,
            "X_test": X_test_scaled,
        },
    }

    roc_inputs: dict[str, tuple[pd.Series, "pd.Series[float]"]] = {}
    all_metrics: dict[str, dict[str, float]] = {}

    for name, run in model_runs.items():
        model = run["model"]
        model.fit(run["X_train"], run["y_train"])
        y_pred = model.predict(run["X_test"])
        y_proba = model.predict_proba(run["X_test"])[:, 1]

        all_metrics[name] = compute_metrics(y_test, y_pred, y_proba)
        roc_inputs[name] = (y_test, y_proba)

        print(f"\n===== {name} =====")
        print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
        print(f"AUC: {all_metrics[name]['auc']:.4f}")

        slug = name.lower().replace(" ", "_")
        plot_confusion_matrix(y_test, y_pred, name, figures_dir / f"confusion_matrix_{slug}.png")

    plot_roc_curves(roc_inputs, figures_dir / "roc_curve_comparison.png")
    # Expanded (post-one-hot) names - XGBoost's feature_importances_ array
    # length matches the encoded column count, not the raw 13-feature list.
    plot_feature_importance(
        model_runs["XGBoost"]["model"], xgb_feature_names, figures_dir / "xgboost_feature_importance.png"
    )

    print("\n===== Model Comparison =====")
    comparison_df = pd.DataFrame(all_metrics).T
    print(comparison_df.round(4).to_string())

    # Each pipeline bundles its already-fitted preprocessor (one-hot +
    # scale/impute) with its already-fitted classifier into one object -
    # scoring becomes a single pipeline.predict_proba(raw_df) call, no
    # separate impute/scale steps for a caller to reproduce by hand.
    xgb_pipeline = Pipeline([("preprocess", xgb_preprocessor), ("classifier", model_runs["XGBoost"]["model"])])
    lr_pipeline = Pipeline(
        [("preprocess", lr_preprocessor), ("classifier", model_runs["Logistic Regression"]["model"])]
    )
    joblib.dump(xgb_pipeline, models_dir / "xgboost_pipeline.pkl")
    joblib.dump(lr_pipeline, models_dir / "logistic_regression_pipeline.pkl")
    # feature_columns.pkl keeps validating the raw, pre-transform input
    # contract (now including Employment_Type as a string instead of
    # Is_Self_Employed as an int) - agentic_api/main.py and app.py assert
    # extraction output matches this list before scoring.
    joblib.dump(feature_columns, models_dir / "feature_columns.pkl")

    print(f"\nSaved model artifacts to {models_dir}")
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the credit-risk models")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "CSV to train on. Defaults to the original synthetic data/raw/applicants.csv. "
            "Pass data/final_training_data.csv to train on the secondary dataset instead "
            "(run prepare_secondary_data.py first)."
        ),
    )
    parser.add_argument(
        "--exclude-features",
        type=lambda s: [f.strip() for f in s.split(",") if f.strip()],
        default=None,
        help="Comma-separated feature name(s) to drop from this run only, e.g. Number_of_Bounced_Transactions_Last_6M.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Where to save model artifacts. Defaults to models/ (production). Use a "
        "different directory for an experimental run so it doesn't overwrite production.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Where to save diagnostic plots. Defaults to reports/figures/ (production).",
    )
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        exclude_features=args.exclude_features,
        models_dir=args.models_dir,
        figures_dir=args.figures_dir,
    )
