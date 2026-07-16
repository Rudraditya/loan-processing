"""Trains and evaluates baseline (Logistic Regression) and XGBoost credit-risk
classifiers on the synthetic applicant dataset, balances the training set with
SMOTE, and serializes the XGBoost model + preprocessing artifacts for the
document-extraction integration phase.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from loan_processing.risk_modeling.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from loan_processing.risk_modeling.labeling import add_default_label

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "applicants.csv"
LABELED_DATA_PATH = REPO_ROOT / "data" / "processed" / "applicants_labeled.csv"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

FEATURE_COLUMNS = [
    "Age",
    "Is_Self_Employed",
    "Monthly_Net_Income",
    "Total_Existing_EMIs",
    "CIBIL_Score",
    "Requested_Loan_Amount",
    "Requested_Tenure_Months",
    "Average_Monthly_Bank_Balance",
    "Number_of_Bounced_Transactions_Last_6M",
    "EMI_to_Income_Ratio",
    "Loan_to_Annual_Income_Ratio",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["Age"] = df["Age"]
    features["Is_Self_Employed"] = (df["Employment_Type"] == "Self-Employed").astype(int)
    features["Monthly_Net_Income"] = df["Monthly_Net_Income"]
    features["Total_Existing_EMIs"] = df["Total_Existing_EMIs"]
    features["CIBIL_Score"] = df["CIBIL_Score"]
    features["Requested_Loan_Amount"] = df["Requested_Loan_Amount"]
    features["Requested_Tenure_Months"] = df["Requested_Tenure_Months"]
    features["Average_Monthly_Bank_Balance"] = df["Average_Monthly_Bank_Balance"]
    features["Number_of_Bounced_Transactions_Last_6M"] = df["Number_of_Bounced_Transactions_Last_6M"]
    features["EMI_to_Income_Ratio"] = df["Total_Existing_EMIs"] / df["Monthly_Net_Income"]
    features["Loan_to_Annual_Income_Ratio"] = df["Requested_Loan_Amount"] / (df["Monthly_Net_Income"] * 12)
    return features[FEATURE_COLUMNS]


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LABELED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_DATA_PATH)
    labeled = add_default_label(raw)
    labeled.to_csv(LABELED_DATA_PATH, index=False)
    print(f"Labeled dataset saved to {LABELED_DATA_PATH}")
    print(f"Observed default rate: {labeled['Default'].mean():.2%}")

    X = build_features(labeled)
    y = labeled["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining class balance before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"Training class balance after SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    roc_inputs: dict[str, tuple[pd.Series, "pd.Series[float]"]] = {}
    all_metrics: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        all_metrics[name] = compute_metrics(y_test, y_pred, y_proba)
        roc_inputs[name] = (y_test, y_proba)

        print(f"\n===== {name} =====")
        print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
        print(f"AUC: {all_metrics[name]['auc']:.4f}")

        slug = name.lower().replace(" ", "_")
        plot_confusion_matrix(y_test, y_pred, name, FIGURES_DIR / f"confusion_matrix_{slug}.png")

    plot_roc_curves(roc_inputs, FIGURES_DIR / "roc_curve_comparison.png")
    plot_feature_importance(
        models["XGBoost"], FEATURE_COLUMNS, FIGURES_DIR / "xgboost_feature_importance.png"
    )

    print("\n===== Model Comparison =====")
    comparison_df = pd.DataFrame(all_metrics).T
    print(comparison_df.round(4).to_string())

    joblib.dump(models["XGBoost"], MODELS_DIR / "xgboost_model.pkl")
    joblib.dump(models["Logistic Regression"], MODELS_DIR / "logistic_regression_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_columns.pkl")

    print(f"\nSaved model artifacts to {MODELS_DIR}")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
