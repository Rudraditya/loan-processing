"""Standing benchmark: evaluates the production Logistic Regression pipeline
under a simulated real-world feature-availability scenario, not just the
clean synthetic test set train.py's own printed report reflects. Exists
because that clean-data number can't reveal a live-deployment-only recall
regression - which is exactly what happened before the Tier 1 extraction
fixes: Number_of_Bounced_Transactions_Last_6M was silently NaN for every
single live-scored applicant (the mock bank statement never rendered
bounce line items), and recall on the real held-out test set crashed from
0.91 to 0.73 despite the clean number staying a reassuring 0.9490 AUC the
whole time - see CLAUDE.md's benchmarking table for the full story.

Cash_Flow_Trend_Slope is the one remaining realistic gap after those fixes
(see record_builder.py/bank_statement_parser.py) - it's genuinely NaN
whenever an uploaded bank statement has under 6 full calendar months of
transaction history, which no amount of extraction-logic work can
manufacture data for. Every other FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS
field is now either a required loan-application-form field or a real
document-derived value, so this is the only scenario worth tracking here
today - add another DEGRADED_FEATURES entry if a future change reopens a
different one.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from loan_processing.risk_modeling.evaluation import compute_metrics
from loan_processing.risk_modeling.train import build_features

REPO_ROOT = Path(__file__).resolve().parents[3]
LABELED_DATA_PATH = REPO_ROOT / "data" / "processed" / "applicants_labeled.csv"
MODELS_DIR = REPO_ROOT / "models"

DEGRADED_FEATURES = ["Cash_Flow_Trend_Slope"]


def _held_out_test_split(labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduces train.py's own train_test_split (test_size=0.2, seed=42)
    exactly, so this scores against the identical held-out rows the
    production model's own reported metrics come from - not a different,
    incomparable split."""
    X = build_features(labeled)
    y = labeled["Default"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_test, y_test


def evaluate_scenarios(
    labeled_data_path: Path | None = None, models_dir: Path | None = None
) -> dict[str, dict[str, float]]:
    """Returns {"clean": {...}, "live_short_history": {...}}, each a
    compute_metrics() dict (accuracy/precision/recall/f1_score/auc)."""
    labeled_data_path = labeled_data_path or LABELED_DATA_PATH
    models_dir = models_dir or MODELS_DIR

    labeled = pd.read_csv(labeled_data_path)
    X_test, y_test = _held_out_test_split(labeled)
    pipeline = joblib.load(models_dir / "logistic_regression_pipeline.pkl")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    clean_metrics = compute_metrics(y_test, y_pred, y_proba)

    X_degraded = X_test.copy()
    X_degraded[DEGRADED_FEATURES] = np.nan
    y_pred_degraded = pipeline.predict(X_degraded)
    y_proba_degraded = pipeline.predict_proba(X_degraded)[:, 1]
    live_metrics = compute_metrics(y_test, y_pred_degraded, y_proba_degraded)

    return {"clean": clean_metrics, "live_short_history": live_metrics}


def main() -> None:
    results = evaluate_scenarios()
    comparison = pd.DataFrame(results).T
    print("===== Production model: clean test set vs. live (short bank-statement-history) scenario =====")
    print(comparison.round(4).to_string())
    recall_drop = results["clean"]["recall"] - results["live_short_history"]["recall"]
    print(f"\nRecall drop (clean -> live): {recall_drop:.4f}")


if __name__ == "__main__":
    main()
