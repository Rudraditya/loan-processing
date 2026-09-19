"""Regression guard for recall under the one remaining realistic
feature-availability gap (Cash_Flow_Trend_Slope NaN for a bank statement
under 6 months of history) - see risk_modeling/evaluate_live_scenario.py.

Unlike most tests in this suite, this one deliberately depends on the real
committed production artifacts (data/processed/applicants_labeled.csv,
models/logistic_regression_pipeline.pkl) rather than a tmp_path fixture -
its entire purpose is catching a regression in the actual deployed model,
which a synthetic fixture can't stand in for.
"""
from __future__ import annotations

import pytest

from loan_processing.risk_modeling.evaluate_live_scenario import (
    LABELED_DATA_PATH,
    MODELS_DIR,
    evaluate_scenarios,
)

pytestmark = pytest.mark.skipif(
    not LABELED_DATA_PATH.exists() or not (MODELS_DIR / "logistic_regression_pipeline.pkl").exists(),
    reason="Requires the committed production labeled dataset and model artifacts.",
)

# Floors, not exact numbers - loosely bounded so a full model retrain
# (different but still-healthy numbers) doesn't spuriously fail this, while
# still catching a real regression, e.g. a future change that silently
# reopens a permanent feature-availability gap like the pre-Tier-1
# bounced-transactions one (recall crashed 0.91 -> 0.73 that time).
_MIN_LIVE_RECALL = 0.80
_MAX_CLEAN_TO_LIVE_RECALL_DROP = 0.15


def test_live_scenario_recall_does_not_regress():
    results = evaluate_scenarios()

    clean_recall = results["clean"]["recall"]
    live_recall = results["live_short_history"]["recall"]

    assert live_recall >= _MIN_LIVE_RECALL, (
        f"Live-scenario recall dropped to {live_recall:.3f}, below the {_MIN_LIVE_RECALL} floor - "
        "a real feature-availability gap may have reopened. See CLAUDE.md's benchmarking "
        "table ('Live-scenario recall benchmark' section) for the incident this guards against."
    )
    assert clean_recall - live_recall <= _MAX_CLEAN_TO_LIVE_RECALL_DROP, (
        f"Clean-vs-live recall gap widened to {clean_recall - live_recall:.3f}, above the "
        f"{_MAX_CLEAN_TO_LIVE_RECALL_DROP} tolerance - the model may be relying on a feature "
        "that isn't reliably available at live scoring time."
    )
