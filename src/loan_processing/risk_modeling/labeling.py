"""Derives a synthetic `Default` target from the generated applicant features.

data_generation produces realistic, correlated *features* but no ground-truth
outcome. This combines the same risk factors (CIBIL score, EMI burden,
bounced transactions, loan-to-income, income level) into a z-scored logistic
risk signal, calibrates its intercept to hit a target base default rate, and
adds Gaussian noise before sampling a Bernoulli outcome — so the label is
grounded in the risk factors but not perfectly separable from them, which is
what makes the downstream classification metrics meaningful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TARGET_RATE = 0.17
DEFAULT_NOISE_STD = 0.75


def _zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _calibrate_intercept(raw_logit: np.ndarray, target_rate: float) -> float:
    """Binary search the intercept b such that mean(sigmoid(raw_logit + b)) ~= target_rate."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(raw_logit + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def add_default_label(
    df: pd.DataFrame,
    seed: int = 42,
    target_default_rate: float = DEFAULT_TARGET_RATE,
    noise_std: float = DEFAULT_NOISE_STD,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    emi_to_income = df["Total_Existing_EMIs"] / df["Monthly_Net_Income"]
    loan_to_annual_income = df["Requested_Loan_Amount"] / (df["Monthly_Net_Income"] * 12)
    cibil_risk = 900 - df["CIBIL_Score"]
    log_income = np.log(df["Monthly_Net_Income"])

    # New-to-credit (NTC) applicants have no CIBIL_Score (NaN), so they get
    # zero contribution from the credit-history term instead of propagating
    # NaN through the rest of the logit — their synthetic risk is driven by
    # the other (alternative-data) factors below, same as a real NTC
    # underwriting decision would be.
    cibil_zscore = _zscore(cibil_risk).fillna(0.0)

    raw_logit = (
        1.4 * cibil_zscore
        + 1.2 * _zscore(emi_to_income)
        + 0.8 * _zscore(df["Number_of_Bounced_Transactions_Last_6M"])
        + 0.6 * _zscore(loan_to_annual_income)
        - 0.5 * _zscore(log_income)
    ).to_numpy()

    intercept = _calibrate_intercept(raw_logit, target_default_rate)
    noisy_logit = raw_logit + intercept + rng.normal(0, noise_std, len(df))
    default_probability = _sigmoid(noisy_logit)

    labeled = df.copy()
    labeled["Default_Probability"] = default_probability.round(4)
    labeled["Default"] = rng.binomial(1, default_probability)
    return labeled
