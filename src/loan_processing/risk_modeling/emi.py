"""Shared estimated-EMI calculation, used by both `labeling.py` (the
synthetic ground-truth risk signal) and `train.py` (a model feature) so the
two formulas/rate assumptions can't drift apart - `labeling.py` cannot
import from `train.py` (train.py already imports labeling.py, and a reverse
import would be circular), so this lives in its own leaf module instead of
being duplicated inline the way `emi_to_income`/`loan_to_annual_income` are
elsewhere in these two files.

Neither the document-extraction pipeline nor the loan-application form
captures a real, product-specific interest rate - this project has no
rate-quoting logic at all - so ASSUMED_ANNUAL_INTEREST_RATE is a deliberate
simplifying assumption (a representative Indian retail/personal-loan rate),
not a value read from any real quote or applicant input.

This exists to close a real gap: `Loan_to_Annual_Income_Ratio` (loan amount
divided by annual income) ignores tenure entirely, and `Requested_Tenure_Months`
on its own carries no income-affordability information - so a large loan
compressed into a short tenure (whose own monthly installment would badly
exceed the applicant's income) looked identical to the same loan spread
over a long, affordable tenure. `estimated_monthly_installment()` computes
the standard reducing-balance EMI for the requested loan/tenure at the
assumed rate, so `Requested_EMI_to_Income_Ratio` can capture that
interaction directly.
"""
from __future__ import annotations

import numpy as np

ASSUMED_ANNUAL_INTEREST_RATE = 0.11  # representative Indian retail-loan rate; not applicant/product-specific


def estimated_monthly_installment(principal, tenure_months, annual_rate: float = ASSUMED_ANNUAL_INTEREST_RATE):
    """Standard reducing-balance EMI formula: P * r * (1+r)^n / ((1+r)^n - 1),
    where r is the monthly rate and n is the tenure in months. Vectorized -
    accepts scalars, numpy arrays, or pandas Series for `principal`/
    `tenure_months`. Returns NaN (rather than raising or dividing by zero)
    wherever `tenure_months` is missing or non-positive.
    """
    principal = np.asarray(principal, dtype=float)
    tenure_months = np.asarray(tenure_months, dtype=float)
    monthly_rate = annual_rate / 12

    valid = tenure_months > 0
    safe_tenure = np.where(valid, tenure_months, 1.0)
    growth = np.power(1 + monthly_rate, safe_tenure)
    installment = principal * monthly_rate * growth / (growth - 1)
    return np.where(valid, installment, np.nan)
