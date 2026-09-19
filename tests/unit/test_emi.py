import numpy as np

from loan_processing.risk_modeling.emi import estimated_monthly_installment


def test_estimated_monthly_installment_matches_hand_computed_amortization():
    # 900,000 over 36 months at the module's assumed 11% annual rate.
    monthly_rate = 0.11 / 12
    expected = 900_000 * monthly_rate * (1 + monthly_rate) ** 36 / ((1 + monthly_rate) ** 36 - 1)

    assert estimated_monthly_installment(900_000, 36) == expected


def test_estimated_monthly_installment_is_higher_for_shorter_tenure():
    short = estimated_monthly_installment(900_000, 12)
    long = estimated_monthly_installment(900_000, 120)

    assert short > long


def test_estimated_monthly_installment_is_nan_for_missing_or_nonpositive_tenure():
    assert np.isnan(estimated_monthly_installment(900_000, np.nan))
    assert np.isnan(estimated_monthly_installment(900_000, 0))
    assert np.isnan(estimated_monthly_installment(900_000, -5))


def test_estimated_monthly_installment_is_vectorized():
    principals = np.array([900_000.0, 500_000.0])
    tenures = np.array([36, np.nan])

    result = estimated_monthly_installment(principals, tenures)

    assert result[0] > 0
    assert np.isnan(result[1])
