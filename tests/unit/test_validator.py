from loan_processing.extraction.validator import compute_discrepancy_flags, is_missing


def test_is_missing_detects_none_and_nan():
    assert is_missing(None) is True
    assert is_missing(float("nan")) is True
    assert is_missing(0.0) is False
    assert is_missing("") is False


def test_no_flags_when_figures_reconcile():
    record = {
        "Gross_Income": 100000.0,
        "Total_Deductions": 25000.0,
        "Monthly_Net_Income": 75000.0,
        "Cash_Flow_Trend": [1, 2, 3, 4, 5, 6],
    }

    assert compute_discrepancy_flags(record) == []


def test_flags_mismatched_gross_minus_deductions():
    record = {
        "Gross_Income": 100000.0,
        "Total_Deductions": 25000.0,
        "Monthly_Net_Income": 60000.0,  # should be 75000
    }

    flags = compute_discrepancy_flags(record)

    assert len(flags) == 1
    assert "doesn't reconcile" in flags[0]


def test_flags_wrong_length_trend():
    record = {"Cash_Flow_Trend": [1, 2, 3]}

    flags = compute_discrepancy_flags(record)

    assert len(flags) == 1
    assert "3 month(s)" in flags[0]


def test_no_false_positive_when_fields_missing():
    record = {
        "Gross_Income": None,
        "Total_Deductions": None,
        "Monthly_Net_Income": 75000.0,
        "Cash_Flow_Trend": None,
    }

    assert compute_discrepancy_flags(record) == []
