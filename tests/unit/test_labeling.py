from loan_processing.data_generation.applicant_generator import generate_applicants
from loan_processing.risk_modeling.emi import estimated_monthly_installment
from loan_processing.risk_modeling.labeling import add_default_label


def test_default_rate_is_close_to_target():
    df = generate_applicants(n=2000, seed=1)
    labeled = add_default_label(df, seed=1, target_default_rate=0.17)

    assert set(labeled["Default"].unique()) <= {0, 1}
    assert abs(labeled["Default"].mean() - 0.17) < 0.03


def test_labeling_is_reproducible_with_same_seed():
    df = generate_applicants(n=200, seed=2)

    first = add_default_label(df, seed=5)
    second = add_default_label(df, seed=5)

    assert first["Default"].equals(second["Default"])


def test_higher_risk_factors_increase_default_probability():
    df = generate_applicants(n=1000, seed=3)
    labeled = add_default_label(df, seed=3)
    emi_ratio = labeled["Total_Existing_EMIs"] / labeled["Monthly_Net_Income"]

    assert emi_ratio.corr(labeled["Default_Probability"]) > 0.2
    assert labeled["CIBIL_Score"].corr(labeled["Default_Probability"]) < -0.2


def test_declining_cash_flow_trend_increases_default_probability():
    df = generate_applicants(n=1000, seed=9)
    labeled = add_default_label(df, seed=9)
    month_cols = [f"Cash_Flow_Month_{i}" for i in range(1, 7)]
    x = list(range(1, 7))
    x_mean = sum(x) / len(x)
    x_centered = [v - x_mean for v in x]
    denom = sum(v**2 for v in x_centered)
    slope = labeled[month_cols].apply(
        lambda row: sum((row[c] - row[month_cols].mean()) * xc for c, xc in zip(month_cols, x_centered)) / denom,
        axis=1,
    )

    # A more negative (declining) trend slope should correlate with higher default risk.
    assert slope.corr(labeled["Default_Probability"]) < -0.1


def test_deductions_to_gross_ratio_has_a_label_correlation():
    df = generate_applicants(n=1000, seed=10)
    labeled = add_default_label(df, seed=10)
    deductions_to_gross = labeled["Total_Deductions"] / labeled["Gross_Income"]

    assert abs(deductions_to_gross.corr(labeled["Default_Probability"])) > 0.05


def test_requested_loan_own_installment_burden_increases_default_probability():
    """Regression guard: the label used to only see Loan_to_Annual_Income_Ratio
    (tenure-agnostic) and Requested_Tenure_Months as independent signals, so a
    large loan compressed into a short tenure - whose own installment would
    badly exceed income - looked no riskier than the same amount spread over
    a long, affordable tenure. requested_emi_to_income closes that gap.
    """
    df = generate_applicants(n=1000, seed=11)
    labeled = add_default_label(df, seed=11)
    requested_emi_to_income = (
        estimated_monthly_installment(labeled["Requested_Loan_Amount"], labeled["Requested_Tenure_Months"])
        / labeled["Monthly_Net_Income"]
    )

    assert requested_emi_to_income.corr(labeled["Default_Probability"]) > 0.2
