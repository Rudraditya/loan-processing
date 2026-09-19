import numpy as np

from loan_processing.data_generation.applicant_generator import generate_applicants, validate_applicants


def test_generates_requested_row_count_and_columns():
    df = generate_applicants(n=50, seed=1)

    assert len(df) == 50
    assert list(df.columns) == [
        "Applicant_ID",
        "Full_Name",
        "Employer_Name",
        "Age",
        "Employment_Type",
        "Monthly_Net_Income",
        "Gross_Income",
        "Total_Deductions",
        "Total_Existing_EMIs",
        "Is_First_Loan",
        "CIBIL_Score",
        "Requested_Loan_Amount",
        "Requested_Tenure_Months",
        "Average_Monthly_Bank_Balance",
        "Number_of_Bounced_Transactions_Last_6M",
        "Cash_Flow_Month_1",
        "Cash_Flow_Month_2",
        "Cash_Flow_Month_3",
        "Cash_Flow_Month_4",
        "Cash_Flow_Month_5",
        "Cash_Flow_Month_6",
    ]


def test_generation_is_reproducible_with_same_seed():
    first = generate_applicants(n=50, seed=7)
    second = generate_applicants(n=50, seed=7)

    assert first.equals(second)


def test_values_stay_within_valid_ranges():
    df = generate_applicants(n=200, seed=2)

    assert df["Age"].between(21, 65).all()
    assert df["CIBIL_Score"].dropna().between(300, 900).all()
    assert (df["Monthly_Net_Income"] > 0).all()
    assert (df["Total_Existing_EMIs"] >= 0).all()
    assert (df["Requested_Tenure_Months"] > 0).all()
    assert set(df["Employment_Type"].unique()) <= {"Salaried", "Self-Employed", "Business Owner"}
    assert not df["Applicant_ID"].duplicated().any()


def test_first_loan_applicants_have_no_cibil_score():
    df = generate_applicants(n=1000, seed=6)

    assert set(df["Is_First_Loan"].unique()) <= {0, 1}
    ntc_share = df["Is_First_Loan"].mean()
    assert 0.10 < ntc_share < 0.20  # target ~15%

    assert df.loc[df["Is_First_Loan"] == 1, "CIBIL_Score"].isna().all()
    assert df.loc[df["Is_First_Loan"] == 0, "CIBIL_Score"].notna().all()


def test_all_rows_pass_schema_validation():
    df = generate_applicants(n=100, seed=3)

    validate_applicants(df)  # raises on any invalid row


def test_higher_emi_burden_correlates_with_lower_cibil_score():
    df = generate_applicants(n=1000, seed=4)
    emi_ratio = df["Total_Existing_EMIs"] / df["Monthly_Net_Income"]

    assert emi_ratio.corr(df["CIBIL_Score"]) < -0.2


def test_gross_income_correlates_with_net_income_and_reconciles_with_deductions():
    df = generate_applicants(n=1000, seed=5)

    assert df["Gross_Income"].corr(df["Monthly_Net_Income"]) > 0.9
    assert (df["Gross_Income"] > df["Monthly_Net_Income"]).all()
    assert ((df["Gross_Income"] - df["Total_Deductions"] - df["Monthly_Net_Income"]).abs() < 0.02).all()


def test_cash_flow_months_average_to_the_stored_bank_balance():
    df = generate_applicants(n=1000, seed=8)
    month_cols = [f"Cash_Flow_Month_{i}" for i in range(1, 7)]

    mean_of_months = df[month_cols].mean(axis=1)
    assert ((mean_of_months - df["Average_Monthly_Bank_Balance"]).abs() < 0.01).all()


def test_cash_flow_trend_direction_is_roughly_balanced_across_the_population():
    """Regression guard: an earlier version of the trend-bias formula only
    ever subtracted non-negative terms (EMI burden, bounced transactions)
    with no centering, so ~81% of applicants ended up with a declining trend
    regardless of how financially healthy they were - every applicant's
    cash-flow chart looked like a variation on the same downward slope. The
    bias must be centered on the population average so roughly half trend
    each way, with direction actually reflecting relative risk.
    """
    df = generate_applicants(n=5000, seed=10)
    month_cols = [f"Cash_Flow_Month_{i}" for i in range(1, 7)]

    x = np.arange(1, 7)
    x_centered = x - x.mean()
    y = df[month_cols].to_numpy(dtype=float)
    slope = ((y - y.mean(axis=1, keepdims=True)) * x_centered).sum(axis=1) / (x_centered**2).sum()

    share_declining = (slope < 0).mean()
    assert 0.35 < share_declining < 0.65


def test_business_owner_income_is_highest_mean_and_most_volatile():
    df = generate_applicants(n=5000, seed=9)
    by_type = df.groupby("Employment_Type")["Monthly_Net_Income"]

    means = by_type.mean()
    stds = by_type.std()

    assert means["Business Owner"] > means["Salaried"]
    assert means["Business Owner"] > means["Self-Employed"]
    assert stds["Business Owner"] > stds["Salaried"]
    assert stds["Business Owner"] > stds["Self-Employed"]
