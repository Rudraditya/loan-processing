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
        "Total_Existing_EMIs",
        "Is_First_Loan",
        "CIBIL_Score",
        "Requested_Loan_Amount",
        "Requested_Tenure_Months",
        "Average_Monthly_Bank_Balance",
        "Number_of_Bounced_Transactions_Last_6M",
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
    assert set(df["Employment_Type"].unique()) <= {"Salaried", "Self-Employed"}
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
