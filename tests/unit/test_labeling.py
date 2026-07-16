from loan_processing.data_generation.applicant_generator import generate_applicants
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
