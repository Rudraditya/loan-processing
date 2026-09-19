"""Generates a grounded, correlated synthetic dataset of loan applicants.

Fields are not sampled independently: income drives EMI burden and bank
balance, EMI burden and missed payments drive CIBIL score, loan amount scales
with income, and requested tenure is derived from a target debt-service
ratio rather than drawn independently of loan size (see the tenure block
below) - so most applicants request a serviceable loan, with a right-skewed
minority requesting a genuinely aggressive one. This keeps the dataset from
looking like independent random noise, which would make it useless for
training a credit model later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from loan_processing.data_generation.schemas import ApplicantRecord
from loan_processing.risk_modeling.emi import ASSUMED_ANNUAL_INTEREST_RATE

DEFAULT_SEED = 42
_TENURE_OPTIONS = np.array([12, 24, 36, 48, 60, 84, 120, 180, 240])
_TENURE_WEIGHTS = np.array([0.08, 0.12, 0.18, 0.16, 0.14, 0.12, 0.10, 0.06, 0.04])
_OUTPUT_COLUMNS = [
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


def generate_applicants(n: int = 1000, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    age = np.clip(rng.normal(38, 10, n), 21, 65).round().astype(int)
    employment_type = rng.choice(
        ["Salaried", "Self-Employed", "Business Owner"], size=n, p=[0.60, 0.25, 0.15]
    )
    is_self_employed = employment_type == "Self-Employed"
    is_business_owner = employment_type == "Business Owner"

    # Self-employed income is lumpier (higher sigma) than salaried. Business
    # owners skew to the highest mean income of the three (more upside) but
    # are also the most volatile (less predictable cash flow than either a
    # salaried employee or a typical self-employed professional). Income also
    # grows with early-career experience and plateaus, via a capped age-based
    # multiplier.
    income_mu = np.select(
        [is_self_employed, is_business_owner],
        [np.log(50_000), np.log(65_000)],
        default=np.log(55_000),
    )
    income_sigma = np.select([is_self_employed, is_business_owner], [0.70, 0.85], default=0.45)
    experience_factor = 1 + 0.015 * np.clip(age - 21, 0, 25)
    monthly_net_income = rng.lognormal(income_mu, income_sigma) * experience_factor

    # EMI burden as a fraction of income: right-skewed, most applicants carry
    # a light-to-moderate load, a tail carries a heavy one.
    emi_burden_ratio = rng.beta(2, 5, n) * 0.65
    total_existing_emis = emi_burden_ratio * monthly_net_income

    # Bounced transactions rise with EMI burden and, for the self-employed
    # and business owners, with income volatility - business owners get the
    # larger bump, reflecting their higher cash-flow variability.
    bounce_lambda = (
        0.3
        + 4.0 * emi_burden_ratio
        + np.where(is_self_employed, 0.3, 0.0)
        + np.where(is_business_owner, 0.45, 0.0)
    )
    number_of_bounced_transactions_last_6m = rng.poisson(bounce_lambda)

    # CIBIL score: population-level baseline noise, pulled down by EMI burden
    # and by bounced transactions, clipped to the valid band.
    cibil_score = (
        rng.normal(750, 60, n)
        - emi_burden_ratio * 150
        - number_of_bounced_transactions_last_6m * 15
    )
    cibil_score = np.clip(cibil_score, 300, 900).round().astype(int)

    # Balance scales with income (a few months' worth), reduced when a large
    # share of income is already committed to EMIs.
    balance_multiplier = rng.uniform(0.5, 3.0, n) * (1 - 0.5 * emi_burden_ratio)
    average_monthly_bank_balance = np.clip(monthly_net_income * balance_multiplier, 0, None)

    # Requested_Tenure_Months is drawn first (a realistic real-world spread
    # of common loan terms, skewed toward mid-length ones), then
    # Requested_Loan_Amount is *derived* from a target debt-service ratio -
    # the loan's own estimated EMI (at emi.py's assumed rate) as a fraction
    # of income - inverted against that tenure via the standard amortization
    # formula. This is the same "how much can I borrow at this EMI, over
    # this term" calculation a real lender/applicant would do, so loan size
    # naturally scales with both income and tenure (a longer tenure
    # affords a bigger loan at the same monthly burden) - rather than
    # sampling loan amount as a flat multiple of annual income independently
    # of tenure, which previously left the majority of applicants requesting
    # a loan whose own EMI already exceeded their monthly income outright.
    # A right-skewed minority of target ratios exceeds 1.0, i.e. still
    # requests a genuinely unaffordable EMI - real, if reckless, applicant
    # behavior, and exactly the pattern a risk model should learn to flag.
    requested_tenure_months = rng.choice(_TENURE_OPTIONS, size=n, p=_TENURE_WEIGHTS)

    monthly_rate = ASSUMED_ANNUAL_INTEREST_RATE / 12
    target_emi_ratio = np.clip(rng.gamma(2.0, 0.2, n), 0.05, 3.0)
    target_installment = target_emi_ratio * monthly_net_income
    growth = (1 + monthly_rate) ** requested_tenure_months
    requested_loan_amount = target_installment * (growth - 1) / (monthly_rate * growth)

    # New-to-credit (NTC) applicants: ~15% are taking their first loan ever,
    # so they have no CIBIL_Score to report. Drawn last (after every other
    # column) so inserting this doesn't shift the RNG stream and change any
    # previously-generated column's values for a given seed.
    is_first_loan = (rng.random(n) < 0.15).astype(int)
    cibil_score_with_ntc = cibil_score.astype(float)
    cibil_score_with_ntc[is_first_loan == 1] = np.nan

    # Gross income / deductions: salaried applicants have formal payroll
    # withholding (higher, tighter range); self-employed and business-owner
    # applicants report more business-expense-style deductions (lower, ~60%
    # of the salaried rate). Net income was already drawn above and stays
    # the source of truth everywhere else - Gross_Income/Total_Deductions
    # are derived from it here. Drawn last (after is_first_loan) so no
    # existing seed=42 column's values shift.
    base_deduction_rate = rng.uniform(0.15, 0.28, n)
    deduction_rate = np.where(
        is_self_employed | is_business_owner, base_deduction_rate * 0.6, base_deduction_rate
    )
    gross_income = monthly_net_income / (1 - deduction_rate)
    total_deductions = gross_income - monthly_net_income

    # 6-month cash-flow trend: a per-applicant monthly drift around the
    # already-drawn Average_Monthly_Bank_Balance. Relative to the *population
    # average* EMI burden / bounced-transaction count - not absolute values -
    # so this is genuinely two-sided: below-average applicants trend upward
    # (improving cash flow), above-average applicants trend downward, rather
    # than every applicant being pulled toward decline by different amounts
    # (an earlier version subtracted only non-negative terms with no
    # centering, so ~81% of applicants ended up declining regardless of how
    # financially healthy they were - this centers the signal so roughly half
    # the population trends each way, matching the direction of an
    # applicant's actual risk profile). month_offsets sums to exactly zero,
    # so the per-row mean of the 6 columns equals Average_Monthly_Bank_Balance
    # by construction - no drift in that existing aggregate figure. Also
    # drawn last, for the same seed=42-safety reason as above.
    month_offsets = np.arange(1, 7) - 3.5
    emi_burden_centered = emi_burden_ratio - emi_burden_ratio.mean()
    bounced_centered = number_of_bounced_transactions_last_6m - number_of_bounced_transactions_last_6m.mean()
    trend_bias = -0.08 * emi_burden_centered - 0.015 * bounced_centered
    monthly_drift_pct = np.clip(trend_bias + rng.normal(0, 0.025, n), -0.12, 0.12)
    cash_flow_months = average_monthly_bank_balance[:, None] * (
        1 + monthly_drift_pct[:, None] * month_offsets[None, :]
    )

    df = pd.DataFrame(
        {
            "Applicant_ID": [f"APP{i + 1:05d}" for i in range(n)],
            "Full_Name": [faker.name() for _ in range(n)],
            "Employer_Name": [faker.company() for _ in range(n)],
            "Age": age,
            "Employment_Type": employment_type,
            "Monthly_Net_Income": monthly_net_income.round(2),
            "Gross_Income": gross_income.round(2),
            "Total_Deductions": total_deductions.round(2),
            "Total_Existing_EMIs": total_existing_emis.round(2),
            "Is_First_Loan": is_first_loan,
            "CIBIL_Score": cibil_score_with_ntc,
            "Requested_Loan_Amount": requested_loan_amount.round(2),
            "Requested_Tenure_Months": requested_tenure_months,
            "Average_Monthly_Bank_Balance": average_monthly_bank_balance.round(2),
            "Number_of_Bounced_Transactions_Last_6M": number_of_bounced_transactions_last_6m,
            "Cash_Flow_Month_1": cash_flow_months[:, 0].round(2),
            "Cash_Flow_Month_2": cash_flow_months[:, 1].round(2),
            "Cash_Flow_Month_3": cash_flow_months[:, 2].round(2),
            "Cash_Flow_Month_4": cash_flow_months[:, 3].round(2),
            "Cash_Flow_Month_5": cash_flow_months[:, 4].round(2),
            "Cash_Flow_Month_6": cash_flow_months[:, 5].round(2),
        }
    )
    return df[_OUTPUT_COLUMNS]


def validate_applicants(df: pd.DataFrame) -> None:
    """Raises if any row fails the ApplicantRecord schema."""
    for row in df.itertuples(index=False):
        cibil_score = None if pd.isna(row.CIBIL_Score) else int(row.CIBIL_Score)
        ApplicantRecord(
            applicant_id=row.Applicant_ID,
            full_name=row.Full_Name,
            employer_name=row.Employer_Name,
            age=row.Age,
            employment_type=row.Employment_Type,
            monthly_net_income=row.Monthly_Net_Income,
            gross_income=row.Gross_Income,
            total_deductions=row.Total_Deductions,
            total_existing_emis=row.Total_Existing_EMIs,
            is_first_loan=row.Is_First_Loan,
            cibil_score=cibil_score,
            requested_loan_amount=row.Requested_Loan_Amount,
            requested_tenure_months=row.Requested_Tenure_Months,
            average_monthly_bank_balance=row.Average_Monthly_Bank_Balance,
            number_of_bounced_transactions_last_6m=row.Number_of_Bounced_Transactions_Last_6M,
        )


def save_applicants(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    applicants = generate_applicants(n=10_000, seed=DEFAULT_SEED)
    validate_applicants(applicants)
    output_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "applicants.csv"
    save_applicants(applicants, output_path)
    print(f"Wrote {len(applicants)} records to {output_path}")
