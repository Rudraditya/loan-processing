"""Generates a grounded, correlated synthetic dataset of loan applicants.

Fields are not sampled independently: income drives EMI burden and bank
balance, EMI burden and missed payments drive CIBIL score, and loan amount
scales with income. This keeps the dataset from looking like independent
random noise, which would make it useless for training a credit model later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from loan_processing.data_generation.schemas import ApplicantRecord

DEFAULT_SEED = 42
_TENURE_OPTIONS = np.array([12, 24, 36, 48, 60, 84, 120])
_OUTPUT_COLUMNS = [
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


def generate_applicants(n: int = 1000, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    age = np.clip(rng.normal(38, 10, n), 21, 65).round().astype(int)
    employment_type = rng.choice(["Salaried", "Self-Employed"], size=n, p=[0.7, 0.3])
    is_self_employed = employment_type == "Self-Employed"

    # Self-employed income is lumpier (higher sigma). Income also grows with
    # early-career experience and plateaus, via a capped age-based multiplier.
    income_mu = np.where(is_self_employed, np.log(50_000), np.log(55_000))
    income_sigma = np.where(is_self_employed, 0.70, 0.45)
    experience_factor = 1 + 0.015 * np.clip(age - 21, 0, 25)
    monthly_net_income = rng.lognormal(income_mu, income_sigma) * experience_factor

    # EMI burden as a fraction of income: right-skewed, most applicants carry
    # a light-to-moderate load, a tail carries a heavy one.
    emi_burden_ratio = rng.beta(2, 5, n) * 0.65
    total_existing_emis = emi_burden_ratio * monthly_net_income

    # Bounced transactions rise with EMI burden and, for the self-employed,
    # with income volatility.
    bounce_lambda = 0.3 + 4.0 * emi_burden_ratio + np.where(is_self_employed, 0.3, 0.0)
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

    annual_income = monthly_net_income * 12
    requested_loan_amount = annual_income * rng.uniform(1.0, 6.0, n)

    # Larger loans skew toward longer tenures; bucket by loan-amount tercile
    # and weight the tenure draw accordingly rather than sampling uniformly.
    tenure_tercile = np.asarray(pd.qcut(requested_loan_amount, 3, labels=[0, 1, 2])).astype(int)
    tenure_weights_by_tercile = [
        np.array([0.30, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02]),
        np.array([0.10, 0.20, 0.25, 0.20, 0.15, 0.07, 0.03]),
        np.array([0.03, 0.07, 0.15, 0.20, 0.25, 0.20, 0.10]),
    ]
    requested_tenure_months = np.array(
        [
            rng.choice(_TENURE_OPTIONS, p=tenure_weights_by_tercile[tercile])
            for tercile in tenure_tercile
        ]
    )

    # New-to-credit (NTC) applicants: ~15% are taking their first loan ever,
    # so they have no CIBIL_Score to report. Drawn last (after every other
    # column) so inserting this doesn't shift the RNG stream and change any
    # previously-generated column's values for a given seed.
    is_first_loan = (rng.random(n) < 0.15).astype(int)
    cibil_score_with_ntc = cibil_score.astype(float)
    cibil_score_with_ntc[is_first_loan == 1] = np.nan

    df = pd.DataFrame(
        {
            "Applicant_ID": [f"APP{i + 1:05d}" for i in range(n)],
            "Full_Name": [faker.name() for _ in range(n)],
            "Employer_Name": [faker.company() for _ in range(n)],
            "Age": age,
            "Employment_Type": employment_type,
            "Monthly_Net_Income": monthly_net_income.round(2),
            "Total_Existing_EMIs": total_existing_emis.round(2),
            "Is_First_Loan": is_first_loan,
            "CIBIL_Score": cibil_score_with_ntc,
            "Requested_Loan_Amount": requested_loan_amount.round(2),
            "Requested_Tenure_Months": requested_tenure_months,
            "Average_Monthly_Bank_Balance": average_monthly_bank_balance.round(2),
            "Number_of_Bounced_Transactions_Last_6M": number_of_bounced_transactions_last_6m,
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
    applicants = generate_applicants(n=1000, seed=DEFAULT_SEED)
    validate_applicants(applicants)
    output_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "applicants.csv"
    save_applicants(applicants, output_path)
    print(f"Wrote {len(applicants)} records to {output_path}")
