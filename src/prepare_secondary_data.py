"""Maps a secondary (Kaggle) credit-risk dataset onto this project's XGBoost
input schema, engineers the Is_First_Loan (NTC) flag, cleans the result, and
exports data/final_training_data.csv for risk_modeling/train.py to consume.

======================================================================
READ THIS BEFORE RUNNING - SCHEMA ASSUMPTIONS, NOT VERIFIED FACTS
======================================================================
As of writing, data/raw_kaggle_credit_data.csv does not exist in this repo.
This script is written against the column names of the commonly-referenced
Kaggle "Credit Risk Dataset" (person_age, person_income, person_emp_length,
person_home_ownership, loan_amnt, loan_percent_income, loan_status,
cb_person_default_on_file, cb_person_cred_hist_length) because the phrasing
of the request ("person_income", "credit history length") matches it - but
this was never confirmed against a real file. If your actual CSV has
different column names, `_REQUIRED_SOURCE_COLUMNS` below will fail loudly
rather than silently producing garbage - update the mapping in
`load_and_map()` to match your real columns before trusting the output.

This source schema cannot supply every target column, no matter how it's
mapped - these are real gaps in the dataset, not implementation bugs:

  - CIBIL_Score: no numeric bureau-score column exists in this dataset at
    all (only cb_person_cred_hist_length, a duration, and
    cb_person_default_on_file, a Y/N flag). CIBIL_Score is therefore NaN
    for every row, not just NTC ones. XGBoost handles this natively (see
    risk_modeling/train.py's missing=np.nan), but it means this secondary
    dataset can't validate the model's use of CIBIL_Score at all.
  - Average_Monthly_Bank_Balance / Number_of_Bounced_Transactions_Last_6M:
    no transaction- or balance-level data exists in this dataset (it's
    loan-application-level, not bank-statement-level). Both are NaN for
    every row.
  - Employment_Type: no Salaried/Self-Employed category exists (only
    person_emp_length, a duration in years). Every row is defaulted to
    "Salaried" - this is a placeholder, not a derived value, and should be
    called out explicitly in anything written up from this data.
  - Total_Existing_EMIs: no existing-debt figure exists either. Approximated
    as `loan_percent_income * Monthly_Net_Income`, which is actually the
    requested loan's share of income, not a pre-existing EMI burden - a
    different real-world quantity being used as a stand-in. Documented here
    so it isn't mistaken for real EMI data downstream.
  - Requested_Tenure_Months: no loan-term/duration field exists in this
    dataset either. NaN for every row.

The task description that prompted this script listed 7 target columns to
guarantee, but risk_modeling.train.build_features() actually requires two
more - Requested_Loan_Amount and Requested_Tenure_Months - to run at all
(it indexes df["Requested_Loan_Amount"] unconditionally). Both are included
below so train.py doesn't crash with a KeyError; loan_amnt maps directly to
the former, the latter has no source and is NaN (see above).

Net effect: only Applicant_ID, Age, Monthly_Net_Income, and Is_First_Loan
would carry genuine signal from this particular secondary dataset; the rest
are either constant defaults or NaN. Report this plainly rather than
implying a like-for-like replacement of the synthetic data.

One field this dataset DOES supply for real: loan_status, the actual
historical default outcome, carried through unchanged as `Default`. This
matters because the original pipeline's applicants.csv has no real-world
label at all - risk_modeling/labeling.py *synthesizes* one from the
features. Reusing that synthetic-labeling step on top of real feature data
here would fabricate a fake target where a real one already exists, which
would invalidate the whole point of benchmarking against real data.
train.py is updated to use this real `Default` column as-is whenever it's
present, and only fall back to synthetic labeling for the original
synthetic dataset, which has no such column.
======================================================================
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_KAGGLE_PATH = REPO_ROOT / "data" / "raw_kaggle_credit_data.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "final_training_data.csv"
# Deliberately a different filename from OUTPUT_PATH - this version's
# CIBIL_Score/Is_First_Loan are fabricated (see inject_synthetic_cibil_score),
# not real Kaggle data, and must never be confused with final_training_data.csv.
SYNTHETIC_CIBIL_OUTPUT_PATH = REPO_ROOT / "data" / "final_training_data_synthetic_cibil.csv"

# Fail loudly and immediately if the real file's columns don't match this
# assumption, rather than silently producing NaN-filled nonsense.
_REQUIRED_SOURCE_COLUMNS = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_percent_income",
    "loan_status",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

FINAL_COLUMNS = [
    "Applicant_ID",
    "Age",
    "Employment_Type",
    "Monthly_Net_Income",
    "Gross_Income",
    "Total_Deductions",
    "Total_Existing_EMIs",
    "CIBIL_Score",
    "Is_First_Loan",
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
    "Default",
]


def load_and_map(path: Path = RAW_KAGGLE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. This script does not fabricate a stand-in - "
            "place the real downloaded CSV there before running."
        )

    raw = pd.read_csv(path)

    missing_columns = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in raw.columns]
    if missing_columns:
        raise ValueError(
            f"raw_kaggle_credit_data.csv is missing expected column(s) {missing_columns}. "
            "The schema assumed by this script (see module docstring) doesn't match this "
            "file - update the mapping in load_and_map() to your real columns before "
            "re-running; don't silently proceed on a guessed schema."
        )

    mapped = pd.DataFrame(index=raw.index)
    mapped["Applicant_ID"] = [f"KAGGLE{i:06d}" for i in range(len(raw))]
    mapped["Age"] = raw["person_age"]

    # No Salaried/Self-Employed field exists in this dataset - documented
    # placeholder default, not a derived value. See module docstring.
    mapped["Employment_Type"] = "Salaried"

    # person_income in this dataset is annual, not monthly.
    mapped["Monthly_Net_Income"] = raw["person_income"] / 12

    # No existing-EMI figure exists; loan_percent_income is the requested
    # loan's share of income, used here as an approximate stand-in for an
    # existing monthly obligation. Not the same real-world quantity - see
    # module docstring.
    mapped["Total_Existing_EMIs"] = raw["loan_percent_income"] * mapped["Monthly_Net_Income"]

    # No numeric bureau score exists in this dataset for anyone, NTC or not.
    mapped["CIBIL_Score"] = np.nan

    # No gross-pay/deductions breakdown exists in this dataset either.
    mapped["Gross_Income"] = np.nan
    mapped["Total_Deductions"] = np.nan

    # Not present in a loan-application-level dataset (no bank-transaction data).
    mapped["Average_Monthly_Bank_Balance"] = np.nan
    mapped["Number_of_Bounced_Transactions_Last_6M"] = np.nan
    for i in range(1, 7):
        mapped[f"Cash_Flow_Month_{i}"] = np.nan

    # loan_amnt is a real, direct source for the requested amount; this
    # dataset has no loan-term/duration field, so tenure is NaN.
    mapped["Requested_Loan_Amount"] = raw["loan_amnt"]
    mapped["Requested_Tenure_Months"] = np.nan

    # The real historical outcome label - see module docstring for why this
    # must NOT be replaced by risk_modeling.labeling's synthetic label.
    mapped["Default"] = raw["loan_status"]

    mapped["_cred_hist_length"] = raw["cb_person_cred_hist_length"]
    mapped["_default_on_file"] = raw["cb_person_default_on_file"]

    return mapped


def engineer_is_first_loan(df: pd.DataFrame) -> pd.DataFrame:
    """Is_First_Loan = 1 for a zero credit-history length. The request also
    said "or a missing credit score" - but this dataset has no numeric
    credit-score column for anyone (see module docstring), so that clause
    would trivially flag every row and isn't applied here. If your real
    file does carry a genuine score column, add its missingness back into
    this condition.
    """
    df = df.copy()
    df["Is_First_Loan"] = (df["_cred_hist_length"] == 0).astype(int)
    df.loc[df["Is_First_Loan"] == 1, "CIBIL_Score"] = np.nan
    return df.drop(columns=["_cred_hist_length", "_default_on_file"])


def inject_synthetic_cibil_score(df: pd.DataFrame, ntc_rate: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """FABRICATES a CIBIL_Score for every row - this dataset has no real
    bureau-score column at all, so there is nothing to "reveal" here; every
    value produced by this function is invented, not sourced. Exists only
    for a deliberate ablation experiment ("does a populated CIBIL_Score
    change model performance vs. it being fully absent"), never as a stand-in
    for real Kaggle data. Callers must keep this output's filename distinct
    from the real mapped data (see SYNTHETIC_CIBIL_OUTPUT_PATH) so results
    from this run are never reported as genuine secondary-dataset numbers.

    Score is drawn ~ Normal(750, 60) clipped to [300, 900], the same
    distribution shape data_generation/applicant_generator.py uses for the
    fully-synthetic dataset, for consistency with the rest of this project.
    `ntc_rate` of rows are then reselected as NTC: Is_First_Loan=1 and
    CIBIL_Score reset to NaN, overwriting whatever load_and_map /
    engineer_is_first_loan had already set — real Is_First_Loan for this
    dataset was always 0 (see module docstring), so this replaces it with a
    synthetic NTC pattern instead of layering onto a real one.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    n = len(df)

    cibil = np.clip(rng.normal(750, 60, n), 300, 900).round().astype(float)
    is_first_loan = (rng.random(n) < ntc_rate).astype(int)
    cibil[is_first_loan == 1] = np.nan

    df["CIBIL_Score"] = cibil
    df["Is_First_Loan"] = is_first_loan
    return df


def clean_and_cap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    # Drop completely corrupted rows: missing core identity/income fields,
    # or values that are physically impossible (this dataset is known to
    # contain a handful of implausible ages, e.g. 144).
    df = df.dropna(subset=["Age", "Monthly_Net_Income"])
    df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]
    df = df[df["Monthly_Net_Income"] > 0]
    df = df.drop_duplicates()

    # Standard statistical caps (1st/99th percentile winsorization) on the
    # two continuous fields prone to extreme outliers, rather than dropping
    # those rows outright.
    for column in ["Monthly_Net_Income", "Total_Existing_EMIs"]:
        lower, upper = df[column].quantile([0.01, 0.99])
        df[column] = df[column].clip(lower, upper)

    dropped = before - len(df)
    print(f"Dropped {dropped} corrupted/invalid row(s) of {before} ({dropped / before:.1%}).")

    return df.reset_index(drop=True)


def main(synthetic_cibil: bool = False) -> None:
    print(f"Loading {RAW_KAGGLE_PATH} ...")
    mapped = load_and_map()

    print("Engineering Is_First_Loan (NTC) flag from credit-history length...")
    engineered = engineer_is_first_loan(mapped)

    print("Cleaning and capping outliers...")
    cleaned = clean_and_cap(engineered)

    if synthetic_cibil:
        print(
            "\n*** --synthetic-cibil: FABRICATING CIBIL_Score and Is_First_Loan for every "
            "row. This is not real Kaggle data - see inject_synthetic_cibil_score()'s "
            "docstring. Writing to a distinctly-named output so it can't be mistaken for "
            "the real mapping. ***\n"
        )
        cleaned = inject_synthetic_cibil_score(cleaned)
        output_path = SYNTHETIC_CIBIL_OUTPUT_PATH
    else:
        output_path = OUTPUT_PATH

    ntc_share = cleaned["Is_First_Loan"].mean()
    print(f"New-to-credit (NTC) share: {ntc_share:.2%}")
    print(f"CIBIL_Score non-null count: {cleaned['CIBIL_Score'].notna().sum()} of {len(cleaned)}")
    print(f"Real default rate (Default column, not synthesized): {cleaned['Default'].mean():.2%}")

    final = cleaned[FINAL_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"Wrote {len(final)} rows to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Map the Kaggle secondary dataset to our training schema")
    parser.add_argument(
        "--synthetic-cibil",
        action="store_true",
        help=(
            "Fabricate a CIBIL_Score + Is_First_Loan pattern (this dataset has no real "
            "bureau-score column at all). Writes to final_training_data_synthetic_cibil.csv, "
            "not final_training_data.csv, so it's never confused with the real mapping."
        ),
    )
    args = parser.parse_args()
    main(synthetic_cibil=args.synthetic_cibil)
