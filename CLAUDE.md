# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two co-existing, **not integrated** subsystems:

1. **AI-powered Loan Assessment Agent** (the active line of work) — an end-to-end pipeline that generates a synthetic applicant dataset, trains an XGBoost credit-risk model on it, simulates mock salary-slip/bank-statement documents, extracts structured data back out of those documents, and scores an applicant via a CLI (`app.py`) that prints a Customer Risk Analysis Report.
2. **Original rules-based document pipeline** (`api/`, `ingestion/`, `extraction/extractor.py`, `validation/`, `scoring/`, `decisioning/`, `output/`, `pipeline.py`) — an earlier, self-contained FastAPI service that accepts uploaded loan documents and returns an Excel decision report using hand-written regex extraction and a DTI-based scoring formula. Still functional and tested, but a separate prototype — it does not call into the ML model or vice versa.

## Commands

Setup (from repo root):
```
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev]"         # or: pip install -r requirements-dev.txt
```

Run all tests: `pytest` (31 tests as of the last checkpoint)

Run a single test: `pytest tests/unit/test_scoring.py::test_high_debt_to_income_lowers_score`

### Rebuilding the ML pipeline from scratch (all steps are seeded/reproducible)

Run in order — each stage's output feeds the next:
```
python -m loan_processing.data_generation.applicant_generator      # -> data/raw/applicants.csv (1000 rows, seed=42)
python -m loan_processing.document_simulation.generate_mock_documents  # -> data/mock_documents/ (5 sampled applicants, seed=42)
python -m loan_processing.risk_modeling.train                      # -> data/processed/applicants_labeled.csv, models/*.pkl, reports/figures/*.png
python -m loan_processing.extraction.extract_applicant [Applicant_ID]  # demo: extract one applicant, print DataFrame + ground-truth cross-check
python app.py <Applicant_ID>                                       # e.g. python app.py APP00522 — full end-to-end report
```
All these outputs are already committed, so none of this needs to be re-run unless you want fresh data or the seeds change.

Run the original rules-based API: `uvicorn loan_processing.api.main:app --app-dir src --reload`

## Architecture — AI Loan Assessment Agent

```
data_generation -> document_simulation -> risk_modeling (label + train) -> extraction -> app.py
```

- **`data_generation/`** (`schemas.py`, `applicant_generator.py`) — generates 1000 synthetic applicants with *correlated*, grounded numeric distributions (income drives EMI burden and bank balance; EMI burden and bounced transactions pull CIBIL score down; loan amount scales with income) rather than independent random noise. `full_name`/`employer_name` are included beyond the originally-requested columns specifically so document_simulation has identity text to render.

- **`risk_modeling/labeling.py`** — the raw dataset has features but no ground-truth outcome, so this derives a synthetic `Default` target: a z-scored logistic combination of the same risk factors, with its intercept calibrated (binary search) to hit a target ~17% base rate, plus Gaussian noise so the label isn't perfectly separable from the features. This is what makes the classifier evaluation in `train.py` meaningful rather than trivial.

- **`risk_modeling/train.py`** — `FEATURE_COLUMNS` and `build_features()` here are the **single source of truth** for the model's expected input schema (11 features, including `Is_Self_Employed` one-hot encoding and two engineered ratios). `extraction/record_builder.py` imports these directly rather than re-deriving them, so extraction output can never drift out of sync with what the saved model actually expects. Trains Logistic Regression (baseline) + XGBoost, applies **SMOTE only to the training split** (never the test set — that would leak), evaluates with precision/recall/F1/AUC, and saves `models/xgboost_model.pkl`, `models/logistic_regression_model.pkl`, `models/scaler.pkl`, `models/feature_columns.pkl`.

- **`document_simulation/`** (`salary_slip_pdf.py`, `bank_statement_xlsx.py`) — renders mock documents from a sampled applicant row. The bank statement simulator solves for the account's opening balance so the statement's **day-weighted** average balance (not a naive per-row mean, which over-weights the elevated post-salary-credit period) lands exactly on the applicant's stored `Average_Monthly_Bank_Balance`. Self-Employed applicants get "INCOME STATEMENT"/"BUSINESS INCOME" wording instead of "SALARY SLIP"/"SALARY".

- **`extraction/`** — `salary_slip_parser.py` (pdfplumber + regex, anchored to the exact labels `salary_slip_pdf.py` writes) and `bank_statement_parser.py` (pandas; recomputes the same day-weighted average independently and cross-checks it against the statement's own stated header value as a verification step — this genuinely validates extraction fidelity, it isn't just reading the header). `record_builder.py` combines both and builds a model-input DataFrame via `risk_modeling.train.build_features`.

  **Known gap, by design, not oversight**: a salary slip + bank statement can only supply 7 of the model's 11 features. `CIBIL_Score`, `Requested_Loan_Amount`, `Requested_Tenure_Months`, and `Number_of_Bounced_Transactions_Last_6M` come back as explicit `NaN` (`FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS` in `record_builder.py`) rather than fabricated defaults. Closing this gap needs a credit-bureau integration, a loan-application-form extractor, and bounced-transaction line items added to the mock bank statement generator.

- **`app.py`** (repo root, not under `src/`) — the CLI entry point. Loads `models/*.pkl`, scores the extracted (partially-`NaN`) feature vector — both `StandardScaler.transform` and XGBoost's `predict_proba` tolerate `NaN` natively (confirmed empirically, not assumed) — and prints a 4-section report. **Deliberate business-rule override**: if any of the 4 required-but-missing fields are absent, the Processing Recommendation is always "FLAG FOR MANUAL REVIEW" regardless of what the model predicts, because a real underwriting system shouldn't auto-approve on an incomplete picture.

### Testing conventions

- `tests/unit/` covers each stage in isolation (data generation reproducibility/ranges, labeling calibration, document rendering round-trips, extraction fidelity, `app.py`'s report/recommendation logic) — no dependency on files under `data/` actually existing on disk; fixtures build everything in `tmp_path`.
- `tests/integration/test_api.py` drives the original FastAPI app with `TestClient`.
- The original rules-based pipeline's tests (`test_scoring.py`, `test_validation.py`, `test_output.py`) are unrelated to the ML pipeline's tests.

## Where things stand

All five phases are built, tested, and executed end-to-end: synthetic data (1000 applicants) -> labeled dataset -> trained XGBoost model (AUC ~0.94) + diagnostic plots -> 5 mock document sets -> extraction verified against ground truth on all 5 -> `app.py` producing full reports for multiple applicants. Everything is committed, including generated artifacts (`data/`, `models/`, `reports/figures/`) — regeneration commands above are for if you want fresh/different data, not required to pick the project back up.

Natural next steps if continuing: an application-form extractor + a simulated credit-bureau lookup to close the 4-field gap above, and/or wiring `app.py` to batch-process all sampled applicants instead of one at a time.
