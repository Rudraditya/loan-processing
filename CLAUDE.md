# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Four co-existing subsystems, mostly **not integrated** with each other:

1. **AI-powered Loan Assessment Agent** (the active line of work) — an end-to-end pipeline that generates a synthetic applicant dataset, trains Logistic Regression and XGBoost credit-risk models on it (Logistic Regression is production — see "Current model status"), simulates mock salary-slip/bank-statement documents, extracts structured data back out of those documents, and scores an applicant via a CLI (`app.py`) that prints a Customer Risk Analysis Report.
2. **Original rules-based document pipeline** (`api/`, `ingestion/`, `extraction/extractor.py`, `validation/`, `scoring/`, `decisioning/`, `output/`, `pipeline.py`) — an earlier, self-contained FastAPI service that accepts uploaded loan documents and returns an Excel decision report using hand-written regex extraction and a DTI-based scoring formula. Still functional and tested, but a separate prototype — it does not call into the ML model or vice versa.
3. **`agentic_api/`** — a FastAPI service standing up an HTTP surface in front of the ML pipeline's document extraction layer (which is otherwise CLI-only via `app.py`). Its one endpoint runs the existing regex parsers and a new Gemini-based multimodal extraction agent (`extraction/gemini_extractor.py`) side-by-side for benchmarking. See "Gemini extraction agent" below.
4. **`frontend/`** — a React + Vite + Zustand dashboard (Ledger/Insights/Pipeline/Upload tabs). Partially integrated: the Upload tab's document-extraction step calls the real `agentic_api` `/extraction/benchmark` endpoint (via a Vite dev-server proxy at `/api`, see `frontend/vite.config.js`); the Ledger and Insights tabs still run entirely on client-side mock data (`frontend/src/store/useLoanStore.js`) since `agentic_api` has no scoring endpoint for them to call yet. See "Where things stand" below.

## Commands

Setup (from repo root):
```
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev]"         # or: pip install -r requirements-dev.txt
```

Run all tests: `pytest` (48 tests as of the last checkpoint)

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

Run the Gemini-benchmarking API: `uvicorn loan_processing.agentic_api.main:app --app-dir src --reload` (needs `GEMINI_API_KEY` in a `.env` file at repo root — copy `.env.example` — or the Gemini side of every request returns a clean per-field error while the regex side still succeeds)

## Architecture — AI Loan Assessment Agent

```
data_generation -> document_simulation -> risk_modeling (label + train) -> extraction -> app.py
```

- **`data_generation/`** (`schemas.py`, `applicant_generator.py`) — generates 1000 synthetic applicants with *correlated*, grounded numeric distributions (income drives EMI burden and bank balance; EMI burden and bounced transactions pull CIBIL score down; loan amount scales with income) rather than independent random noise. `full_name`/`employer_name` are included beyond the originally-requested columns specifically so document_simulation has identity text to render.

- **`risk_modeling/labeling.py`** — the raw dataset has features but no ground-truth outcome, so this derives a synthetic `Default` target: a z-scored logistic combination of the same risk factors, with its intercept calibrated (binary search) to hit a target ~17% base rate, plus Gaussian noise so the label isn't perfectly separable from the features. This is what makes the classifier evaluation in `train.py` meaningful rather than trivial.

- **`risk_modeling/train.py`** — `FEATURE_COLUMNS` and `build_features()` here are the **single source of truth** for the model's expected input schema (11 features, including `Is_Self_Employed` one-hot encoding, `Is_First_Loan`, and two engineered ratios). **`CIBIL_Score` (a credit-bureau score) is deliberately excluded** — the model relies entirely on income/EMI/balance/repayment-behavior signals instead of a bureau pull; it's still generated as a synthetic applicant attribute (`data_generation/`) and still feeds the synthetic `Default` label in `labeling.py`, it's just never handed to the classifier as an input. `extraction/record_builder.py` imports `FEATURE_COLUMNS`/`build_features()` directly rather than re-deriving them, so extraction output can never drift out of sync with what the saved model actually expects. Trains Logistic Regression + XGBoost, applies **SMOTE only to the training split** (never the test set — that would leak; on the synthetic production dataset training now has no per-row missingness at all, since `CIBIL_Score` was the only field that ever had it — but the NaN-aware SMOTE path is retained because rows with any missing value, or, on a secondary dataset, whole columns missing for every row, are excluded from SMOTE's oversampling and kept unresampled instead, since it can't interpolate a genuinely-absent feature), evaluates with precision/recall/F1/AUC, and saves `models/xgboost_model.pkl`, `models/logistic_regression_model.pkl`, `models/scaler.pkl` (fit on the raw NaN-preserving data XGBoost trains on), `models/logistic_regression_scaler.pkl` + `models/logistic_regression_medians.pkl` (Logistic Regression's own scaler and per-feature training medians — it's fit on median-imputed data, so it needs its own scaler distinct from `scaler.pkl`, and the medians must be persisted so a caller can reproduce the same imputation at inference time), and `models/feature_columns.pkl`.

  `main()` takes four optional overrides, all defaulting to the original synthetic/production behavior: `data_path` (train on a different CSV — see secondary-dataset section below), `exclude_features` (drop column(s) from *this run's* feature set only, without touching the shared `FEATURE_COLUMNS` contract `record_builder.py` depends on), and `models_dir`/`figures_dir` (save artifacts somewhere other than `models/`/`reports/figures/`, so an experimental run can never silently overwrite the production model). Exposed as `--data-path`, `--exclude-features`, `--models-dir`, `--figures-dir` on the CLI.

- **`document_simulation/`** (`salary_slip_pdf.py`, `bank_statement_xlsx.py`) — renders mock documents from a sampled applicant row. The bank statement simulator solves for the account's opening balance so the statement's **day-weighted** average balance (not a naive per-row mean, which over-weights the elevated post-salary-credit period) lands exactly on the applicant's stored `Average_Monthly_Bank_Balance`. Self-Employed applicants get "INCOME STATEMENT"/"BUSINESS INCOME" wording instead of "SALARY SLIP"/"SALARY".

- **`extraction/`** — `salary_slip_parser.py` (pdfplumber + regex, anchored to the exact labels `salary_slip_pdf.py` writes) and `bank_statement_parser.py` (pandas; recomputes the same day-weighted average independently and cross-checks it against the statement's own stated header value as a verification step — this genuinely validates extraction fidelity, it isn't just reading the header). `record_builder.py` combines both and builds a model-input DataFrame via `risk_modeling.train.build_features`.

  **Known gap, by design, not oversight**: a salary slip + bank statement can only directly supply 6 of the model's 11 features non-`NaN` (a 7th, the derived `Loan_to_Annual_Income_Ratio`, also ends up `NaN` since it depends on `Requested_Loan_Amount`). `Is_First_Loan`, `Requested_Loan_Amount`, `Requested_Tenure_Months`, and `Number_of_Bounced_Transactions_Last_6M` come back as explicit `NaN` (`FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS` in `record_builder.py`) rather than fabricated defaults. `build_applicant_record()` takes an optional `is_first_loan` override standing in for the (not-yet-built) loan-application-form extractor. Closing the rest of the gap needs bounced-transaction line items added to the mock bank statement generator (there's no `CIBIL_Score`/credit-bureau gap to close — the model doesn't use one).

  **`gemini_extractor.py`** — a second extraction strategy for the same 7 document-derivable fields (plus the always-empty `Number_of_Bounced_Transactions_Last_6M`, since the mock statements don't render bounced-transaction rows for either extractor to find), using Gemini's multimodal document understanding instead of regex, via the deprecated `google-generativeai` SDK (Google has EOL'd it in favor of `google-genai` — flagged, not yet migrated). Reads the salary slip PDF natively; the bank statement is rendered to CSV text first since Gemini can't parse raw `.xlsx` binary. Output is constrained via Pydantic `response_schema` + `response_mime_type="application/json"`, not prompt instructions alone. Needs `GEMINI_API_KEY`; `extract_with_gemini_safe()` never raises, returning a null-filled dict + `_error` key on any failure (missing key, rate limit, schema violation) instead. Exercised via `agentic_api/main.py`'s `/extraction/benchmark` endpoint, which runs both extractors on the same upload and returns both results with timing — a failure in one path doesn't take down the other.

- **`app.py`** (repo root, not under `src/`) — the CLI entry point. Loads the **Logistic Regression** model + its own scaler + its own training medians (`logistic_regression_model.pkl` / `logistic_regression_scaler.pkl` / `logistic_regression_medians.pkl`), scores the extracted (partially-`NaN`) feature vector, and prints a report (5 sections for a first-time/NTC applicant, 4 otherwise). Logistic Regression, unlike XGBoost, can't accept `NaN` at all, so `score_applicant()` imputes any missing feature with its training-set median (persisted by `train.py`) before scaling and predicting — this is a real behavior difference from the previous XGBoost-in-production setup, which routed missing values via learned split directions instead of imputing. It does **not** weaken the report's honesty about missing data, though: **deliberate business-rule override** — if any required-but-missing field is absent, the Processing Recommendation is always "FLAG FOR MANUAL REVIEW" regardless of what the model predicts, because a real underwriting system shouldn't auto-approve on an incomplete picture; that check reads the record directly; it doesn't care what the model did internally. Accepts an optional `--first-loan {yes,no}` flag standing in for the application-form input; when `yes`, the report adds an NTC section noting the model doesn't use a bureau score for *any* applicant, so a first-time borrower isn't weighted differently on that front — the section flags instead that they have no repayment track record behind the alternative-data signals (`Monthly_Net_Income`, `Average_Monthly_Bank_Balance`) the assessment already rests on.

## Secondary-dataset benchmarking (`src/prepare_secondary_data.py`)

Maps a real Kaggle dataset onto the training schema above, as an academic-benchmark comparison against the synthetic data — kept entirely separate from the production pipeline (see "Current model status" below for what's actually loaded where).

- **Input**: `data/raw_kaggle_credit_data.csv` — the well-known Kaggle "Credit Risk Dataset" (`person_age`, `person_income`, `loan_amnt`, `loan_percent_income`, `loan_status`, `cb_person_default_on_file`, `cb_person_cred_hist_length`, ...). **Not committed** (large downloaded file, currently untracked) — place it there yourself to rerun this.
- **Real, structural gap, not a bug**: this dataset has no bureau-score column, no bank-transaction data, and no employment-type category at all. `CIBIL_Score` (moot either way now that it's not a model feature — see "Current model status" below), `Average_Monthly_Bank_Balance`, `Number_of_Bounced_Transactions_Last_6M`, and `Requested_Tenure_Months` come back `NaN` for every single row (not just NTC ones); `Employment_Type` is defaulted to `"Salaried"` for everyone. Only `Applicant_ID` (synthesized), `Age`, `Monthly_Net_Income`, `Requested_Loan_Amount`, and `Is_First_Loan` (from `cb_person_cred_hist_length == 0` — 0% of real rows meet this, so it's a genuine finding, not a script bug) carry real signal. `loan_status` is carried through unchanged as `Default` — a real label, so `train.py` skips `labeling.py`'s synthetic-label generation whenever a `Default` column is already present.
- **`inject_synthetic_cibil_score()`** (`--synthetic-cibil` flag) — deliberately **fabricates** a `CIBIL_Score`/`Is_First_Loan` pattern (drawn independently of the real outcome) for a controlled ablation ("does a populated-but-uninformative feature change model behavior"). Writes to a distinctly-named `data/final_training_data_synthetic_cibil.csv`, never the real mapping's output file, specifically so a fabricated run can never be mistaken for a real-data benchmark result.
- Both outputs feed `risk_modeling/train.py --data-path <file>`.

### Current model status

`CIBIL_Score` was removed from `FEATURE_COLUMNS` (production model schema is now 11 features, not 12) and both models were retrained — see "risk_modeling/train.py" above for why. **`app.py` was then switched from XGBoost to Logistic Regression as the production model** — on this 11-feature (no-CIBIL) schema, Logistic Regression's AUC (0.9048) now exceeds XGBoost's (0.8879), a reversal from the pre-CIBIL-removal numbers where XGBoost led.

| Location | Trained on | Features | Model | AUC | Status |
|---|---|---|---|---|---|
| `models/logistic_regression_*.pkl` (**production** — what `app.py` loads) | Synthetic `data/raw/applicants.csv` | 11 (`CIBIL_Score` excluded) | Logistic Regression | 0.9048 | Current |
| `models/xgboost_model.pkl` | Synthetic `data/raw/applicants.csv` | 11 (`CIBIL_Score` excluded) | XGBoost | 0.8879 | Trained every run alongside the production model, but not loaded by `app.py` |
| `models/experiments/kaggle_no_cibil/` | Real Kaggle mapping, `CIBIL_Score` excluded | 11 | XGBoost | 0.826 | Experiment, not loaded by anything. Predates the production schema change above but happens to use the same 11-feature set by coincidence (it was a deliberate ablation at the time; now it's just what the production schema always excludes). |

A run on the real Kaggle mapping with `CIBIL_Score` left in as an always-`NaN` column (AUC 0.828) and a run with a fabricated, intentionally-uncorrelated `CIBIL_Score` (AUC 0.801) were also produced during benchmarking but weren't saved to disk under their own directory — their metrics are only recorded in conversation history, not reproducible from a committed artifact. Since `CIBIL_Score` is no longer a feature the production code path can even accept, regenerating either would need a temporary local re-add to `FEATURE_COLUMNS`, not just the `--data-path`/`--synthetic-cibil` flags.

**Do not use `models/experiments/*` results as if they were the production model's numbers, and do not report the fabricated-CIBIL run's metrics as reflecting real secondary-dataset performance** — it was an explicit, labeled ablation on invented values, not a finding about CIBIL scores.

### Testing conventions

- `tests/unit/` covers each stage in isolation (data generation reproducibility/ranges, labeling calibration, document rendering round-trips, extraction fidelity, `app.py`'s report/recommendation logic, `gemini_extractor.py`'s schema validation and error handling via a mocked SDK — no real API key needed to run the suite) — no dependency on files under `data/` actually existing on disk; fixtures build everything in `tmp_path`.
- `test_prepare_secondary_data.py` verifies `prepare_secondary_data.py`'s mapping/NTC-engineering/cleaning/synthetic-CIBIL-injection logic against a small hand-built fixture matching the *assumed* Kaggle schema — it does not depend on the real (uncommitted) `raw_kaggle_credit_data.csv` and proves nothing about real-world numbers, only that the script's own logic is correct. `pyproject.toml` sets `pythonpath = ["src"]` so this loose top-level script (not part of the `loan_processing` package) is importable in tests.
- `tests/integration/test_api.py` drives the original FastAPI app with `TestClient`; `tests/integration/test_agentic_api.py` drives the new one the same way, using the real committed mock documents, with the Gemini side exercised only through its no-API-key failure path (deterministic, no network call).
- The original rules-based pipeline's tests (`test_scoring.py`, `test_validation.py`, `test_output.py`) are unrelated to the ML pipeline's tests.

## Where things stand

All five phases are built, tested, and executed end-to-end: synthetic data (1000 applicants, including ~15% new-to-credit — `CIBIL_Score` is still generated for these rows and still genuinely `NaN` for NTC ones, it's just no longer handed to the model as a feature for *any* applicant) -> labeled dataset -> trained Logistic Regression + XGBoost models + diagnostic plots -> 5 mock document sets -> extraction verified against ground truth on all 5 -> `app.py` producing full reports for multiple applicants, with a dedicated NTC section when applicable. A Gemini-based extraction agent now runs alongside the regex parsers behind `agentic_api/`'s benchmark endpoint. **`app.py` currently loads the Logistic Regression model, with `CIBIL_Score` excluded from its feature set (AUC 0.9048) — see "Current model status" above before assuming any other model/number is what's loaded.** Everything from the original five phases is committed, including generated artifacts (`data/`, `models/`, `reports/figures/`) — regeneration commands above are for if you want fresh/different data, not required to pick the project back up. The secondary-dataset files (`data/raw_kaggle_credit_data.csv`, `data/final_training_data*.csv`, `models/experiments/`) are **not** committed as of this writing.

The `frontend/` (React + Vite + Zustand) dashboard is a fourth, UI-only subsystem layered on top of the three above — the Ledger/Insights/Pipeline tabs still run on client-side mock data (`frontend/src/store/useLoanStore.js`, a seeded PRNG generator, not wired to any backend), while the Upload tab's document-extraction step is wired to `agentic_api`'s `/extraction/benchmark` endpoint for real (via a Vite dev-server proxy at `/api`) — it does not call the risk model, since `agentic_api` has no scoring endpoint yet.

Natural next steps if continuing: a real loan-application-form extractor to close the remaining field gap above; migrating `gemini_extractor.py` off the EOL'd `google-generativeai` SDK onto `google-genai`; deciding whether the Kaggle secondary-dataset files belong in version control or stay local-only; wiring `app.py` to batch-process all sampled applicants instead of one at a time; and/or adding a scoring endpoint to `agentic_api` so the frontend's Ledger/Insights tabs could run on real data instead of the mock store.
