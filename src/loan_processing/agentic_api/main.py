"""FastAPI service wrapping the ML loan-assessment pipeline's document
extraction layer. This pipeline (data_generation -> document_simulation ->
risk_modeling -> extraction -> app.py) is otherwise only exposed via the
app.py CLI; this app gives it an actual HTTP surface for the frontend's
Upload flow - an extraction preview endpoint and a combined
extraction+scoring endpoint.

Extraction tries the regex parsers first (fast, free, exact when the upload
matches this project's own mock-document format) and falls back to the
Gemini multimodal extraction agent (extraction/gemini_extractor.py) only
when regex fails - a real-world document whose labels don't match, or a
file format regex can't handle at all (a photographed salary slip, a PDF
bank statement). Gemini is never called on the common/happy path, which
keeps the slow, occasionally-flaky external API call off the critical path
for the documents this app generates itself.

Run with:
    uvicorn loan_processing.agentic_api.main:app --app-dir src --reload

Not to be confused with loan_processing.api.main:app, the separate,
unrelated FastAPI service for the original rules-based document pipeline
(api/, extraction/extractor.py, DTI-based scoring) - see CLAUDE.md.
"""
from __future__ import annotations

import io
import math
import time
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from loan_processing.agentic_api.excel_manager import append_to_excel
from loan_processing.extraction.bank_statement_parser import expand_cash_flow_trend, parse_bank_statement
from loan_processing.extraction.gemini_extractor import extract_with_gemini_safe
from loan_processing.extraction.record_builder import FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS, calculate_age
from loan_processing.extraction.salary_slip_parser import parse_salary_slip
from loan_processing.extraction.validator import compute_discrepancy_flags
from loan_processing.extraction.validator import is_missing as _is_missing
from loan_processing.risk_modeling.emi import estimated_monthly_installment
from loan_processing.risk_modeling.train import FEATURE_COLUMNS, build_features

app = FastAPI(title="Loan Assessment Agent - Extraction & Scoring API")

# main.py lives at <repo_root>/src/loan_processing/agentic_api/main.py - same
# depth as risk_modeling/train.py, which derives REPO_ROOT the same way.
REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"

# The regex parsers (pdfplumber / pandas) are each anchored to exactly one
# format - no OCR, no PDF table extraction - so they only ever handle
# "application/pdf" for the salary slip and the xlsx mime types for the bank
# statement. Gemini reads documents visually, so it also accepts a
# photographed/scanned salary slip (JPEG) or a PDF bank statement - accepted
# here so those formats reach _extract_financial_fields() and fall through
# to Gemini, rather than being rejected before either extractor runs.
# Detected from the upload's filename extension rather than the
# browser-supplied content_type, which isn't always reliable.
_SALARY_SLIP_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
_BANK_STATEMENT_MIME_TYPES = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pdf": "application/pdf",
}


class ExtractionResult(BaseModel):
    method: str | None = None  # "regex" or "gemini" - whichever path actually supplied the data
    elapsed_seconds: float
    data: dict | None = None
    error: str | None = None
    discrepancy_flags: list[str] = Field(default_factory=list)


def _detect_mime_type(filename: str | None, allowed: dict[str, str], kind: str) -> str:
    suffix = Path(filename).suffix.lower() if filename else ""
    mime_type = allowed.get(suffix)
    if mime_type is None:
        raise ValueError(
            f"Unsupported {kind} file type '{suffix or '(unknown)'}' - expected one of: "
            f"{', '.join(sorted(allowed))}"
        )
    return mime_type


def _detect_salary_slip_mime_type(filename: str | None) -> str:
    return _detect_mime_type(filename, _SALARY_SLIP_MIME_TYPES, "salary slip")


def _detect_bank_statement_mime_type(filename: str | None) -> str:
    return _detect_mime_type(filename, _BANK_STATEMENT_MIME_TYPES, "bank statement")


def _extract_with_regex(salary_slip_bytes: bytes, bank_statement_bytes: bytes) -> dict:
    salary_fields = parse_salary_slip(io.BytesIO(salary_slip_bytes))
    bank_fields = parse_bank_statement(io.BytesIO(bank_statement_bytes))
    return {
        "Full_Name": salary_fields["Full_Name"],
        "Monthly_Net_Income": salary_fields["Monthly_Net_Income"],
        "Gross_Income": salary_fields["Gross_Income"],
        "Total_Deductions": salary_fields["Total_Deductions"],
        "Total_Existing_EMIs": bank_fields["Total_Existing_EMIs"],
        "Average_Monthly_Bank_Balance": bank_fields["Average_Monthly_Bank_Balance"],
        "Consistency_Verified": bank_fields.get("Consistency_Verified"),
        "Cash_Flow_Trend": bank_fields.get("Cash_Flow_Trend"),
        "Number_of_Bounced_Transactions_Last_6M": bank_fields.get("Number_of_Bounced_Transactions_Last_6M"),
    }


def _extract_financial_fields(
    salary_bytes: bytes,
    salary_mime_type: str,
    bank_bytes: bytes,
    bank_mime_type: str,
) -> tuple[dict, str]:
    """Tries the regex parsers first; falls back to the Gemini extraction
    agent when regex can't handle it - either it raised (labels didn't
    match this project's own mock-document format) or the upload isn't a
    PDF salary slip + XLSX bank statement to begin with (regex has no OCR
    and no PDF table extraction, so those formats never even reach the
    parsers). Raises ValueError, carrying both failure reasons, only if
    neither path recovers the data.
    """
    xlsx_mime_types = set(_BANK_STATEMENT_MIME_TYPES.values()) - {"application/pdf"}
    regex_error: str | None
    if salary_mime_type == "application/pdf" and bank_mime_type in xlsx_mime_types:
        try:
            return _extract_with_regex(salary_bytes, bank_bytes), "regex"
        except Exception as exc:  # noqa: BLE001 - fall through to Gemini
            regex_error = str(exc)
    else:
        regex_error = (
            "regex extraction only supports a PDF salary slip + XLSX bank statement - "
            f"got {salary_mime_type} / {bank_mime_type}"
        )

    gemini_data = extract_with_gemini_safe(
        salary_bytes, bank_bytes, salary_slip_mime_type=salary_mime_type, bank_statement_mime_type=bank_mime_type
    )
    gemini_error = gemini_data.pop("_error", None)
    if gemini_error is None:
        gemini_data["Consistency_Verified"] = None  # Gemini doesn't cross-check its own numbers
        return gemini_data, "gemini"

    raise ValueError(
        f"Could not extract financial data from the uploaded documents - regex parser: "
        f"{regex_error}; Gemini agent: {gemini_error}"
    )


@app.post("/extraction/preview", response_model=ExtractionResult)
async def preview_extraction(
    salary_slip: UploadFile = File(...),
    bank_statement: UploadFile = File(...),
) -> ExtractionResult:
    """Extracts the uploaded documents (regex first, Gemini fallback - see
    _extract_financial_fields) and returns the recovered fields, method, and
    timing, so the Upload flow can show a preview before committing to
    Score & Add to Ledger. A wrong file format or a failure on both
    extraction paths is captured in `error` rather than raising, so the
    frontend always gets a clean response to render.
    """
    salary_bytes = await salary_slip.read()
    bank_bytes = await bank_statement.read()

    start = time.perf_counter()
    try:
        salary_mime_type = _detect_salary_slip_mime_type(salary_slip.filename)
        bank_mime_type = _detect_bank_statement_mime_type(bank_statement.filename)
        data, method = _extract_financial_fields(salary_bytes, salary_mime_type, bank_bytes, bank_mime_type)
        discrepancy_flags = compute_discrepancy_flags(data)
        # Consistency_Verified is a computed cross-check /scoring/assess
        # relies on, not a field recovered from the documents - it doesn't
        # belong in a "here's what we found" preview.
        data.pop("Consistency_Verified", None)
        return ExtractionResult(
            method=method,
            elapsed_seconds=time.perf_counter() - start,
            data=data,
            discrepancy_flags=discrepancy_flags,
        )
    except Exception as exc:  # noqa: BLE001 - isolate any failure into a clean error response
        return ExtractionResult(elapsed_seconds=time.perf_counter() - start, error=str(exc))


# --- Scoring: extraction + the production risk model, in one call --------
#
# /extraction/preview above only previews the extracted fields - it never
# touches the risk model. This is the endpoint the frontend's Upload tab
# actually needs to turn an upload into a scored Ledger row: it extracts
# (regex first, Gemini fallback - see _extract_financial_fields), fills in
# the loan-application-form fields the documents can't supply, scores with
# the same production Logistic Regression model app.py loads, and returns a
# record shaped like a Ledger row.
#
# Applicant_ID is a loan-application-form field, not a document field: a
# real-world salary slip/bank statement has no machine-readable applicant
# identifier the way this project's own mock documents do (see
# salary_slip_parser.py), so it's always assigned by the caller rather than
# recovered from extraction or fabricated. Applicant_Name is also
# caller-supplied, and *is* cross-checked against the document - the salary
# slip's "Employee Name" field. Unlike the bank statement's own
# average-balance cross-check (a soft Consistency_Flag - the record still
# reaches the Ledger, just flagged for review), a name mismatch is a hard
# rejection: identity mismatch means these might not even be the right
# person's documents, so the request 422s and the caller has to correct the
# name and resubmit rather than the record silently landing in the Ledger.
#
# Age is caller-supplied too, but indirectly: the caller sends a date of
# birth (a document-printed age goes stale the moment it ages past the
# document's issue date, and a real document wouldn't reliably carry one
# anyway), and the endpoint derives Age itself via calculate_age() as of
# today - the model always sees a freshly-computed age, never a typed-in
# number or a document-recovered one.


class ScoringResult(BaseModel):
    Applicant_ID: str
    Age: int
    Employment_Type: str
    Monthly_Net_Income: float
    Gross_Income: float | None
    Total_Deductions: float | None
    Total_Existing_EMIs: float
    Requested_Loan_Amount: float
    Requested_Tenure_Months: int
    # The *requested* loan's own estimated monthly installment (emi.py, at
    # its assumed rate) - not extracted from any document, and distinct from
    # Total_Existing_EMIs (a real, extracted current-debt figure). Surfaces
    # the same affordability signal Requested_EMI_to_Income_Ratio feeds the
    # model with, which was otherwise invisible to a human reviewer.
    Estimated_Monthly_EMI: float
    Average_Monthly_Bank_Balance: float
    Number_of_Bounced_Transactions_Last_6M: int | None
    Cash_Flow_Trend: list[float] | None  # 6 months, oldest->newest; None if extraction couldn't recover it
    Extracted_Full_Name: str | None  # name recovered from the salary slip, if any
    Name_Match: bool | None  # None when the document didn't yield a name to check against
    Bank_Consistency_Verified: bool | None  # None when the statement had no stated average to check against
    Extraction_Confidence: float
    Consistency_Flag: bool
    Processing_Latency: float
    Default_Probability: float
    Classification_Verdict: int
    Top_Risk_Drivers: list[str]
    Extraction_Method: str  # "regex" or "gemini" - whichever path actually supplied the data
    Model_Used: str | None  # the specific Gemini model tier that served the request; None on the regex path
    Fallback_Triggered: bool  # True if Gemini's Pro tier hit a quota error and Flash had to cover for it
    Missing_Fields: list[str]
    Discrepancy_Flags: list[str]


_RISK_DRIVER_LABELS = {
    "high_dti": "Spends a large share of income on existing debt",
    "bounced": "Multiple bounced payments in bank history",
    "low_balance": "Low savings compared to income",
    "high_lti": "Loan amount is large compared to income",
    "long_tenure": "Requested a long repayment period",
}


def _normalize_name(name: str) -> str:
    # Case/whitespace-insensitive comparison - "Rahul   Verma" on a bank
    # statement's fixed-width layout should still match "Rahul Verma" typed
    # into the form.
    return " ".join(name.strip().lower().split())


@lru_cache(maxsize=1)
def _load_scoring_artifacts():
    # Same artifacts app.py loads for the CLI - the production model. The
    # pipeline bundles preprocessing (one-hot encoding Employment_Type,
    # median-imputing + scaling the numeric features) and the classifier
    # into one fitted sklearn Pipeline - see risk_modeling/train.py.
    pipeline = joblib.load(MODELS_DIR / "logistic_regression_pipeline.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    return pipeline, feature_columns


def _compute_top_risk_drivers(record: dict) -> list[str]:
    income = record["Monthly_Net_Income"]
    emi = record["Total_Existing_EMIs"]
    balance = record["Average_Monthly_Bank_Balance"]
    loan_amount = record["Requested_Loan_Amount"]
    tenure = record["Requested_Tenure_Months"]
    bounced = record["Number_of_Bounced_Transactions_Last_6M"]

    dti = emi / income if income else 0.0
    lti = loan_amount / (income * 12) if income else 0.0

    drivers: list[str] = []
    if dti > 0.35:
        drivers.append(_RISK_DRIVER_LABELS["high_dti"])
    if not _is_missing(bounced) and bounced >= 2:
        drivers.append(_RISK_DRIVER_LABELS["bounced"])
    if balance < income * 0.7:
        drivers.append(_RISK_DRIVER_LABELS["low_balance"])
    if lti > 3:
        drivers.append(_RISK_DRIVER_LABELS["high_lti"])
    if not _is_missing(tenure) and tenure >= 72 and len(drivers) < 2:
        drivers.append(_RISK_DRIVER_LABELS["long_tenure"])
    return drivers[:3]


def _extraction_confidence(method: str, missing_fields: list[str]) -> float:
    # Neither extractor emits a real per-field confidence score, so this is a
    # heuristic derived from real signals (which extractor supplied the data,
    # how many application-form-only fields are still unfilled) rather than
    # a calibrated probability.
    base = 97.5 if method == "regex" else 91.0
    return round(max(base - 6 * len(missing_fields), 55.0), 1)


@app.post("/scoring/assess", response_model=ScoringResult)
async def score_application(
    salary_slip: UploadFile = File(...),
    bank_statement: UploadFile = File(...),
    applicant_id: str = Form(...),
    applicant_name: str = Form(...),
    date_of_birth: str = Form(...),
    employment_status: Literal["Salaried", "Self-Employed", "Business Owner"] = Form(...),
    requested_loan_amount: float = Form(...),
    is_first_loan: bool = Form(False),
    requested_tenure_months: int = Form(...),
) -> ScoringResult:
    start = time.perf_counter()
    salary_bytes = await salary_slip.read()
    bank_bytes = await bank_statement.read()

    applicant_id = applicant_id.strip()
    if not applicant_id:
        raise HTTPException(
            status_code=422,
            detail="Applicant ID is required - real-world documents don't carry a machine-readable one, "
            "so it has to be assigned here.",
        )

    try:
        parsed_date_of_birth = date.fromisoformat(date_of_birth)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Date of birth must be in YYYY-MM-DD format: {exc}"
        ) from exc
    age = calculate_age(parsed_date_of_birth)
    if age < 0:
        raise HTTPException(status_code=422, detail="Date of birth can't be in the future.")

    try:
        salary_mime_type = _detect_salary_slip_mime_type(salary_slip.filename)
        bank_mime_type = _detect_bank_statement_mime_type(bank_statement.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        extracted, method = _extract_financial_fields(salary_bytes, salary_mime_type, bank_bytes, bank_mime_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    extracted_full_name = extracted.get("Full_Name")
    name_match: bool | None = None
    if isinstance(extracted_full_name, str) and extracted_full_name.strip():
        name_match = _normalize_name(extracted_full_name) == _normalize_name(applicant_name)
    else:
        extracted_full_name = None  # normalize "" / non-string noise to a clean None

    if name_match is False:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The applicant name entered (\"{applicant_name}\") doesn't match the name on the "
                f"salary slip (\"{extracted_full_name}\"). Correct the Applicant Name field and resubmit."
            ),
        )

    hard_required = ["Monthly_Net_Income", "Total_Existing_EMIs", "Average_Monthly_Bank_Balance"]
    hard_missing = [f for f in hard_required if _is_missing(extracted.get(f))]
    if hard_missing:
        raise HTTPException(
            status_code=422,
            detail=f"Extraction ({method}) could not recover required field(s): {', '.join(hard_missing)}.",
        )

    record: dict = {
        "Applicant_ID": applicant_id,
        "Age": age,
        "Employment_Type": employment_status,
        "Monthly_Net_Income": extracted["Monthly_Net_Income"],
        "Gross_Income": extracted.get("Gross_Income", math.nan),
        "Total_Deductions": extracted.get("Total_Deductions", math.nan),
        "Total_Existing_EMIs": extracted["Total_Existing_EMIs"],
        "Average_Monthly_Bank_Balance": extracted["Average_Monthly_Bank_Balance"],
        "Requested_Loan_Amount": requested_loan_amount,
        "Requested_Tenure_Months": requested_tenure_months,
        "Is_First_Loan": 1 if is_first_loan else 0,
        "Number_of_Bounced_Transactions_Last_6M": (
            math.nan if _is_missing(extracted.get("Number_of_Bounced_Transactions_Last_6M"))
            else extracted["Number_of_Bounced_Transactions_Last_6M"]
        ),
        **expand_cash_flow_trend(extracted.get("Cash_Flow_Trend")),
    }

    discrepancy_flags = compute_discrepancy_flags(extracted)
    missing_fields = [f for f in FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS if _is_missing(record.get(f))]

    model_input = build_features(pd.DataFrame([record]))[FEATURE_COLUMNS]
    pipeline, feature_columns = _load_scoring_artifacts()
    assert list(model_input.columns) == feature_columns, (
        "Feature order from the extraction layer doesn't match the saved model's feature_columns.pkl."
    )
    probability_of_default = float(pipeline.predict_proba(model_input)[0, 1])
    classification_verdict = 1 if probability_of_default >= 0.5 else 0

    bank_consistency = extracted.get("Consistency_Verified")
    # No missing required fields and no bank-statement cross-check mismatch -
    # mirrors app.py's "don't auto-approve on an incomplete picture" rule.
    # (A name mismatch is handled separately, above, as a hard 422 rather
    # than a soft flag - by this point name_match can only be True or None.)
    consistency_flag = bool(bank_consistency is not False and not missing_fields)

    top_risk_drivers = _compute_top_risk_drivers(record)
    extraction_confidence = _extraction_confidence(method, missing_fields)

    cash_flow_trend = [record[f"Cash_Flow_Month_{i}"] for i in range(1, 7)]
    if any(_is_missing(v) for v in cash_flow_trend):
        cash_flow_trend = None

    estimated_monthly_emi = float(
        estimated_monthly_installment(record["Requested_Loan_Amount"], record["Requested_Tenure_Months"])
    )

    return ScoringResult(
        Applicant_ID=applicant_id,
        Age=int(record["Age"]),
        Employment_Type=record["Employment_Type"],
        Monthly_Net_Income=float(record["Monthly_Net_Income"]),
        Gross_Income=None if _is_missing(record["Gross_Income"]) else float(record["Gross_Income"]),
        Total_Deductions=None if _is_missing(record["Total_Deductions"]) else float(record["Total_Deductions"]),
        Total_Existing_EMIs=float(record["Total_Existing_EMIs"]),
        Requested_Loan_Amount=float(record["Requested_Loan_Amount"]),
        Requested_Tenure_Months=requested_tenure_months,
        Estimated_Monthly_EMI=round(estimated_monthly_emi, 2),
        Average_Monthly_Bank_Balance=float(record["Average_Monthly_Bank_Balance"]),
        Number_of_Bounced_Transactions_Last_6M=(
            None if _is_missing(record["Number_of_Bounced_Transactions_Last_6M"])
            else int(record["Number_of_Bounced_Transactions_Last_6M"])
        ),
        Cash_Flow_Trend=cash_flow_trend,
        Extracted_Full_Name=extracted_full_name,
        Name_Match=name_match,
        Bank_Consistency_Verified=bank_consistency,
        Extraction_Confidence=extraction_confidence,
        Consistency_Flag=consistency_flag,
        Processing_Latency=round(time.perf_counter() - start, 2),
        Default_Probability=round(probability_of_default, 4),
        Classification_Verdict=classification_verdict,
        Top_Risk_Drivers=top_risk_drivers,
        Extraction_Method=method,
        Model_Used=extracted.get("model_used"),
        Fallback_Triggered=bool(extracted.get("fallback_triggered")),
        Missing_Fields=missing_fields,
        Discrepancy_Flags=discrepancy_flags,
    )


class SaveApplicantRequest(BaseModel):
    Applicant_ID: str
    Employment_Type: str | None = None
    Monthly_Net_Income: float | None = None
    Requested_Loan_Amount: float | None = None
    Default_Probability: float | None = None
    Classification_Verdict: int | None = None
    Final_Status: str  # "approved" or "rejected" - the HITL decision, not the model's verdict


@app.post("/save-applicant")
def save_applicant(payload: SaveApplicantRequest) -> dict[str, str]:
    """Called by the frontend's Pending Reviews dashboard (Quick Approve) and
    Document Inspection Drawer (Approve/Reject) whenever a HITL decision is
    made, to append it to the persistent Excel summary. A plain `def` route
    - Starlette runs sync path operations in a thread pool automatically, so
    this blocking pandas/openpyxl write never blocks the event loop.
    """
    append_to_excel(payload.model_dump())
    return {"status": "saved"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
