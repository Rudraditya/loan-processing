"""FastAPI service wrapping the ML loan-assessment pipeline's document
extraction layer. This pipeline (data_generation -> document_simulation ->
risk_modeling -> extraction -> app.py) is otherwise only exposed via the
app.py CLI; this app gives it an actual HTTP surface, specifically for
benchmarking the existing regex-based parsers against the new Gemini
multimodal extraction agent (extraction/gemini_extractor.py).

Run with:
    uvicorn loan_processing.agentic_api.main:app --app-dir src --reload

Not to be confused with loan_processing.api.main:app, the separate,
unrelated FastAPI service for the original rules-based document pipeline
(api/, extraction/extractor.py, DTI-based scoring) - see CLAUDE.md.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from loan_processing.extraction.bank_statement_parser import parse_bank_statement
from loan_processing.extraction.gemini_extractor import extract_with_gemini_safe
from loan_processing.extraction.salary_slip_parser import parse_salary_slip

app = FastAPI(title="Loan Assessment Agent - Extraction Benchmark API")

# The regex parsers (pdfplumber / pandas) are each anchored to exactly one
# format - no OCR, no PDF table extraction - so they only ever handle
# "application/pdf" for the salary slip and the xlsx mime types for the bank
# statement. Gemini reads documents visually, so it also accepts a
# photographed/scanned salary slip (JPEG) or a PDF bank statement. Detected
# from the upload's filename extension rather than the browser-supplied
# content_type, which isn't always reliable.
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
    method: str
    elapsed_seconds: float
    data: dict | None = None
    error: str | None = None


class BenchmarkResponse(BaseModel):
    regex: ExtractionResult
    gemini: ExtractionResult


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


def _extract_with_regex(
    salary_slip_bytes: bytes,
    salary_mime_type: str,
    bank_statement_bytes: bytes,
    bank_mime_type: str,
) -> dict:
    if salary_mime_type != "application/pdf":
        raise ValueError(
            f"Regex extraction only supports PDF salary slips (pdfplumber has no OCR) - got "
            f"{salary_mime_type}. See the Gemini result for image-based documents."
        )
    if bank_mime_type == "application/pdf":
        raise ValueError(
            "Regex extraction only supports XLSX bank statements (pandas has no PDF table "
            "extraction here) - got application/pdf. See the Gemini result for PDF bank statements."
        )
    salary_fields = parse_salary_slip(io.BytesIO(salary_slip_bytes))
    bank_fields = parse_bank_statement(io.BytesIO(bank_statement_bytes))
    return {
        "Applicant_ID": salary_fields["Applicant_ID"],
        "Age": salary_fields["Age"],
        "Employment_Type": salary_fields["Employment_Type"],
        "Monthly_Net_Income": salary_fields["Monthly_Net_Income"],
        "Total_Existing_EMIs": bank_fields["Total_Existing_EMIs"],
        "Average_Monthly_Bank_Balance": bank_fields["Average_Monthly_Bank_Balance"],
        # The regex parser has no way to recover this - the mock bank statement
        # doesn't render bounced-transaction line items at all (see
        # record_builder.FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS). Left out
        # here rather than fabricated, unlike Gemini which is asked to report 0.
    }


@app.post("/extraction/benchmark", response_model=BenchmarkResponse)
async def benchmark_extraction(
    salary_slip: UploadFile = File(...),
    bank_statement: UploadFile = File(...),
) -> BenchmarkResponse:
    """Runs the regex parsers and the Gemini extraction agent side-by-side
    on the same uploaded documents and returns both results plus timing, so
    the two extraction strategies can be compared directly. A failure in
    either path is captured in that path's `error` field rather than
    failing the whole request - the other path's result is still returned.

    The salary slip may be a PDF or a JPEG (a photographed/scanned slip), and
    the bank statement may be XLSX or a PDF: Gemini reads any of these
    natively, but the regex path is PDF-only for the salary slip and
    XLSX-only for the bank statement (no OCR, no PDF table extraction), and
    reports a clear per-path error rather than crashing when it gets a
    format it can't handle. An unrecognized file extension on either upload
    fails both paths the same way, before either extractor runs.
    """
    salary_bytes = await salary_slip.read()
    bank_bytes = await bank_statement.read()

    try:
        salary_mime_type = _detect_salary_slip_mime_type(salary_slip.filename)
        bank_mime_type = _detect_bank_statement_mime_type(bank_statement.filename)
    except ValueError as exc:
        unsupported = ExtractionResult(method="n/a", elapsed_seconds=0.0, error=str(exc))
        return BenchmarkResponse(
            regex=unsupported.model_copy(update={"method": "regex"}),
            gemini=unsupported.model_copy(update={"method": "gemini"}),
        )

    regex_start = time.perf_counter()
    try:
        regex_data = _extract_with_regex(salary_bytes, salary_mime_type, bank_bytes, bank_mime_type)
        regex_result = ExtractionResult(
            method="regex",
            elapsed_seconds=time.perf_counter() - regex_start,
            data=regex_data,
        )
    except Exception as exc:  # noqa: BLE001 - isolate a parser failure to its own result
        regex_result = ExtractionResult(
            method="regex",
            elapsed_seconds=time.perf_counter() - regex_start,
            error=str(exc),
        )

    gemini_start = time.perf_counter()
    gemini_data = extract_with_gemini_safe(
        salary_bytes, bank_bytes, salary_slip_mime_type=salary_mime_type, bank_statement_mime_type=bank_mime_type
    )
    gemini_error = gemini_data.pop("_error", None)
    gemini_result = ExtractionResult(
        method="gemini",
        elapsed_seconds=time.perf_counter() - gemini_start,
        data=None if gemini_error else gemini_data,
        error=gemini_error,
    )

    return BenchmarkResponse(regex=regex_result, gemini=gemini_result)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
