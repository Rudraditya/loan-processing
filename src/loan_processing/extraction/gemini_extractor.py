"""Multimodal LLM-based extraction agent using Google's Gemini API.

Extracts the same document-derivable fields as salary_slip_parser.py /
bank_statement_parser.py, but via a vision-language model instead of regex,
so it can (in principle) handle format variation across real-world bank
statements and salary slips that a regex parser anchored to one exact
layout can't. This is kept side-by-side with the regex parsers, not a
replacement - see agentic_api/main.py for the benchmarking endpoint that
runs both and compares them.

Gemini's inline multimodal input reads PDFs and images (e.g. a photographed
or scanned salary slip, mime_type "image/jpeg") natively - it understands
the visual layout, not just embedded text, so it isn't limited to documents
that have a real text layer the way the regex parser is. A PDF bank
statement gets the same binary-attachment treatment. It does not parse raw
.xlsx binary, though - a spreadsheet isn't a document type these models are
trained to read directly - so an xlsx bank statement is rendered to a
plain-text CSV table via pandas and passed as prompt text instead.

Number_of_Bounced_Transactions_Last_6M is included in the requested schema,
but the mock bank statements this project generates don't actually contain
any bounced-transaction line items (see record_builder.py's
FIELDS_NOT_AVAILABLE_FROM_THESE_DOCUMENTS) - a correct extraction of these
particular documents should report 0, not a fabricated nonzero count.
"""
from __future__ import annotations

import io
import json
import os
from typing import Literal

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPICallError
from pydantic import BaseModel, ValidationError

load_dotenv()

DEFAULT_MODEL_NAME = "gemini-flash-latest"  # gemini-1.5-flash was retired; gemini-2.5-flash is blocked for new-user
# accounts even though ListModels lists it - the "-latest" alias always points at whatever's currently available.
_API_KEY_ENV_VAR = "GEMINI_API_KEY"

SYSTEM_PROMPT = """You are a financial data extraction agent for a loan underwriting \
system. You will be given a salary slip (as a PDF document) and a bank statement \
(rendered as a CSV-formatted text table, since spreadsheets can't be attached \
directly). Extract ONLY the fields listed below, exactly as they appear in the \
documents - do not estimate, round beyond what's shown, or invent a value for \
anything not explicitly present. If a numeric field is genuinely absent from both \
documents, use null rather than guessing.

Fields to extract:
- Applicant_ID: the applicant identifier shown on the documents (string)
- Age: the applicant's age in years (integer)
- Employment_Type: exactly "Salaried" or "Self-Employed", based on the salary \
slip's own labeling (string)
- Monthly_Net_Income: the net pay / take-home amount for one month, as shown on \
the salary slip (float, no currency symbol or thousands separators)
- Total_Existing_EMIs: the recurring monthly EMI debit amount visible in the bank \
statement (float)
- Average_Monthly_Bank_Balance: the average account balance for the statement \
period - use the statement's own stated average if it shows one, otherwise compute \
a day-weighted average across the listed balances (float)
- Number_of_Bounced_Transactions_Last_6M: count of bounced / returned / \
insufficient-funds transactions across the statement period; 0 if none appear \
(integer)

Respond with a single JSON object matching exactly this schema. Do not include any \
explanation, markdown formatting, or text outside the JSON object."""


class GeminiExtractionSchema(BaseModel):
    Applicant_ID: str
    Age: int
    Employment_Type: Literal["Salaried", "Self-Employed"]
    Monthly_Net_Income: float
    Total_Existing_EMIs: float
    Average_Monthly_Bank_Balance: float
    Number_of_Bounced_Transactions_Last_6M: int


class GeminiExtractionError(Exception):
    """Raised when Gemini extraction fails after all handling. Callers that
    want a safe-default fallback instead of a raised error should use
    extract_with_gemini_safe() rather than catching this directly.
    """


def _configure() -> None:
    api_key = os.getenv(_API_KEY_ENV_VAR)
    if not api_key:
        raise GeminiExtractionError(
            f"{_API_KEY_ENV_VAR} is not set. Add it to a .env file at the repo root "
            "(see .env.example)."
        )
    genai.configure(api_key=api_key)


def _bank_statement_to_csv_text(bank_statement_bytes: bytes) -> str:
    """Gemini can't read raw .xlsx binary, so render every sheet as CSV text."""
    sheets = pd.read_excel(io.BytesIO(bank_statement_bytes), sheet_name=None, header=None)
    parts = []
    for sheet_name, df in sheets.items():
        parts.append(f"--- Sheet: {sheet_name} ---")
        parts.append(df.to_csv(index=False, header=False))
    return "\n".join(parts)


def extract_with_gemini(
    salary_slip_bytes: bytes,
    bank_statement_bytes: bytes,
    model_name: str = DEFAULT_MODEL_NAME,
    salary_slip_mime_type: str = "application/pdf",
    bank_statement_mime_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
) -> dict:
    """Runs both documents through Gemini and returns a dict matching
    GeminiExtractionSchema.

    `salary_slip_mime_type` defaults to PDF but accepts any mime type Gemini's
    multimodal input understands (e.g. "image/jpeg") - unlike the regex
    parser, which is PDF-text-only, Gemini reads the document visually, so a
    photographed/scanned salary slip works the same way a PDF does.

    `bank_statement_mime_type` defaults to xlsx, which still needs the
    CSV-text conversion below (Gemini can't read raw .xlsx binary). A PDF
    bank statement skips that conversion entirely and is attached as binary,
    same as the salary slip - Gemini reads PDFs natively.

    Raises GeminiExtractionError on any failure: missing API key, a rate
    limit / API error, or output that didn't validate against the schema
    despite the structured-output constraint (defense in depth - schema
    enforcement reduces but doesn't eliminate malformed output).
    """
    _configure()

    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)
    if bank_statement_mime_type == "application/pdf":
        bank_statement_part: dict | str = {"mime_type": "application/pdf", "data": bank_statement_bytes}
    else:
        bank_statement_text = _bank_statement_to_csv_text(bank_statement_bytes)
        bank_statement_part = "Bank statement (rendered as CSV text):\n" + bank_statement_text

    try:
        response = model.generate_content(
            [
                {"mime_type": salary_slip_mime_type, "data": salary_slip_bytes},
                bank_statement_part,
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=GeminiExtractionSchema,
            ),
        )
    except GoogleAPICallError as exc:
        # Covers rate limits (ResourceExhausted), auth failures, timeouts, etc.
        raise GeminiExtractionError(f"Gemini API call failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - last-resort guard around a third-party SDK call
        raise GeminiExtractionError(f"Unexpected error calling Gemini: {exc}") from exc

    try:
        payload = json.loads(response.text)
        validated = GeminiExtractionSchema.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise GeminiExtractionError(
            f"Gemini returned output that didn't match the expected schema: {exc}"
        ) from exc

    return validated.model_dump()


def extract_with_gemini_safe(
    salary_slip_bytes: bytes,
    bank_statement_bytes: bytes,
    salary_slip_mime_type: str = "application/pdf",
    bank_statement_mime_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
) -> dict:
    """Same as extract_with_gemini, but never raises: returns a null-filled
    dict plus an "_error" key on failure instead, for callers (like the
    benchmarking endpoint) that need to always get a response back rather
    than losing the whole request when only the Gemini side fails.
    """
    try:
        return extract_with_gemini(
            salary_slip_bytes,
            bank_statement_bytes,
            salary_slip_mime_type=salary_slip_mime_type,
            bank_statement_mime_type=bank_statement_mime_type,
        )
    except GeminiExtractionError as exc:
        fallback = {field: None for field in GeminiExtractionSchema.model_fields}
        fallback["_error"] = str(exc)
        return fallback


if __name__ == "__main__":
    import sys
    from pathlib import Path

    mock_docs_dir = Path(__file__).resolve().parents[3] / "data" / "mock_documents"
    applicant_id = sys.argv[1] if len(sys.argv) > 1 else sorted(mock_docs_dir.glob("*_salary.pdf"))[0].stem.removesuffix("_salary")

    salary_path = mock_docs_dir / f"{applicant_id}_salary.pdf"
    bank_path = mock_docs_dir / f"{applicant_id}_bank.xlsx"
    print(f"Extracting {applicant_id} via Gemini ({DEFAULT_MODEL_NAME})...\n")

    result = extract_with_gemini_safe(salary_path.read_bytes(), bank_path.read_bytes())
    print(json.dumps(result, indent=2))
