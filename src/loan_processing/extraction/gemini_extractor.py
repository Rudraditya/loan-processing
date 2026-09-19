"""Multimodal LLM-based extraction agent using Google's Gemini API, via the
current `google-genai` SDK (migrated off the now-EOL'd `google-generativeai`
SDK referenced in the "Reliability" note below).

Fallback extraction strategy: `agentic_api/main.py` tries the regex parsers
first (fast, free, exact when the upload matches this project's own
mock-document format) and only calls this module when regex fails - e.g. a
real-world salary slip/bank statement whose labels don't match the exact
text `salary_slip_pdf.py`/`bank_statement_xlsx.py` write. Gemini reads the
document visually rather than matching literal label strings, so it can (in
principle) handle format variation a regex parser anchored to one exact
layout can't.

Extracts the same fields the regex path recovers - Applicant_ID, Age, and
Employment_Type are deliberately NOT part of this schema, for the same
reason they aren't regex-extracted (see salary_slip_parser.py /
record_builder.py): Applicant_ID is a caller-assigned serial number with no
reliable machine-readable form on a real document, Age is derived from a
caller-supplied date of birth rather than trusted from a printed figure
that goes stale, and Employment_Type is now a loan-application-form field
(`employment_status` in agentic_api/main.py) rather than something read off
a document - a self-reported document label isn't a reliable substitute for
what the applicant actually selects on the form, and document extraction
had no way to offer a third category ("Business Owner") beyond whatever two
values happened to be printed.

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

Cash_Flow_Trend works two ways: this project's own mock bank statements
print an explicit "Monthly Cash Flow Trend" section (6 stated monthly
averages) that Gemini reads directly. A real-world bank statement won't
have that section, but SYSTEM_PROMPT also instructs Gemini to compute the
same 6 monthly day-weighted averages itself from a raw dated-transaction-
with-running-balance table when no explicit summary exists - the same
method bank_statement_xlsx.py/bank_statement_parser.py use internally,
applied per month instead of over the whole period. Verified against a
real 6-month statement with Python-computed day-weighted averages: Gemini's
values matched to the cent. Only returns null if fewer than 6 full calendar
months of dated data exist to compute from.

Reliability: a prior deployment of this module saw a single request hang
for 5+ minutes with no visible error - consistent with the deprecated
`google-generativeai` SDK's internal retry/backoff behavior running
silently before finally succeeding. `_REQUEST_TIMEOUT_SECONDS` below caps
the whole call (including any internal retries) at a bounded worst case, so
a slow/stuck request now fails fast with a clear timeout error - which
extract_with_gemini_safe() turns into a clean per-request failure - rather
than blocking the caller indefinitely.

Tiered model fallback: the primary model (`DEFAULT_MODEL_NAME`, the "Pro"
tier) is tried first; if it fails specifically on a quota/rate-limit/token-
limit error (HTTP 429, status RESOURCE_EXHAUSTED, or a token-limit message),
`extract_with_gemini()` logs a warning and immediately retries once against
`FALLBACK_MODEL_NAME` (the "Flash" tier), passing the exact same prompt,
schema, and document payload. Other API errors (auth failures, malformed
requests) are NOT retried against the fallback model or a secondary key -
swapping models/keys wouldn't fix those, so retrying would just cost a
second call for no benefit. The returned dict always carries `model_used`
and `fallback_triggered` so callers (and the frontend) can tell which tier
actually served the request.

Secondary API key fallback: if `GEMINI_API_KEY_SECONDARY` is also set (in
addition to the required `GEMINI_API_KEY`), a quota/rate-limit error that
exhausts *both* model tiers on the primary key triggers the same Pro->Flash
sequence again on the secondary key before finally raising. This is purely
a last-resort escape hatch for the free-tier daily quota (20 requests/day
as of writing) - most deployments only need the primary key, and the
secondary is never consulted unless the primary is fully exhausted on both
tiers.
"""
from __future__ import annotations

import io
import json
import os

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, ValidationError

load_dotenv()

DEFAULT_MODEL_NAME = "gemini-pro-latest"  # the "Pro" tier - the user's originally-specified gemini-1.5-pro
# is fully retired (a hard 404, confirmed via ListModels); "-latest" always resolves to whatever's
# currently served instead.
FALLBACK_MODEL_NAME = "gemini-flash-latest"  # the "Flash" tier - gemini-1.5-flash is likewise a hard 404;
# used automatically when the Pro tier hits a quota/rate-limit error, see extract_with_gemini() below.
_API_KEY_ENV_VAR = "GEMINI_API_KEY"
_SECONDARY_API_KEY_ENV_VAR = "GEMINI_API_KEY_SECONDARY"

# Generous enough for normal multimodal processing (observed successful
# calls run 5-15s), tight enough that a stuck/retrying request can't hang
# the caller for minutes - see module docstring.
_REQUEST_TIMEOUT_SECONDS = 45

SYSTEM_PROMPT = """You are a financial data extraction agent for a loan underwriting \
system. You will be given a salary slip (as a PDF document or image) and a bank \
statement (rendered as a CSV-formatted text table, since spreadsheets can't be \
attached directly, or as a PDF document). Extract ONLY the fields listed below, \
exactly as they appear in the documents - do not estimate, round beyond what's \
shown, or invent a value for anything not explicitly present. If a numeric field \
is genuinely absent from both documents, use null rather than guessing.

Fields to extract:
- Full_Name: the applicant's full name, as shown on the salary slip (e.g. next to \
a label like "Employee Name", "Name", or similar) (string)
- Monthly_Net_Income: the net pay / take-home amount for one month, as shown on \
the salary slip (float, no currency symbol or thousands separators)
- Gross_Income: the gross/total pay before deductions, as shown on the salary slip \
(float)
- Total_Deductions: total tax/other deductions shown on the salary slip (float)
- Total_Existing_EMIs: the recurring monthly EMI debit amount visible in the bank \
statement (float)
- Average_Monthly_Bank_Balance: the average account balance for the statement \
period - use the statement's own stated average if it shows one, otherwise compute \
a day-weighted average across the listed balances (float)
- Number_of_Bounced_Transactions_Last_6M: count of bounced / returned / \
insufficient-funds transactions across the statement period; 0 if none appear \
(integer)
- Cash_Flow_Trend: a list of 6 numbers, oldest month first, representing the \
average account balance for each of the most recent 6 calendar months in the \
statement. If the statement already includes a "Monthly Cash Flow Trend" section \
stating these 6 monthly averages directly, use those exact stated values. \
Otherwise, if the statement instead lists dated transactions with a running \
balance column, compute each month's day-weighted average yourself: for each \
calendar month, weight every balance reading by the number of days it stayed in \
effect (from that transaction's date until the next transaction's date, or the \
end of the month for the last transaction) and average those weighted balances \
over the days in that month - the same method used for Average_Monthly_Bank_Balance \
above, applied separately per month instead of over the whole period. Only return \
null if the statement has fewer than 6 full calendar months of dated transaction/ \
balance data to compute this from.

Respond with a single JSON object matching exactly this schema. Do not include any \
explanation, markdown formatting, or text outside the JSON object."""


class GeminiExtractionSchema(BaseModel):
    Full_Name: str
    Monthly_Net_Income: float
    Gross_Income: float
    Total_Deductions: float
    Total_Existing_EMIs: float
    Average_Monthly_Bank_Balance: float
    Number_of_Bounced_Transactions_Last_6M: int
    Cash_Flow_Trend: list[float] | None = None


class GeminiExtractionError(Exception):
    """Raised when Gemini extraction fails after all handling. Callers that
    want a safe-default fallback instead of a raised error should use
    extract_with_gemini_safe() rather than catching this directly.
    """


def _resolve_api_keys() -> list[str]:
    """Every configured Gemini API key, primary first. GEMINI_API_KEY_SECONDARY
    is optional - see module docstring's "Secondary API key fallback" note.
    """
    keys = [os.getenv(_API_KEY_ENV_VAR), os.getenv(_SECONDARY_API_KEY_ENV_VAR)]
    keys = [key for key in keys if key]
    if not keys:
        raise GeminiExtractionError(
            f"{_API_KEY_ENV_VAR} is not set. Add it to a .env file at the repo root "
            "(see .env.example)."
        )
    return keys


def _client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _bank_statement_to_csv_text(bank_statement_bytes: bytes) -> str:
    """Gemini can't read raw .xlsx binary, so render every sheet as CSV text."""
    sheets = pd.read_excel(io.BytesIO(bank_statement_bytes), sheet_name=None, header=None)
    parts = []
    for sheet_name, df in sheets.items():
        parts.append(f"--- Sheet: {sheet_name} ---")
        parts.append(df.to_csv(index=False, header=False))
    return "\n".join(parts)


def _is_quota_or_rate_limit_error(exc: APIError) -> bool:
    """True for HTTP 429 / RESOURCE_EXHAUSTED / token-limit-shaped failures -
    the class of error a lighter fallback model can plausibly route around.
    Deliberately narrower than "any APIError": an auth failure or malformed
    request would fail identically on the fallback model too, so those are
    not retried (see extract_with_gemini()'s docstring).
    """
    if exc.code == 429:
        return True
    if (exc.status or "").upper() == "RESOURCE_EXHAUSTED":
        return True
    message = str(exc).lower()
    return "quota" in message or "rate limit" in message or "token limit" in message


def _build_contents(
    salary_slip_bytes: bytes,
    bank_statement_bytes: bytes,
    salary_slip_mime_type: str,
    bank_statement_mime_type: str,
) -> list[types.Part | str]:
    if bank_statement_mime_type == "application/pdf":
        bank_statement_part: types.Part | str = types.Part.from_bytes(
            data=bank_statement_bytes, mime_type="application/pdf"
        )
    else:
        bank_statement_text = _bank_statement_to_csv_text(bank_statement_bytes)
        bank_statement_part = "Bank statement (rendered as CSV text):\n" + bank_statement_text

    return [
        types.Part.from_bytes(data=salary_slip_bytes, mime_type=salary_slip_mime_type),
        bank_statement_part,
    ]


def _build_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=GeminiExtractionSchema,
        # Caps the whole call (including any SDK-internal retries) at a
        # bounded worst case - see module docstring on why this matters.
        # HttpOptions.timeout is milliseconds, not seconds.
        http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_SECONDS * 1000),
    )


def extract_with_gemini(
    salary_slip_bytes: bytes,
    bank_statement_bytes: bytes,
    model_name: str = DEFAULT_MODEL_NAME,
    salary_slip_mime_type: str = "application/pdf",
    bank_statement_mime_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
) -> dict:
    """Runs both documents through Gemini and returns a dict matching
    GeminiExtractionSchema, plus two sibling metadata keys: `model_used`
    (whichever model actually served the request) and `fallback_triggered`
    (True if the Pro tier hit a quota/rate-limit error and Flash had to
    cover for it).

    `salary_slip_mime_type` defaults to PDF but accepts any mime type Gemini's
    multimodal input understands (e.g. "image/jpeg") - unlike the regex
    parser, which is PDF-text-only, Gemini reads the document visually, so a
    photographed/scanned salary slip works the same way a PDF does.

    `bank_statement_mime_type` defaults to xlsx, which still needs the
    CSV-text conversion below (Gemini can't read raw .xlsx binary). A PDF
    bank statement skips that conversion entirely and is attached as binary,
    same as the salary slip - Gemini reads PDFs natively.

    If `model_name` (the Pro tier) hits a quota/rate-limit error, this
    automatically retries once against FALLBACK_MODEL_NAME (the Flash tier)
    on the same API key, passing the exact same prompt/schema/document
    payload. If GEMINI_API_KEY_SECONDARY is also configured and both tiers
    are exhausted on the primary key, the same Pro->Flash sequence is
    retried once more on the secondary key before giving up. Raises
    GeminiExtractionError if every configured key/tier combination fails, or
    on any other failure: no API key configured at all, a non-quota API
    error, a request that exceeded _REQUEST_TIMEOUT_SECONDS, or output that
    didn't validate against the schema despite the structured-output
    constraint (defense in depth - schema enforcement reduces but doesn't
    eliminate malformed output).
    """
    api_keys = _resolve_api_keys()
    contents = _build_contents(
        salary_slip_bytes, bank_statement_bytes, salary_slip_mime_type, bank_statement_mime_type
    )
    config = _build_config()

    # Ordered attempt sequence: Pro then Flash on the primary key, then (only
    # if configured and only ever reached once every tier on the primary key
    # is exhausted) Pro then Flash again on the secondary key. A key is a
    # last-resort fallback for the one before it, never a load-balancing
    # split.
    attempts: list[tuple[str, str]] = []
    for api_key in api_keys:
        attempts.append((api_key, model_name))
        if model_name != FALLBACK_MODEL_NAME:
            attempts.append((api_key, FALLBACK_MODEL_NAME))

    response = None
    used_model = model_name
    fallback_triggered = False

    for attempt_index, (api_key, tier_model) in enumerate(attempts):
        client = _client(api_key)
        try:
            response = client.models.generate_content(model=tier_model, contents=contents, config=config)
            used_model = tier_model
            fallback_triggered = attempt_index > 0
            break
        except APIError as exc:
            if not _is_quota_or_rate_limit_error(exc):
                # Covers auth failures and DeadlineExceeded-equivalent
                # timeouts (our own timeout above firing) - not retried on
                # another model/key, since neither would be fixed by either.
                raise GeminiExtractionError(f"Gemini API call failed: {exc}") from exc
            if attempt_index == len(attempts) - 1:
                if len(api_keys) == 1:
                    raise GeminiExtractionError(
                        f"Gemini API call failed on both {model_name} and the "
                        f"{FALLBACK_MODEL_NAME} fallback: {exc}"
                    ) from exc
                raise GeminiExtractionError(
                    f"Gemini API call failed on all {len(attempts)} attempt(s) across "
                    f"{len(api_keys)} configured API keys (models and secondary key "
                    f"fallback all exhausted): {exc}"
                ) from exc
            next_key, next_tier = attempts[attempt_index + 1]
            next_key_label = "the same API key" if next_key == api_key else "the secondary API key"
            key_number = api_keys.index(api_key) + 1
            print(
                f"[WARNING] Gemini {tier_model} quota exhausted ({exc.code}) on API key "
                f"#{key_number}. Initiating fallback to {next_tier} ({next_key_label})."
            )
        except Exception as exc:  # noqa: BLE001 - last-resort guard around a third-party SDK call
            raise GeminiExtractionError(f"Unexpected error calling Gemini: {exc}") from exc

    try:
        payload = json.loads(response.text)
        validated = GeminiExtractionSchema.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise GeminiExtractionError(
            f"Gemini returned output that didn't match the expected schema: {exc}"
        ) from exc

    result = validated.model_dump()
    result["model_used"] = used_model
    result["fallback_triggered"] = fallback_triggered
    return result


def extract_with_gemini_safe(
    salary_slip_bytes: bytes,
    bank_statement_bytes: bytes,
    salary_slip_mime_type: str = "application/pdf",
    bank_statement_mime_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
) -> dict:
    """Same as extract_with_gemini, but never raises: returns a null-filled
    dict plus an "_error" key on failure instead, for callers (like the
    scoring/extraction-preview endpoints) that need to always get a response
    back rather than losing the whole request when Gemini fails too.
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
        fallback["model_used"] = None
        fallback["fallback_triggered"] = False
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
