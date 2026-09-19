import io
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from google.genai.errors import APIError

from loan_processing.extraction.gemini_extractor import (
    DEFAULT_MODEL_NAME,
    FALLBACK_MODEL_NAME,
    GeminiExtractionError,
    _bank_statement_to_csv_text,
    extract_with_gemini,
    extract_with_gemini_safe,
)

VALID_PAYLOAD = {
    "Full_Name": "Jane Doe",
    "Monthly_Net_Income": 75000.0,
    "Gross_Income": 96153.85,
    "Total_Deductions": 21153.85,
    "Total_Existing_EMIs": 12000.0,
    "Average_Monthly_Bank_Balance": 150000.0,
    "Number_of_Bounced_Transactions_Last_6M": 0,
    "Cash_Flow_Trend": [140000.0, 142000.0, 145000.0, 148000.0, 150000.0, 155000.0],
}

# extract_with_gemini() adds these two sibling metadata keys to whatever
# GeminiExtractionSchema validates - expected shape when the primary (Pro)
# model serves the request directly, no fallback needed.
NO_FALLBACK_METADATA = {"model_used": DEFAULT_MODEL_NAME, "fallback_triggered": False}


@pytest.fixture(autouse=True)
def _isolate_secondary_key(monkeypatch):
    """GEMINI_API_KEY_SECONDARY is optional and, unlike GEMINI_API_KEY, no
    test sets it by default - without this, a real secondary key present in
    a developer's local .env (loaded into os.environ by load_dotenv() at
    import time) would silently leak into every test here, turning
    single-key-assumption tests into real API calls. Tests that exercise the
    secondary-key path set it explicitly via monkeypatch.setenv.
    """
    monkeypatch.delenv("GEMINI_API_KEY_SECONDARY", raising=False)


def _quota_error(code: int = 429, status: str = "RESOURCE_EXHAUSTED") -> APIError:
    return APIError(code, {"error": {"message": "quota exceeded", "code": code, "status": status}}, response=None)


@pytest.fixture
def bank_statement_bytes() -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_excel(buffer, sheet_name="Statement", index=False)
    return buffer.getvalue()


def _mock_client_with_response(mock_client_cls, response):
    mock_client_cls.return_value.models.generate_content.return_value = response
    return mock_client_cls.return_value.models.generate_content


def test_extract_with_gemini_returns_validated_dict(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        _mock_client_with_response(mock_client_cls, mock_response)
        result = extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

    mock_client_cls.assert_called_once_with(api_key="test-key")
    assert result == {**VALID_PAYLOAD, **NO_FALLBACK_METADATA}


def test_extract_with_gemini_passes_request_timeout(monkeypatch, bank_statement_bytes):
    """A prior deployment saw a single request hang for 5+ minutes with no
    visible error - http_options.timeout caps the whole call (including any
    SDK-internal retries) so a stuck request fails fast instead. This
    asserts the timeout is actually wired through to the SDK call, not just
    documented.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = _mock_client_with_response(mock_client_cls, mock_response)
        extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

        config = generate_content.call_args.kwargs["config"]
    assert config.http_options is not None
    assert config.http_options.timeout > 0


def test_extract_with_gemini_defaults_to_pdf_mime_type(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = _mock_client_with_response(mock_client_cls, mock_response)
        extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

        contents = generate_content.call_args.kwargs["contents"]
    assert contents[0].inline_data.mime_type == "application/pdf"
    assert contents[0].inline_data.data == b"%PDF-fake"


def test_extract_with_gemini_forwards_image_mime_type(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = _mock_client_with_response(mock_client_cls, mock_response)
        result = extract_with_gemini(
            b"\xff\xd8\xff-fake-jpeg", bank_statement_bytes, salary_slip_mime_type="image/jpeg"
        )

        contents = generate_content.call_args.kwargs["contents"]
    assert contents[0].inline_data.mime_type == "image/jpeg"
    assert contents[0].inline_data.data == b"\xff\xd8\xff-fake-jpeg"
    assert result == {**VALID_PAYLOAD, **NO_FALLBACK_METADATA}


def test_extract_with_gemini_raises_without_api_key(monkeypatch, bank_statement_bytes):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiExtractionError, match="GEMINI_API_KEY"):
        extract_with_gemini(b"%PDF-fake", bank_statement_bytes)


def test_extract_with_gemini_raises_on_malformed_json(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text="not valid json")

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        _mock_client_with_response(mock_client_cls, mock_response)

        with pytest.raises(GeminiExtractionError, match="didn't match the expected schema"):
            extract_with_gemini(b"%PDF-fake", bank_statement_bytes)


def test_extract_with_gemini_raises_on_schema_violation(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["Monthly_Net_Income"]  # required field missing - not a JSON syntax error
    mock_response = MagicMock(text=json.dumps(bad_payload))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        _mock_client_with_response(mock_client_cls, mock_response)

        with pytest.raises(GeminiExtractionError):
            extract_with_gemini(b"%PDF-fake", bank_statement_bytes)


def test_extract_with_gemini_raises_on_timeout(monkeypatch, bank_statement_bytes):
    """An APIError from the SDK (our http_options timeout firing) must be
    caught and turned into a GeminiExtractionError, not left to propagate as
    a raw SDK exception.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    timeout_error = APIError(
        408, {"error": {"message": "timed out", "code": 408, "status": "DEADLINE_EXCEEDED"}}, response=None
    )

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = timeout_error

        with pytest.raises(GeminiExtractionError, match="Gemini API call failed"):
            extract_with_gemini(b"%PDF-fake", bank_statement_bytes)


def test_extract_with_gemini_safe_never_raises_and_reports_error(monkeypatch, bank_statement_bytes):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = extract_with_gemini_safe(b"%PDF-fake", bank_statement_bytes)

    assert result["_error"] is not None
    assert result["Full_Name"] is None


def test_extract_with_gemini_safe_forwards_salary_slip_mime_type(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = _mock_client_with_response(mock_client_cls, mock_response)
        extract_with_gemini_safe(b"jpeg-bytes", bank_statement_bytes, salary_slip_mime_type="image/jpeg")

        contents = generate_content.call_args.kwargs["contents"]
    assert contents[0].inline_data.mime_type == "image/jpeg"


def test_bank_statement_to_csv_text_renders_all_sheets(bank_statement_bytes):
    text = _bank_statement_to_csv_text(bank_statement_bytes)

    assert "Sheet: Statement" in text


def test_extract_with_gemini_defaults_bank_statement_to_csv_conversion(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = _mock_client_with_response(mock_client_cls, mock_response)
        extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

        contents = generate_content.call_args.kwargs["contents"]
    # Second content part is the CSV-rendered text, not a binary attachment.
    assert isinstance(contents[1], str)
    assert "Sheet: Statement" in contents[1]


def test_extract_with_gemini_attaches_pdf_bank_statement_as_binary(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = _mock_client_with_response(mock_client_cls, mock_response)
        result = extract_with_gemini(
            b"%PDF-salary-fake",
            b"%PDF-bank-statement-fake",
            bank_statement_mime_type="application/pdf",
        )

        contents = generate_content.call_args.kwargs["contents"]
    assert contents[1].inline_data.mime_type == "application/pdf"
    assert contents[1].inline_data.data == b"%PDF-bank-statement-fake"
    assert result == {**VALID_PAYLOAD, **NO_FALLBACK_METADATA}


def test_extract_with_gemini_falls_back_to_flash_on_quota_error(monkeypatch, bank_statement_bytes, capsys):
    """The Pro tier hitting a 429/RESOURCE_EXHAUSTED error must trigger an
    automatic, same-request retry against the Flash tier - not surface as a
    failure - with a console warning logged and model_used/fallback_triggered
    reflecting what actually happened.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = _mock_client_with_response(mock_client_cls, mock_response)
        generate_content.side_effect = [_quota_error(), mock_response]

        result = extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

        assert generate_content.call_count == 2
        first_call, second_call = generate_content.call_args_list
        assert first_call.kwargs["model"] == DEFAULT_MODEL_NAME
        assert second_call.kwargs["model"] == FALLBACK_MODEL_NAME
        # Same prompt/schema/document payload on both attempts.
        assert first_call.kwargs["contents"] == second_call.kwargs["contents"]
        assert first_call.kwargs["config"] == second_call.kwargs["config"]

    assert result == {**VALID_PAYLOAD, "model_used": FALLBACK_MODEL_NAME, "fallback_triggered": True}

    captured = capsys.readouterr()
    assert "[WARNING]" in captured.out
    assert "quota exhausted" in captured.out
    assert FALLBACK_MODEL_NAME in captured.out


def test_extract_with_gemini_raises_when_fallback_also_fails(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = [_quota_error(), _quota_error()]

        with pytest.raises(GeminiExtractionError, match="failed on both"):
            extract_with_gemini(b"%PDF-fake", bank_statement_bytes)


def test_extract_with_gemini_does_not_fall_back_on_non_quota_error(monkeypatch, bank_statement_bytes):
    """A non-quota API error (e.g. an auth failure) shouldn't trigger a
    second call against the fallback model - a different model wouldn't fix
    an auth problem, so retrying would just waste a call.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    auth_error = APIError(401, {"error": {"message": "invalid API key", "code": 401, "status": "UNAUTHENTICATED"}})

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = mock_client_cls.return_value.models.generate_content
        generate_content.side_effect = auth_error

        with pytest.raises(GeminiExtractionError, match="Gemini API call failed"):
            extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

    assert generate_content.call_count == 1


def test_extract_with_gemini_falls_back_to_secondary_key_when_primary_fully_exhausted(
    monkeypatch, bank_statement_bytes, capsys
):
    """Both tiers (Pro, then Flash) exhausting their quota on the primary key
    must trigger the same Pro->Flash sequence on the secondary key, not an
    immediate failure - this is the whole point of configuring a secondary
    key at all.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setenv("GEMINI_API_KEY_SECONDARY", "secondary-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = mock_client_cls.return_value.models.generate_content
        generate_content.side_effect = [_quota_error(), _quota_error(), mock_response]

        result = extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

        assert generate_content.call_count == 3
        assert mock_client_cls.call_args_list == [
            ((), {"api_key": "primary-key"}),
            ((), {"api_key": "primary-key"}),
            ((), {"api_key": "secondary-key"}),
        ]
        first_call, second_call, third_call = generate_content.call_args_list
        assert first_call.kwargs["model"] == DEFAULT_MODEL_NAME
        assert second_call.kwargs["model"] == FALLBACK_MODEL_NAME
        assert third_call.kwargs["model"] == DEFAULT_MODEL_NAME

    assert result == {**VALID_PAYLOAD, "model_used": DEFAULT_MODEL_NAME, "fallback_triggered": True}

    captured = capsys.readouterr()
    assert captured.out.count("[WARNING]") == 2
    assert "secondary API key" in captured.out


def test_extract_with_gemini_raises_when_secondary_key_also_fully_exhausted(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setenv("GEMINI_API_KEY_SECONDARY", "secondary-key")

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = mock_client_cls.return_value.models.generate_content
        generate_content.side_effect = [_quota_error(), _quota_error(), _quota_error(), _quota_error()]

        with pytest.raises(GeminiExtractionError, match="all 4 attempt"):
            extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

        assert generate_content.call_count == 4


def test_extract_with_gemini_does_not_try_secondary_key_on_non_quota_error(monkeypatch, bank_statement_bytes):
    """A non-quota error (auth failure) on the primary key shouldn't trigger
    trying the secondary key either - same rationale as not falling back to
    a different model tier for this error class.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setenv("GEMINI_API_KEY_SECONDARY", "secondary-key")
    auth_error = APIError(401, {"error": {"message": "invalid API key", "code": 401, "status": "UNAUTHENTICATED"}})

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        generate_content = mock_client_cls.return_value.models.generate_content
        generate_content.side_effect = auth_error

        with pytest.raises(GeminiExtractionError, match="Gemini API call failed"):
            extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

    assert generate_content.call_count == 1


def test_extract_with_gemini_safe_reports_fallback_metadata(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with patch("loan_processing.extraction.gemini_extractor.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = [_quota_error(), mock_response]
        result = extract_with_gemini_safe(b"%PDF-fake", bank_statement_bytes)

    assert result["model_used"] == FALLBACK_MODEL_NAME
    assert result["fallback_triggered"] is True


def test_extract_with_gemini_safe_defaults_fallback_metadata_on_total_failure(monkeypatch, bank_statement_bytes):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = extract_with_gemini_safe(b"%PDF-fake", bank_statement_bytes)

    assert result["model_used"] is None
    assert result["fallback_triggered"] is False
