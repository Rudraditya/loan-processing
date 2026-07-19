import io
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from loan_processing.extraction.gemini_extractor import (
    GeminiExtractionError,
    _bank_statement_to_csv_text,
    extract_with_gemini,
    extract_with_gemini_safe,
)

VALID_PAYLOAD = {
    "Applicant_ID": "APP00001",
    "Age": 34,
    "Employment_Type": "Salaried",
    "Monthly_Net_Income": 75000.0,
    "Total_Existing_EMIs": 12000.0,
    "Average_Monthly_Bank_Balance": 150000.0,
    "Number_of_Bounced_Transactions_Last_6M": 0,
}


@pytest.fixture
def bank_statement_bytes() -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_excel(buffer, sheet_name="Statement", index=False)
    return buffer.getvalue()


def test_extract_with_gemini_returns_validated_dict(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with (
        patch("loan_processing.extraction.gemini_extractor.genai.configure") as mock_configure,
        patch("loan_processing.extraction.gemini_extractor.genai.GenerativeModel") as mock_model_cls,
    ):
        mock_model_cls.return_value.generate_content.return_value = mock_response
        result = extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

    mock_configure.assert_called_once_with(api_key="test-key")
    assert result == VALID_PAYLOAD


def test_extract_with_gemini_defaults_to_pdf_mime_type(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with (
        patch("loan_processing.extraction.gemini_extractor.genai.configure"),
        patch("loan_processing.extraction.gemini_extractor.genai.GenerativeModel") as mock_model_cls,
    ):
        mock_model_cls.return_value.generate_content.return_value = mock_response
        extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

        contents = mock_model_cls.return_value.generate_content.call_args[0][0]
    assert contents[0] == {"mime_type": "application/pdf", "data": b"%PDF-fake"}


def test_extract_with_gemini_forwards_image_mime_type(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with (
        patch("loan_processing.extraction.gemini_extractor.genai.configure"),
        patch("loan_processing.extraction.gemini_extractor.genai.GenerativeModel") as mock_model_cls,
    ):
        mock_model_cls.return_value.generate_content.return_value = mock_response
        result = extract_with_gemini(
            b"\xff\xd8\xff-fake-jpeg", bank_statement_bytes, salary_slip_mime_type="image/jpeg"
        )

        contents = mock_model_cls.return_value.generate_content.call_args[0][0]
    assert contents[0] == {"mime_type": "image/jpeg", "data": b"\xff\xd8\xff-fake-jpeg"}
    assert result == VALID_PAYLOAD


def test_extract_with_gemini_raises_without_api_key(monkeypatch, bank_statement_bytes):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiExtractionError, match="GEMINI_API_KEY"):
        extract_with_gemini(b"%PDF-fake", bank_statement_bytes)


def test_extract_with_gemini_raises_on_malformed_json(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text="not valid json")

    with (
        patch("loan_processing.extraction.gemini_extractor.genai.configure"),
        patch("loan_processing.extraction.gemini_extractor.genai.GenerativeModel") as mock_model_cls,
    ):
        mock_model_cls.return_value.generate_content.return_value = mock_response

        with pytest.raises(GeminiExtractionError, match="didn't match the expected schema"):
            extract_with_gemini(b"%PDF-fake", bank_statement_bytes)


def test_extract_with_gemini_raises_on_schema_violation(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["Employment_Type"] = "Retired"  # not in the allowed Literal
    mock_response = MagicMock(text=json.dumps(bad_payload))

    with (
        patch("loan_processing.extraction.gemini_extractor.genai.configure"),
        patch("loan_processing.extraction.gemini_extractor.genai.GenerativeModel") as mock_model_cls,
    ):
        mock_model_cls.return_value.generate_content.return_value = mock_response

        with pytest.raises(GeminiExtractionError):
            extract_with_gemini(b"%PDF-fake", bank_statement_bytes)


def test_extract_with_gemini_safe_never_raises_and_reports_error(monkeypatch, bank_statement_bytes):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = extract_with_gemini_safe(b"%PDF-fake", bank_statement_bytes)

    assert result["_error"] is not None
    assert result["Applicant_ID"] is None


def test_extract_with_gemini_safe_forwards_salary_slip_mime_type(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with (
        patch("loan_processing.extraction.gemini_extractor.genai.configure"),
        patch("loan_processing.extraction.gemini_extractor.genai.GenerativeModel") as mock_model_cls,
    ):
        mock_model_cls.return_value.generate_content.return_value = mock_response
        extract_with_gemini_safe(b"jpeg-bytes", bank_statement_bytes, salary_slip_mime_type="image/jpeg")

        contents = mock_model_cls.return_value.generate_content.call_args[0][0]
    assert contents[0]["mime_type"] == "image/jpeg"


def test_bank_statement_to_csv_text_renders_all_sheets(bank_statement_bytes):
    text = _bank_statement_to_csv_text(bank_statement_bytes)

    assert "Sheet: Statement" in text


def test_extract_with_gemini_defaults_bank_statement_to_csv_conversion(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with (
        patch("loan_processing.extraction.gemini_extractor.genai.configure"),
        patch("loan_processing.extraction.gemini_extractor.genai.GenerativeModel") as mock_model_cls,
    ):
        mock_model_cls.return_value.generate_content.return_value = mock_response
        extract_with_gemini(b"%PDF-fake", bank_statement_bytes)

        contents = mock_model_cls.return_value.generate_content.call_args[0][0]
    # Second content part is the CSV-rendered text, not a binary attachment.
    assert isinstance(contents[1], str)
    assert "Sheet: Statement" in contents[1]


def test_extract_with_gemini_attaches_pdf_bank_statement_as_binary(monkeypatch, bank_statement_bytes):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_response = MagicMock(text=json.dumps(VALID_PAYLOAD))

    with (
        patch("loan_processing.extraction.gemini_extractor.genai.configure"),
        patch("loan_processing.extraction.gemini_extractor.genai.GenerativeModel") as mock_model_cls,
    ):
        mock_model_cls.return_value.generate_content.return_value = mock_response
        result = extract_with_gemini(
            b"%PDF-salary-fake",
            b"%PDF-bank-statement-fake",
            bank_statement_mime_type="application/pdf",
        )

        contents = mock_model_cls.return_value.generate_content.call_args[0][0]
    assert contents[1] == {"mime_type": "application/pdf", "data": b"%PDF-bank-statement-fake"}
    assert result == VALID_PAYLOAD
