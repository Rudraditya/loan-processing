import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from loan_processing.agentic_api.main import app

client = TestClient(app)

REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_DOCS_DIR = REPO_ROOT / "data" / "mock_documents"


@pytest.fixture
def mock_documents() -> tuple[bytes, bytes]:
    applicant_id = sorted(MOCK_DOCS_DIR.glob("*_salary.pdf"))[0].stem.removesuffix("_salary")
    salary_bytes = (MOCK_DOCS_DIR / f"{applicant_id}_salary.pdf").read_bytes()
    bank_bytes = (MOCK_DOCS_DIR / f"{applicant_id}_bank.xlsx").read_bytes()
    return salary_bytes, bank_bytes


def test_benchmark_runs_regex_and_reports_gemini_failure_without_api_key(mock_documents, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    salary_bytes, bank_bytes = mock_documents

    files = {
        "salary_slip": ("salary.pdf", io.BytesIO(salary_bytes), "application/pdf"),
        "bank_statement": (
            "bank.xlsx",
            io.BytesIO(bank_bytes),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    }

    response = client.post("/extraction/benchmark", files=files)

    assert response.status_code == 200
    body = response.json()

    assert body["regex"]["error"] is None
    assert body["regex"]["data"]["Employment_Type"] in ("Salaried", "Self-Employed")
    assert body["regex"]["elapsed_seconds"] >= 0

    # No GEMINI_API_KEY configured in this test environment - the endpoint should
    # still return 200 with the regex result intact and the failure isolated here.
    assert body["gemini"]["data"] is None
    assert "GEMINI_API_KEY" in body["gemini"]["error"]


def test_benchmark_rejects_image_salary_slip_on_regex_path_only(mock_documents, monkeypatch):
    """A JPEG salary slip can't go through pdfplumber (no OCR) - the regex
    path should report a clear per-path error rather than crash, while the
    request still succeeds overall (Gemini would handle it, but without a
    real API key here it fails for the usual no-key reason, not because of
    the image format).
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _, bank_bytes = mock_documents

    files = {
        "salary_slip": ("salary.jpg", io.BytesIO(b"\xff\xd8\xff-fake-jpeg-bytes"), "image/jpeg"),
        "bank_statement": (
            "bank.xlsx",
            io.BytesIO(bank_bytes),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    }

    response = client.post("/extraction/benchmark", files=files)

    assert response.status_code == 200
    body = response.json()

    assert body["regex"]["data"] is None
    assert "only supports PDF" in body["regex"]["error"]

    assert body["gemini"]["data"] is None
    assert "GEMINI_API_KEY" in body["gemini"]["error"]


def test_benchmark_rejects_unsupported_salary_slip_extension(mock_documents):
    _, bank_bytes = mock_documents

    files = {
        "salary_slip": ("salary.png", io.BytesIO(b"not-a-real-file"), "image/png"),
        "bank_statement": (
            "bank.xlsx",
            io.BytesIO(bank_bytes),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    }

    response = client.post("/extraction/benchmark", files=files)

    assert response.status_code == 200
    body = response.json()

    assert "Unsupported salary slip file type" in body["regex"]["error"]
    assert "Unsupported salary slip file type" in body["gemini"]["error"]


def test_benchmark_rejects_pdf_bank_statement_on_regex_path_only(mock_documents, monkeypatch):
    """A PDF bank statement can't go through pandas.read_excel - the regex
    path should report a clear per-path error rather than crash, mirroring
    the JPEG-salary-slip case.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    salary_bytes, _ = mock_documents

    files = {
        "salary_slip": ("salary.pdf", io.BytesIO(salary_bytes), "application/pdf"),
        "bank_statement": ("bank.pdf", io.BytesIO(b"%PDF-fake-bank-statement"), "application/pdf"),
    }

    response = client.post("/extraction/benchmark", files=files)

    assert response.status_code == 200
    body = response.json()

    assert body["regex"]["data"] is None
    assert "only supports XLSX bank statements" in body["regex"]["error"]

    assert body["gemini"]["data"] is None
    assert "GEMINI_API_KEY" in body["gemini"]["error"]


def test_benchmark_rejects_unsupported_bank_statement_extension(mock_documents):
    salary_bytes, _ = mock_documents

    files = {
        "salary_slip": ("salary.pdf", io.BytesIO(salary_bytes), "application/pdf"),
        "bank_statement": ("bank.docx", io.BytesIO(b"not-a-real-file"), "application/msword"),
    }

    response = client.post("/extraction/benchmark", files=files)

    assert response.status_code == 200
    body = response.json()

    assert "Unsupported bank statement file type" in body["regex"]["error"]
    assert "Unsupported bank statement file type" in body["gemini"]["error"]


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
