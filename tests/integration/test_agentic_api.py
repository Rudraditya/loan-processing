import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

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


@pytest.fixture
def mismatched_gross_deductions_salary_slip_pdf() -> bytes:
    """A salary slip using the exact labels salary_slip_parser.py's regex is
    anchored to, but with Gross Pay - Tax Deductions deliberately not equal
    to Net Pay - regex succeeds (the labels match), but the figures don't
    reconcile, which is exactly what compute_discrepancy_flags() should
    catch.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.drawString(50, 800, "Mismatch Test Employer")
    c.drawString(50, 780, "Employee Name: Test Applicant")
    c.drawString(50, 760, "Employment Type: Salaried")
    c.drawString(50, 740, "Gross Pay: Rs. 100,000.00")
    c.drawString(50, 720, "Tax Deductions: Rs. 25,000.00")
    c.drawString(50, 700, "Net Pay: Rs. 60,000.00")  # should be 75,000.00
    c.showPage()
    c.save()
    return buffer.getvalue()


@pytest.fixture
def non_matching_salary_slip_pdf() -> bytes:
    """A well-formed PDF that doesn't use any of the labels
    salary_slip_parser.py's regex is anchored to - simulates a real-world
    salary slip so regex genuinely fails (not just a bad file format),
    forcing the Gemini fallback path to actually be attempted.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.drawString(50, 800, "Totally Different Payslip Co.")
    c.drawString(50, 780, "Employee: Someone Real")
    c.drawString(50, 760, "Take Home Pay: 50,000.00")
    c.showPage()
    c.save()
    return buffer.getvalue()


def test_preview_extraction_runs_regex_successfully(mock_documents):
    salary_bytes, bank_bytes = mock_documents

    files = {
        "salary_slip": ("salary.pdf", io.BytesIO(salary_bytes), "application/pdf"),
        "bank_statement": (
            "bank.xlsx",
            io.BytesIO(bank_bytes),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    }

    response = client.post("/extraction/preview", files=files)

    assert response.status_code == 200
    body = response.json()

    assert body["error"] is None
    assert body["method"] == "regex"
    # Employment_Type is a loan-application-form field now, not extracted -
    # it should never appear in extraction/preview's recovered data.
    assert "Employment_Type" not in body["data"]
    assert body["elapsed_seconds"] >= 0
    assert body["discrepancy_flags"] == []


def test_preview_extraction_flags_mismatched_gross_and_deductions(
    mismatched_gross_deductions_salary_slip_pdf, mock_documents
):
    _, bank_bytes = mock_documents

    files = {
        "salary_slip": ("salary.pdf", io.BytesIO(mismatched_gross_deductions_salary_slip_pdf), "application/pdf"),
        "bank_statement": (
            "bank.xlsx",
            io.BytesIO(bank_bytes),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    }

    response = client.post("/extraction/preview", files=files)

    assert response.status_code == 200
    body = response.json()

    assert body["method"] == "regex"
    assert len(body["discrepancy_flags"]) == 1
    assert "doesn't reconcile" in body["discrepancy_flags"][0]


def test_preview_extraction_falls_back_to_gemini_when_regex_labels_dont_match(
    monkeypatch, non_matching_salary_slip_pdf, mock_documents
):
    """Regex fails on a PDF that doesn't have the exact anchored labels (a
    real-world, non-mock salary slip) - this asserts Gemini actually gets
    invoked as the fallback rather than the request just giving up, using
    the deterministic no-API-key failure path so this doesn't need network
    access or a real key. The combined error naming both failure reasons is
    the proof Gemini was attempted, not just regex.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_SECONDARY", raising=False)
    _, bank_bytes = mock_documents

    files = {
        "salary_slip": ("salary.pdf", io.BytesIO(non_matching_salary_slip_pdf), "application/pdf"),
        "bank_statement": (
            "bank.xlsx",
            io.BytesIO(bank_bytes),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    }

    response = client.post("/extraction/preview", files=files)

    assert response.status_code == 200
    body = response.json()

    assert body["data"] is None
    assert "regex parser:" in body["error"]
    assert "Gemini agent:" in body["error"]
    assert "GEMINI_API_KEY" in body["error"]


def test_preview_extraction_routes_jpeg_salary_slip_straight_to_gemini(monkeypatch, mock_documents):
    """Regex has no OCR, so a JPEG never even reaches the parser - this
    should fall straight through to the Gemini fallback (exercised via the
    deterministic no-key path), not be rejected outright the way an
    unsupported extension is.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_SECONDARY", raising=False)
    _, bank_bytes = mock_documents

    files = {
        "salary_slip": ("salary.jpg", io.BytesIO(b"\xff\xd8\xff-fake-jpeg-bytes"), "image/jpeg"),
        "bank_statement": (
            "bank.xlsx",
            io.BytesIO(bank_bytes),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    }

    response = client.post("/extraction/preview", files=files)

    assert response.status_code == 200
    body = response.json()

    assert body["data"] is None
    assert "regex extraction only supports a PDF salary slip" in body["error"]
    assert "Gemini agent:" in body["error"]
    assert "GEMINI_API_KEY" in body["error"]


def test_preview_extraction_rejects_unsupported_salary_slip_extension(mock_documents):
    _, bank_bytes = mock_documents

    files = {
        "salary_slip": ("salary.png", io.BytesIO(b"not-a-real-file"), "image/png"),
        "bank_statement": (
            "bank.xlsx",
            io.BytesIO(bank_bytes),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    }

    response = client.post("/extraction/preview", files=files)

    assert response.status_code == 200
    body = response.json()

    assert "Unsupported salary slip file type" in body["error"]


def test_preview_extraction_rejects_unsupported_bank_statement_extension(mock_documents):
    salary_bytes, _ = mock_documents

    files = {
        "salary_slip": ("salary.pdf", io.BytesIO(salary_bytes), "application/pdf"),
        "bank_statement": ("bank.docx", io.BytesIO(b"not-a-real-file"), "application/msword"),
    }

    response = client.post("/extraction/preview", files=files)

    assert response.status_code == 200
    body = response.json()

    assert "Unsupported bank statement file type" in body["error"]


def test_scoring_falls_back_to_gemini_when_regex_labels_dont_match(
    monkeypatch, non_matching_salary_slip_pdf, mock_documents
):
    """Same fallback contract as /extraction/preview, but on the endpoint
    that actually feeds the ML model - proves Gemini is attempted here too,
    not just in the preview.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_SECONDARY", raising=False)
    _, bank_bytes = mock_documents

    files = {
        "salary_slip": ("salary.pdf", io.BytesIO(non_matching_salary_slip_pdf), "application/pdf"),
        "bank_statement": (
            "bank.xlsx",
            io.BytesIO(bank_bytes),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    }
    data = {
        "applicant_id": "APP-TEST-001",
        "applicant_name": "Someone Real",
        "date_of_birth": "1990-01-01",
        "employment_status": "Salaried",
        "requested_loan_amount": "500000",
        "requested_tenure_months": "36",
    }

    response = client.post("/scoring/assess", files=files, data=data)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "regex parser:" in detail
    assert "Gemini agent:" in detail
    assert "GEMINI_API_KEY" in detail


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
