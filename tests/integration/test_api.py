import io

import openpyxl
from fastapi.testclient import TestClient

from loan_processing.api.main import app

client = TestClient(app)


def test_process_application_returns_excel_workbook():
    application_form = b"Applicant Name: Jane Doe\nRequested Amount: $20,000\nMonthly Debt: $500"
    tax_form = b"Total Income: $90,000\nTax Year: 2025"

    files = [
        ("files", ("application_form.txt", io.BytesIO(application_form), "text/plain")),
        ("files", ("tax_form.txt", io.BytesIO(tax_form), "text/plain")),
    ]

    response = client.post("/applications/process", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    workbook = openpyxl.load_workbook(io.BytesIO(response.content))
    assert "Summary" in workbook.sheetnames


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
