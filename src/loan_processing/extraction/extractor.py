"""Text extraction (OCR/parsing) and field parsing from document text.

Text extraction dispatches on file extension to the right backend (pdfplumber
for PDFs, pytesseract for images). Field parsing is pure text-in/dict-out so
it can be unit tested without exercising OCR or needing real documents.
"""
from __future__ import annotations

import io
import re

from loan_processing.models import DocumentType, ExtractedFields

_PDF_EXTENSIONS = (".pdf",)
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")


def extract_text(filename: str, content: bytes) -> str:
    lowered = filename.lower()
    if lowered.endswith(_PDF_EXTENSIONS):
        return _extract_pdf_text(content)
    if lowered.endswith(_IMAGE_EXTENSIONS):
        return _extract_image_text(content)
    return content.decode("utf-8", errors="ignore")


def _extract_pdf_text(content: bytes) -> str:
    import pdfplumber

    pages_text: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def _extract_image_text(content: bytes) -> str:
    import pytesseract
    from PIL import Image

    image = Image.open(io.BytesIO(content))
    return pytesseract.image_to_string(image)


# Field patterns are intentionally simple heuristics over free text. They are
# a placeholder for a real OCR/NLP extraction service and are expected to be
# swapped out per document-type as real templates are onboarded.
_FIELD_PATTERNS: dict[DocumentType, dict[str, str]] = {
    DocumentType.PAY_STUB: {
        "employer_name": r"Employer[:\s]+(.+)",
        "gross_pay": r"Gross Pay[:\s]+\$?([\d,]+\.?\d*)",
        "net_pay": r"Net Pay[:\s]+\$?([\d,]+\.?\d*)",
        "pay_period": r"Pay Period[:\s]+(.+)",
    },
    DocumentType.BANK_STATEMENT: {
        "account_holder": r"Account Holder[:\s]+(.+)",
        "closing_balance": r"Closing Balance[:\s]+\$?([\d,]+\.?\d*)",
        "average_balance": r"Average Balance[:\s]+\$?([\d,]+\.?\d*)",
    },
    DocumentType.TAX_FORM: {
        "annual_income": r"(?:Total|Annual) Income[:\s]+\$?([\d,]+\.?\d*)",
        "tax_year": r"Tax Year[:\s]+(\d{4})",
    },
    DocumentType.APPLICATION_FORM: {
        "applicant_name": r"Applicant Name[:\s]+(.+)",
        "requested_amount": r"Requested Amount[:\s]+\$?([\d,]+\.?\d*)",
        "employment_status": r"Employment Status[:\s]+(.+)",
        "monthly_debt": r"Monthly Debt[:\s]+\$?([\d,]+\.?\d*)",
    },
}


def extract_fields(doc_type: DocumentType, text: str, source_filename: str) -> ExtractedFields:
    patterns = _FIELD_PATTERNS.get(doc_type, {})
    fields: dict[str, str] = {}
    for field_name, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fields[field_name] = match.group(1).strip()
    return ExtractedFields(doc_type=doc_type, source_filename=source_filename, fields=fields)
