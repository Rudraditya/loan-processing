"""Turns raw uploaded file bytes into UploadedDocument records."""
from __future__ import annotations

from loan_processing.extraction.extractor import extract_text
from loan_processing.models import DocumentType, UploadedDocument

_TYPE_HINTS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.PAY_STUB: ("paystub", "pay_stub", "pay-stub", "payslip"),
    DocumentType.BANK_STATEMENT: ("bankstatement", "bank_statement", "bank-statement"),
    DocumentType.TAX_FORM: ("w2", "1099", "taxform", "tax_form", "tax-form"),
    DocumentType.APPLICATION_FORM: ("application", "app_form"),
}


def detect_document_type(filename: str) -> DocumentType:
    normalized = filename.lower()
    for doc_type, hints in _TYPE_HINTS.items():
        if any(hint in normalized for hint in hints):
            return doc_type
    return DocumentType.UNKNOWN


def load_document(filename: str, content: bytes) -> UploadedDocument:
    doc_type = detect_document_type(filename)
    raw_text = extract_text(filename, content)
    return UploadedDocument(filename=filename, doc_type=doc_type, raw_text=raw_text)
