"""Validates extracted fields: required-field presence and cross-document checks."""
from __future__ import annotations

from loan_processing.models import DocumentType, ExtractedFields, ValidationIssue, ValidationResult

_REQUIRED_FIELDS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.PAY_STUB: ("gross_pay", "net_pay"),
    DocumentType.BANK_STATEMENT: ("closing_balance",),
    DocumentType.TAX_FORM: ("annual_income",),
    DocumentType.APPLICATION_FORM: ("applicant_name", "requested_amount"),
}


def validate_extracted_data(extracted: list[ExtractedFields]) -> ValidationResult:
    issues: list[ValidationIssue] = []

    present_types = {doc.doc_type for doc in extracted}
    if DocumentType.APPLICATION_FORM not in present_types:
        issues.append(
            ValidationIssue(
                field="application_form",
                message="No application form was uploaded.",
                severity="error",
            )
        )

    for doc in extracted:
        for field_name in _REQUIRED_FIELDS.get(doc.doc_type, ()):
            if not doc.fields.get(field_name):
                issues.append(
                    ValidationIssue(
                        field=field_name,
                        message=f"Missing '{field_name}' in {doc.source_filename} ({doc.doc_type.value}).",
                        severity="error",
                    )
                )

    if DocumentType.PAY_STUB not in present_types and DocumentType.TAX_FORM not in present_types:
        issues.append(
            ValidationIssue(
                field="income_proof",
                message="No pay stub or tax form provided to verify income.",
                severity="warning",
            )
        )

    is_valid = not any(issue.severity == "error" for issue in issues)
    return ValidationResult(is_valid=is_valid, issues=issues)
