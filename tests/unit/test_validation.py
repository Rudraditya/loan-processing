from loan_processing.models import DocumentType, ExtractedFields
from loan_processing.validation.validator import validate_extracted_data


def test_complete_application_is_valid():
    extracted = [
        ExtractedFields(
            doc_type=DocumentType.APPLICATION_FORM,
            source_filename="application.txt",
            fields={"applicant_name": "Jane Doe", "requested_amount": "20000"},
        ),
        ExtractedFields(
            doc_type=DocumentType.PAY_STUB,
            source_filename="paystub.txt",
            fields={"gross_pay": "5000", "net_pay": "3900"},
        ),
    ]

    result = validate_extracted_data(extracted)

    assert result.is_valid
    assert not any(issue.severity == "error" for issue in result.issues)


def test_missing_application_form_is_an_error():
    extracted = [
        ExtractedFields(
            doc_type=DocumentType.PAY_STUB,
            source_filename="paystub.txt",
            fields={"gross_pay": "5000", "net_pay": "3900"},
        ),
    ]

    result = validate_extracted_data(extracted)

    assert not result.is_valid
    assert any(issue.field == "application_form" for issue in result.issues)


def test_missing_required_field_is_an_error():
    extracted = [
        ExtractedFields(
            doc_type=DocumentType.APPLICATION_FORM,
            source_filename="application.txt",
            fields={"applicant_name": "Jane Doe"},
        ),
    ]

    result = validate_extracted_data(extracted)

    assert not result.is_valid
    assert any(issue.field == "requested_amount" for issue in result.issues)
