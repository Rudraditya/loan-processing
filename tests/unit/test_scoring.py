from loan_processing.models import DocumentType, ExtractedFields
from loan_processing.scoring.scorer import compute_score


def _doc(doc_type: DocumentType, **fields: str) -> ExtractedFields:
    return ExtractedFields(doc_type=doc_type, source_filename=f"{doc_type.value}.txt", fields=fields)


def test_strong_applicant_scores_high_and_low_risk():
    extracted = [
        _doc(DocumentType.TAX_FORM, annual_income="120000"),
        _doc(DocumentType.APPLICATION_FORM, monthly_debt="500"),
        _doc(DocumentType.BANK_STATEMENT, closing_balance="15000"),
    ]

    result = compute_score(extracted)

    assert result.score > 650
    assert result.risk_tier == "low"
    assert result.factors["debt_to_income_ratio"] < 0.2


def test_missing_income_is_penalized():
    extracted = [_doc(DocumentType.APPLICATION_FORM, monthly_debt="500")]

    result = compute_score(extracted)

    assert result.factors["debt_to_income_ratio"] == -1.0
    assert result.score < 650


def test_high_debt_to_income_lowers_score():
    extracted = [
        _doc(DocumentType.TAX_FORM, annual_income="24000"),
        _doc(DocumentType.APPLICATION_FORM, monthly_debt="1800"),
    ]

    result = compute_score(extracted)

    assert result.factors["debt_to_income_ratio"] > 0.5
    assert result.risk_tier == "high"
