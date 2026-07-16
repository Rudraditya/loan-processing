"""Orchestrates the stage sequence: ingest -> extract -> validate -> score -> decide."""
from __future__ import annotations

import uuid

from loan_processing.decisioning.decision_engine import decide
from loan_processing.extraction.extractor import extract_fields
from loan_processing.ingestion.document_loader import load_document
from loan_processing.models import LoanApplicationResult
from loan_processing.scoring.scorer import compute_score
from loan_processing.validation.validator import validate_extracted_data


def run_pipeline(files: list[tuple[str, bytes]]) -> LoanApplicationResult:
    """Run the full pipeline over a set of (filename, content) uploads."""
    documents = [load_document(filename, content) for filename, content in files]
    extracted = [
        extract_fields(doc.doc_type, doc.raw_text, doc.filename) for doc in documents
    ]

    validation = validate_extracted_data(extracted)
    score = compute_score(extracted)
    decision = decide(score, validation)

    return LoanApplicationResult(
        application_id=str(uuid.uuid4()),
        extracted=extracted,
        validation=validation,
        score=score,
        decision=decision,
    )
