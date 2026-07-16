"""FastAPI app: upload loan documents, get back an Excel decision report."""
from __future__ import annotations

from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse

from loan_processing.output.excel_writer import write_excel
from loan_processing.pipeline import run_pipeline

app = FastAPI(title="Loan Processing")


@app.post("/applications/process")
async def process_application(files: list[UploadFile]) -> StreamingResponse:
    uploads = [(file.filename or "unknown", await file.read()) for file in files]
    result = run_pipeline(uploads)
    workbook = write_excel(result)

    headers = {"Content-Disposition": f'attachment; filename="{result.application_id}.xlsx"'}
    return StreamingResponse(
        workbook,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
