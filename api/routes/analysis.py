from fastapi import APIRouter
from src.po_analysis import review_po  # Ensure this function exists
from src.clause_analysis import analyze_clauses_batch  # Ensure this function exists

router = APIRouter()

@router.post("/run")
async def run_analysis(document_id: str):
    po_result = review_po(document_id)
    clause_result = analyze_clauses_batch(document_id)
    return {"po_analysis": po_result, "clause_analysis": clause_result}

@router.get("/results/{document_id}")
async def get_analysis_results(document_id: str):
    return {"document_id": document_id, "results": "Analysis results here"}  # Placeholder

