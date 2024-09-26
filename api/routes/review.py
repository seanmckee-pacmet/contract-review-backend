from fastapi import APIRouter

router = APIRouter()

@router.post("/submit")
async def submit_review(document_id: str, review_text: str):
    return {"message": "Review submitted successfully", "document_id": document_id, "review": review_text}

@router.get("/reviews/{document_id}")
async def get_reviews(document_id: str):
    return {"document_id": document_id, "reviews": ["Review 1", "Review 2"]}  # Placeholder



# 