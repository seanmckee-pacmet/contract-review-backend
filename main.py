from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.analysis import router as analysis_router
from api.routes.document import router as document_router
from api.routes.review import router as review_router
from api.routes.chat import router as chat_router
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])
app.include_router(document_router, prefix="/documents", tags=["Documents"])
app.include_router(review_router, prefix="/reviews", tags=["Reviews"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Contract Review API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)