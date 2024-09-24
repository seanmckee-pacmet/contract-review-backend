import os
import tempfile
from supabase import create_client, Client
from dotenv import load_dotenv
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import uuid

from fastapi import APIRouter, UploadFile
from src.document_processing import chunk_markdown_text, determine_document_type
from src.get_formatted_text import parse_document
router = APIRouter()


load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

# CRUD for managing documents and companies


# COMPANY ROUTES

# get companies
@router.get("/companies")
async def list_companies():
    return supabase.table("Companies").select("*").execute()

# add company
@router.post("/company")
async def add_company(company_name: str):
    supabase.table("Companies").insert({"name": company_name}).execute()
    return {"message": "Company added successfully", "company_name": company_name}

# delete company
@router.delete("/company/{company_id}")
async def delete_company(company_id: str):
    supabase.table("Companies").delete().eq("id", company_id).execute()
    return {"message": "Company deleted successfully", "company_id": company_id}


# DOCUMENT ROUTES

# delete document
@router.delete("/document/{document_id}")
async def delete_document(document_id: int):
    return {"message": "Document deleted successfully", "document_id": document_id}


# upload document
# chunking, embedding, storing embeddings in supabase as well as the doc name, company_id, doctype, content, doc id
# Chunks: id, document_id, content, embedding
# Documents: id, doc_type, company_id, name
@router.post("/upload/{company_id}")
async def upload_document(company_id: str, file: UploadFile):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the file path
        file_path = os.path.join(temp_dir, file.filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        print("file_path: ", file_path)
        
        # Move these operations inside the tempfile context
        content = parse_document(file_path)
        print("content: ", content)
        doc_type = determine_document_type(content)
        print("doc_type: ", doc_type)
        chunks = chunk_markdown_text(content)
        print("chunks: ", chunks)
    # Return statement can be outside the context manager
    return {"number_of_chunks": len(chunks), "first_chunk": chunks[0]}


# list documents
@router.get("/documents/{company_id}")
async def list_documents(company_id: int):
    return {"documents": ["doc1.pdf", "doc2.pdf", "doc3.pdf"]}  # Placeholder






