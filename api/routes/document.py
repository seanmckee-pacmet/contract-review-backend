import os
import shutil
import tempfile
import PyPDF2
from supabase import create_client, Client
from dotenv import load_dotenv
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import uuid
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from fastapi import APIRouter, File, UploadFile
from src.embeddings import create_embeddings
from src.document_processing import chunk_markdown_text, determine_document_type
from src.get_formatted_text import parse_document
import asyncio
import nest_asyncio
router = APIRouter()
load_dotenv()
nest_asyncio.apply()

router = APIRouter()
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
async def delete_document(document_id: str):
    supabase.table("Documents").delete().eq("id", document_id).execute()
    # supabase.table("Chunks").delete().eq("document_id", document_id).execute()
    return {"message": "Document deleted successfully", "document_id": document_id}

# upload document
# chunking, embedding, storing embeddings in supabase as well as the doc name, company_id, doctype, content, doc id
# Chunks: id, document_id, content, embedding
# Documents: id, doc_type, company_id, name
@router.post("/upload/{company_id}")
async def upload_document(company_id: str, file: UploadFile = File(...)):
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Set up LlamaParse with options to process all pages
        parser = LlamaParse(
            result_type="markdown",
            num_workers=2,  # Adjust based on your system's capabilities
            max_chunks=None  # Process all chunks (pages)
        )
        
        # Use SimpleDirectoryReader to parse the file
        file_extractor = {".pdf": parser}
        
        # Run the parsing in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(
            None,
            lambda: SimpleDirectoryReader(input_files=[temp_file_path], file_extractor=file_extractor).load_data()
        )
        
        # Combine all parsed documents into a single markdown string
        markdown_content = "\n\n".join([doc.text for doc in documents]) if documents else "No content extracted"

        # TODO: Process the markdown_content (chunking, embedding, storing in Supabase)
        chunks = chunk_markdown_text(markdown_content)
        embeddings = create_embeddings(chunks)

        # Insert the document and retrieve the id
        result = supabase.table("Documents").insert({
            "company_id": company_id,
            "name": file.filename,
            "doc_type": "PDF"
        }).execute()
        
        # Extract the id from the result
        document_id = result.data[0]['id']
        # upsert to supabase Chunks table with id, document_id, content, embedding
        for chunk, embedding in zip(chunks, embeddings):
            supabase.table("Chunks").upsert({
                "document_id": document_id,  # You need to define this earlier
                "content": chunk['metadata']['header'] + ": " + chunk['page_content'],
                "embedding": embedding,
                "header": chunk['metadata']['header']
            }).execute()

        return {"filename": file.filename, "contents": markdown_content, "document_id": document_id}
    finally:
        # Clean up: remove the temporary file
        os.unlink(temp_file_path)
    # get text from pdf
    


# list documents
@router.get("/{company_id}")
async def list_documents(company_id: str):
    return supabase.table("Documents").select("*").eq("company_id", company_id).execute()


# get document chunks with headers
@router.get("/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    return supabase.table("Chunks").select("*").eq("document_id", document_id).execute()






