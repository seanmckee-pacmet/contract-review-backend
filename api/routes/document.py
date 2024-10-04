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
from src.po_analysis import review_po
from src.embeddings import create_embeddings
from src.document_processing import chunk_markdown_text, determine_document_type
from src.get_formatted_text import parse_document
import asyncio
import nest_asyncio
from PIL import Image
import pytesseract

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
    print(f"Starting upload process for file: {file.filename}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    print(f"Temporary file created at: {temp_file_path}")

    try:
        # Parse document to get markdown text
        markdown_content = parse_document(temp_file_path)
        print("Document parsed and converted to markdown")

        # Determine document type
        doc_type = determine_document_type(markdown_content[:2000])
        print(f"Determined document type: {doc_type}")

        if doc_type not in ['QD', 'TC', 'PO']:
            print(f"Unsupported document type: {doc_type}")
            return {"error": f"Unsupported document type: {doc_type}"}
        
        print("Chunking and embedding content")
        chunks = chunk_markdown_text(markdown_content)
        print(f"Created {len(chunks)} chunks")
        embeddings = create_embeddings(chunks)
        print("Embeddings created")

        print("Inserting document into Supabase")
        result = supabase.table("Documents").insert({
            "company_id": company_id,
            "name": file.filename,
            "doc_type": doc_type
        }).execute()
        
        document_id = result.data[0]['id']
        print(f"Document inserted with ID: {document_id}")

        print("Upserting chunks to Supabase")
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            supabase.table("Chunks").upsert({
                "document_id": document_id,
                "content": chunk['metadata']['header'] + ": " + chunk['page_content'],
                "embedding": embedding,
                "header": chunk['metadata']['header']
            }).execute()
            if i % 10 == 0:  # Print progress every 10 chunks
                print(f"Upserted {i+1}/{len(chunks)} chunks")

        if doc_type == 'PO':
            print("Running PO analysis")
            po_analysis = review_po(markdown_content)
            # Convert POAnalysisResponse to a dictionary
            po_analysis_dict = po_analysis.dict()
            supabase.table("po_analysis").insert({
                "doc_id": document_id,  # Changed from "document_id" to "doc_id"
                "analysis": po_analysis_dict
            }).execute()
            print("PO analysis completed and stored")

        print("Upload process completed successfully")
        return {"filename": file.filename, "doc_type": doc_type, "document_id": document_id}

    except Exception as e:
        print(f"Error during upload process: {str(e)}")
        raise

    finally:
        print(f"Cleaning up temporary file: {temp_file_path}")
        os.unlink(temp_file_path)

# list documents
@router.get("/{company_id}")
async def list_documents(company_id: str):
    return supabase.table("Documents").select("*").eq("company_id", company_id).execute()


# get document chunks with headers
@router.get("/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    return supabase.table("Chunks").select("*").eq("document_id", document_id).execute()

# extract, upload, and analyze purchase order (multi-page tiff file)
@router.post("/purchase-order/{document_id}")
async def upload_purchase_order(document_id: str, file: UploadFile = File(...)):
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tiff') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
        
    # Set the path to the Tesseract executable

    try:
        # Open the multi-page TIFF file
        with Image.open(temp_file_path) as img:
            text = ""
            # Iterate through all pages
            for i in range(img.n_frames):
                img.seek(i)
                # Extract text from each page
                page_text = pytesseract.image_to_string(img)
                text += f"Page {i+1}:\n{page_text}\n\n"

        # Process the extracted text (you can add more processing here)

        # analyze the purchase order
        analysis = review_po(text)

        

        # For now, we'll just return the extracted text
        return {"document_id": document_id, "extracted_text": text, "analysis": analysis}

    finally:
        # Clean up: remove the temporary file
        os.unlink(temp_file_path)

# ... rest of the code ...






