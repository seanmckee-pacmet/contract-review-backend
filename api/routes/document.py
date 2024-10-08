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
from src.get_formatted_text import parse_document, get_plain_text
import asyncio
import nest_asyncio
from PIL import Image
import pytesseract
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
from openai import OpenAI

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

@router.post("/advanced-upload/{company_id}")
async def advanced_upload(company_id: str, file: UploadFile = File(...)):

    class Header(BaseModel):
        header: str = Field(description="The header or clause ID of the document")
        description: str = Field(description="A brief description of the header or clause")
    
    class Headers(BaseModel):
        headers: list[Header]

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # get plain text from file
        plain_text = get_plain_text(temp_file_path)  # Pass the file path instead of file object
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        
        
        model = ChatOpenAI(model="gpt-4o", temperature=0)  # Updated model name
        structured_llm = model.with_structured_output(Headers)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant specialized in analyzing legal documents. 
            Your task is to identify clause headers/IDs first, then general titles/headers if they are not clauses from the given document text. 
            These headers should be able to classify any text in the document.
            
            For each header, provide a brief description of its purpose or content.
             
             Do not miss or skip any headers or clause IDs
            
            Return the results in a structured format with a list of headers, each containing a 'header' and 'description' field."""),
            ("human", "Please analyze the following document text and extract the clause headers or IDs with descriptions. Return the results in the specified structured format.\n\nDocument text:\n{plain_text}")
        ])
        formatted_prompt = prompt_template.format_messages(plain_text=plain_text)
        headers = structured_llm.invoke(formatted_prompt)

        doc_type = determine_document_type(plain_text[:2000])

        # using langchian and open ai batch processing, chunk document, then with a batch process with open ai, assign headers to each chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n"])
        chunks = splitter.create_documents([plain_text])
        

        # Create OpenAI client
        client = OpenAI()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Process chunks and assign headers
        chunk_headers = []
        for i, chunk in enumerate(chunks):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are an AI assistant specialized in analyzing legal documents. Your task is to determine the most appropriate header for the given chunk of text based on the provided headers. Here are the available headers: {json.dumps([h.dict() for h in headers.headers])}"},
                    {"role": "user", "content": f"Please analyze the following chunk of text and return only the name of the most appropriate header from the list provided:\n\n{chunk.page_content}"}
                ],
            )
            assigned_header = response.choices[0].message.content.strip()
            
            # Create embedding for the chunk
            embedding = embeddings.embed_query(chunk.page_content)

            chunk_headers.append({
                "chunk_id": f"chunk-{i}", 
                "header": assigned_header,
                "embedding": embedding
            })

        # Insert document into Supabase
        result = supabase.table("Documents").insert({
            "company_id": company_id,
            "name": file.filename,
            "doc_type": doc_type
        }).execute()
        
        document_id = result.data[0]['id']

        # Insert chunks with assigned headers and embeddings into Supabase
        for i, (chunk, header_info) in enumerate(zip(chunks, chunk_headers)):
            supabase.table("Chunks").insert({
                "document_id": document_id,
                "content": chunk.page_content,
                "header": header_info['header'],
                "embedding": header_info['embedding']
            }).execute()

        return {
            "message": "Advanced upload process completed",
            "company_id": company_id,
            "filename": file.filename,
            "document_id": document_id,
            "headers": headers.dict(),
            "chunk_headers": chunk_headers
        }

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

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






