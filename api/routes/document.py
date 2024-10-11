import os
import tempfile
from supabase import create_client, Client
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from fastapi import APIRouter, File, UploadFile
from src.po_analysis import review_po
from src.embeddings import create_embeddings
from src.document_processing import chunk_markdown_text, determine_document_type
from src.get_formatted_text import parse_document, get_plain_text
import nest_asyncio
from PIL import Image
import pytesseract
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter
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

# @router.post("/upload/{company_id}")
# async def upload(company_id: str, file: UploadFile = File(...)):
#     # Create a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
#         content = await file.read()
#         temp_file.write(content)
#         temp_file_path = temp_file.name

#     try:
#         # Get plain text
#         plain_text = get_plain_text(temp_file_path)
        
#         # Get doc type
#         doc_type = determine_document_type(plain_text[:2000])
        
#         # Insert document into Supabase
#         document_result = supabase.table("Documents").insert({
#             "company_id": company_id,
#             "name": file.filename,
#             "doc_type": doc_type
#         }).execute()
#         document_id = document_result.data[0]['id']

#         if doc_type == "QD":
#             print("Uploading QD")
#             await qd_upload(plain_text, document_id, company_id, file.filename)
#             print("Uploading general")
#             await general_upload(plain_text, document_id, company_id, doc_type)
#         elif doc_type == "TC":
#             print("Uploading TC")
#             await general_upload(plain_text, document_id, company_id, doc_type)
#         elif doc_type == "PO":
#             print("Uploading PO")
#             po_analysis = review_po(plain_text)
#             supabase.table("po_analysis").insert({
#                 "doc_id": document_id,
#                 "analysis": po_analysis.dict()
#             }).execute()
#             print("Uploading general")
#             await general_upload(plain_text, document_id, company_id, doc_type)

#         return {"message": "Document uploaded successfully", "document_id": document_id, "doc_type": doc_type}

#     finally:
#         # Clean up the temporary file
#         os.unlink(temp_file_path)

# async def general_upload(plain_text: str, document_id: str, company_id: str, doc_type: str):
#     # llama parse to markdown
#     #llama parse to markdown
#     parser = LlamaParse(
#         result_type="markdown"
#     )

#     # markdown_content = parse_document(plain_text)

#     headers_to_split_on = [
#         ("#", "Header 1"),
#         ("##", "Header 2"),
#         ("###", "Header 3"),
#     ]

#     markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
#     md_header_splits = markdown_splitter.split_text(markdown_content)

#     # Define RecursiveCharacterTextSplitter
#     chunk_size = 800
#     chunk_overlap = 100
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, 
#         chunk_overlap=chunk_overlap

#     )

#     # Split the markdown header splits into smaller chunks
#     chunks = text_splitter.split_documents(md_header_splits)

#     print(f"Chunking completed. Total chunks: {len(chunks)}")
    
#     # Create embeddings
#     embedding_model = OpenAIEmbeddings()
#     embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
    
#     # Upsert chunks to supabase
#     print("Upserting chunks")
#     for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#         if len(chunk.page_content.strip()) > 1:  # Only upload chunks with more than one character
#             supabase.table("Chunks").upsert({
#                 "document_id": document_id,
#                 "content": chunk.page_content,
#                 "embedding": embedding,
#                 "header": chunk.metadata.get('header', doc_type)  # Use the header from metadata if available, otherwise use doc_type
#             }).execute()
        
#     print(f"Uploaded {len(chunks)} chunks")

# # @router.post("/advanced-upload/{company_id}")
# async def qd_upload(plain_text: str, document_id: str, company_id: str, file_name: str):

#     class Header(BaseModel):
#         header: str = Field(description="The header or clause ID of the document")
#         description: str = Field(description="A summary of the text associated with this header including all important details, numbers, figures, percentages, and dates")
    
#     class Headers(BaseModel):
#         headers: list[Header]


#     try:
#         # get plain text from file
#         os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        
#         model = ChatOpenAI(model="gpt-4o", temperature=0)  # Updated model name
#         structured_llm = model.with_structured_output(Headers)
#         prompt_template = ChatPromptTemplate.from_messages([
#             ("system", """You are an AI assistant specialized in analyzing legal documents. 
#             Your task is to identify clause headers/IDs first, then general titles/headers if they are not clauses from the given document text. 
#             These headers should be able to classify any text in the document.
#              Clause headers/IDs preferably should include the code without description or additional text.
#              For example: 1.1, 1.2, 1.3, etc. or A, B, C, etc. Or A1, A2, A3, etc.
#              Not: 1: Scope of Work, 2: Terms and Conditions, etc.
            
#             For each header, provide summary of the text associated with this header including all important details, numbers, figures, percentages, weights, dimensions, dates, and anything like that.
#              Include as much detail as possible that would be needed later.

#             It is important that you include ALL headers and descriptions in the response.
#              Do not ignore or skip any headers/clauses.

#             Return the results in a structured format with a list of headers, each containing a 'header' and 'description' field."""),
#             ("human", "Please analyze the following document text and extract the clause headers or IDs with descriptions. Return the results in the specified structured format.\n\nDocument text:\n{plain_text}. ")
#         ])
#         formatted_prompt = prompt_template.format_messages(plain_text=plain_text)
#         headers = structured_llm.invoke(formatted_prompt)
#         print(headers)

#         # Create embeddings
#         embeddings_model = OpenAIEmbeddings()
#         embeddings = embeddings_model.embed_documents([header.header + ": " + header.description for header in headers.headers])

#         #get doc type
#         doc_type = determine_document_type(plain_text[:2000])

#         # Upsert headers and embeddings to Supabase
#         for header, embedding in zip(headers.headers, embeddings):
#             supabase.table("Chunks").upsert({
#                 "document_id": document_id,
#                 "header": header.header,
#                 "content": header.description,
#                 "embedding": embedding
#             }).execute()

#         return {"message": "Document processed and stored successfully", "document_id": document_id}
#     finally:
#         pass
# upload document
# chunking, embedding, storing embeddings in supabase as well as the doc name, company_id, doctype, content, doc id
# Chunks: id, document_id, content, embedding
# Documents: id, doc_type, company_id, name
@router.post("/upload/{company_id}")
async def upload_document(company_id: str, file: UploadFile = File(...)):
    print("Calling upload document")
    print(f"Starting upload process for file: {file.filename}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        content = await file.read()
        print("File read")
        temp_file.write(content)
        print("File written to temp file")
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