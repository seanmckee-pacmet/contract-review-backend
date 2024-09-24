import os
from supabase import create_client, Client
from dotenv import load_dotenv
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import uuid

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

# TODO: add functions for supabase

# add company
def add_company(company_name: str):
    supabase.table("Companies").insert({"name": company_name}).execute()

# get companies
def get_companies():
    response = supabase.table("Companies").select("*").execute()
    return response.data

# delete company
def delete_company(company_id: str):
    supabase.table("Companies").delete().eq("id", company_id).execute()

# upload document
def upload_document(company_id: str, file_path: str, doc_type: str):
    return ''


# get documents
def get_documents(company_id: str):
    response = supabase.table("Documents").select("*").eq("company_id", company_id).execute()
    return response.data

# delete document
def delete_document(document_id: str):
    supabase.table("Documents").delete().eq("id", document_id).execute()

