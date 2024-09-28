from typing import List
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
from openai import OpenAI
import os
import json

router = APIRouter()
load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)
client = OpenAI()

class DocIDs(BaseModel):
    ids: List[str]

@router.post("/review")
async def review(doc_ids: DocIDs, review_criteria_group_id: str):

    # get all required data from documents
    documents = []
    for doc_id in doc_ids.ids:  
        result = supabase.table("Documents").select("*").eq("id", doc_id).execute()
        documents.append(result.data[0])
    
    review_criteria_clause_ids = supabase.table("groups_clauses").select("clause_id").eq("group_id", review_criteria_group_id).execute()

    review_criteria_clauses = []
    for clause_id in review_criteria_clause_ids.data:
        clause = supabase.table("clauses").select("*").eq("id", clause_id["clause_id"]).execute()
        review_criteria_clauses.append(clause.data[0])

    # find which index of the document is the PO
    po_index = 0
    for i, doc in enumerate(documents):
        if doc["doc_type"] == "PO":
            po_index = i
            break
    qd_index = 0
    for i, doc in enumerate(documents):
        if doc["doc_type"] == "QD":
            qd_index = i
            break

    # get PO analysis
    po_analysis = supabase.table("po_analysis").select("analysis").eq("doc_id", doc_ids.ids[po_index]).execute()

    # Parse the JSON string in the analysis field
    analysis_data = json.loads(po_analysis.data[0]["analysis"])

    # Check if all_invoked is true
    all_invoked = analysis_data.get("all_invoked", False)

    # You can now use the all_invoked variable in your logic
    if all_invoked:
        # Handle the case when all clauses are invoked
        # semantic search for each Name: Description pair in review_criteria_clauses
        clauses_with_matched_documents = []
        for clause in review_criteria_clauses:
            query = f"{clause['name']}: {clause['description']}"
            query_embedding = get_embedding(query)
            
            response = supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'document_ids': doc_ids.ids,
                    'match_threshold': 0.1,
                    'match_count': 5
                }
            ).execute()
            
            matched_documents = response.data
            # Process matched_documents as needed
            # For example, you could add them to the clause dictionary
            clause['matched_documents'] = matched_documents
            clauses_with_matched_documents.append(clause)
    else:
        # Handle the case when not all clauses are invoked
        # Search for chunks with clause identifiers in their headers
        clauses_with_matched_chunks = []
        for clause_id in analysis_data["clause_identifiers"]:
            response = supabase.rpc(
                'search_chunks_by_header_and_doc_id',
                {
                    'search_term': clause_id,
                    'doc_id': doc_ids.ids[qd_index]
                }
            ).execute()
            clause_data = {
                "clauseId": clause_id,
                "matchedChunks": response.data
            }
            
            clauses_with_matched_chunks.append(clause_data)

    # do a semantic search on the other documents for each clause besides the QD
    semantic_search_results = []
    for clause in review_criteria_clauses:
        if clause["id"] != review_criteria_group_id:
            query = f"{clause['name']}: {clause['description']}"
            query_embedding = get_embedding(query)

            for doc in documents:
                if doc["id"] != doc_ids.ids[qd_index]:
                    response = supabase.rpc(
                        'match_documents',
                        {
                            'query_embedding': query_embedding,
                            'document_ids': [doc["id"]],
                            'match_threshold': 0.1,
                            'match_count': 5
                        }
                    ).execute() 
                    matched_chunks = []
                    for chunk in response.data:
                        matched_chunk = {
                            "content": chunk['content']
                        }
                        if 'header' in chunk:
                            matched_chunk["header"] = chunk['header']
                        matched_chunks.append(matched_chunk)
                    clause_data = {
                        "clauseId": clause["id"],
                        "matchedChunks": matched_chunks
                    }
                    semantic_search_results.append(clause_data) 

    # Create context from clauses_with_matched_chunks and semantic_search_results
    context = ""
    
    # Add clauses_with_matched_chunks to context
    for clause in clauses_with_matched_chunks:
        context += f"Clause ID: {clause['clauseId']}\n"
        for chunk in clause['matchedChunks']:
            context += f"Header: {chunk['header']}\n"
            context += f"Content: {chunk['content']}\n\n"
    
    # Add semantic_search_results to context (only header and content)
    for result in semantic_search_results:
        context += f"Clause ID: {result['clauseId']}\n"
        for chunk in result['matchedChunks']:
            if 'header' in chunk:
                context += f"Header: {chunk['header']}\n"
            context += f"Content: {chunk['content']}\n\n"



    
    
    class Clause(BaseModel):
        name: str
        relevantChunks: List[str]

    class Review(BaseModel):
        clauses: List[Clause]

    # create a review with open ai
    review = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Create a contract review of the documents provided based on the review criteria."},
            {"role": "user", "content": "Here is the context: " + str(context) +
             "Here are the review criteria: " + str(review_criteria_clauses) +
             "Using the context, create a review of the contract based on the review criteria. Only include clauses that are relevant to the review criteria."
             "For each clause being searched for use relevant quotes from the chunks to create a cohesive and complete review of the contract."
             }
        ],
        response_format=Review
    )
    return json.loads(review.choices[0].message.content)

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
