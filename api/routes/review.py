from typing import List, Dict
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

class Clause(BaseModel):
    header: str
    content: str

class QuoteInfo(BaseModel):
    document_type: str
    header: str
    content: str

class ClauseReview(BaseModel):
    clause_name: str
    clause_description: str
    quotes: List[QuoteInfo]

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

async def get_documents(doc_ids: List[str]) -> List[Dict]:
    """Fetch documents from Supabase."""
    documents = []
    for doc_id in doc_ids:
        result = supabase.table("Documents").select("*").eq("id", doc_id).execute()
        documents.append(result.data[0])
    return documents

async def get_po_analysis(po_id: str) -> Dict:
    """Fetch PO analysis from Supabase."""
    po_analysis = supabase.table("po_analysis").select("analysis").eq("doc_id", po_id).execute()
    return json.loads(po_analysis.data[0]["analysis"])

async def get_clauses(review_criteria_group_id: str) -> List[Dict]:
    """Fetch clauses based on review criteria group."""
    criteria = supabase.table("groups_clauses").select("*").eq("group_id", review_criteria_group_id).execute()
    clauses = []
    for item in criteria.data:
        clause_info = supabase.table("clauses").select("name", "description").eq("id", item['clause_id']).execute()
        if clause_info.data:
            clauses.append(clause_info.data[0])
    return clauses

async def get_document_types(doc_ids: List[str]) -> Dict[str, str]:
    """Get document types for each document ID."""
    doc_types = {}
    for doc_id in doc_ids:
        doc_type_result = supabase.table("Documents").select("doc_type").eq("id", doc_id).execute()
        doc_types[doc_id] = doc_type_result.data[0]['doc_type']
    return doc_types

async def perform_semantic_search(clause: Dict, doc_ids: List[str], doc_types: Dict[str, str]) -> Dict[str, List]:
    """Perform semantic search for a clause across all documents."""
    query = f"{clause['name']}: {clause['description']}"
    query_embedding = get_embedding(query)
    results_by_doc_type = {}

    for doc_id in doc_ids:
        doc_type = doc_types[doc_id]
        response = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'document_ids': [doc_id],
                'match_threshold': 0.1,
                'match_count': 3
            }
        ).execute()
        
        if doc_type not in results_by_doc_type:
            results_by_doc_type[doc_type] = []
        results_by_doc_type[doc_type].extend(response.data)

    return results_by_doc_type

def build_context(clause: Dict, results_by_doc_type: Dict[str, List]) -> str:
    """Build context string from search results with clear labeling."""
    context = f"CLAUSE: {clause['name']}\nDESCRIPTION: {clause['description']}\n\n"  # Added description
    for doc_type, results in results_by_doc_type.items():
        context += f"DOCUMENT_TYPE: {doc_type}\n"
        for result in results:
            if 'header' in result:
                context += f"HEADER: {result['header']}\n"
            context += f"QUOTE: {result['content']}\n\n"
        context += "\n"
    return context


async def get_clause_review(clause_name: str, clause_description: str, context: str, analysis_data: Dict) -> Dict:
    """Get clause review using OpenAI API."""
    review = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an expert aerospace contract reviewer. Extract relevant quotes from the context for the given clause."},
            {"role": "user", "content": f"""
                Clause name: {clause_name}
                Clause description: {clause_description}
                Context: {context}
                The context is organized by document type, header, and quote. Include relevant quotes from TC or PO documents.
                For the QD document, only include quotes with headers in this list: {analysis_data['clause_identifiers']}
                Extract all relevant quotes for this clause, including document type, header, and content.
                Follow these steps:
                1. Read the clause name and description carefully.
                2. Read each document's content, looking for relevant quotes.
                3. Include only relevant quotes in the review.
                4. Ensure QD document quotes have headers from the provided list.
                5. Return the clause name, clause description, and a list of relevant quotes (document type, header, and content).
                6. Double-check that all quotes are relevant and have appropriate headers.
                Only include quote information, no additional context or explanations.
                Make absolutely certain that each quote you include directly mentions or infers the invocation of the clause.
                Otherwise do not include it.
                Format each quote as follows:
                {{
                    "document_type": "The type of document (TC, PO, or QD)",
                    "header": "The header or clause title",
                    "content": "The actual quote content"
                }}

                At the beginning of the list, include a quote that is labled as follows:
                {{
                    "document_type": "Summary",
                    "header": "Summary",
                    "content": "A summary of the clause and its purpose" 
                    // This quote should be a summary of the results of the clause review.
                    // information such as what was found and where it was found. do not explain what the clause definition it.
                }}
            """}
        ],
        response_format=ClauseReview
    )
    return json.loads(review.choices[0].message.content)

@router.post("/review1")
async def review1(doc_ids: DocIDs, review_criteria_group_id: str):
    """Main function to process document review."""
    documents = await get_documents(doc_ids.ids)
    po_id = next(doc["id"] for doc in documents if doc["doc_type"] == "PO")
    analysis_data = await get_po_analysis(po_id)
    clauses = await get_clauses(review_criteria_group_id)
    doc_types = await get_document_types(doc_ids.ids)

    clause_reviews = []
    for clause in clauses:
        results_by_doc_type = await perform_semantic_search(clause, doc_ids.ids, doc_types)
        context = build_context(clause, results_by_doc_type)
        clause_review = await get_clause_review(clause['name'], clause['description'], context, analysis_data)
        clause_reviews.append(clause_review)

    # Process empty reviews
    for i, clause_review in enumerate(clause_reviews):
        if len(clause_review['quotes']) == 0:
            clause_review = await get_clause_review(clause_review['clause_name'], clause_review['clause_description'], context, analysis_data)
            if len(clause_review['quotes']) == 0:
                clause_review['quotes'].append({
                    "document_type": "N/A",
                    "header": "No Relevant Quotes",
                    "content": "No relevant quotes found"
                })
        clause_reviews[i] = clause_review

    # Filter and deduplicate clause reviews
    filtered_clause_reviews = []
    seen_clauses = set()
    for clause_review in clause_reviews:
        clause_name = clause_review['clause_name']
        if (len(clause_review['quotes']) > 0 and clause_review['quotes'][0]['header'] != "No Relevant Quotes") or clause_name not in seen_clauses:
            filtered_clause_reviews.append(json.dumps(clause_review))
            seen_clauses.add(clause_name)

    return filtered_clause_reviews