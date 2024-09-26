from typing import List
from fastapi import APIRouter
from openai import OpenAI
from supabase import create_client, Client
import os

client = OpenAI()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

router = APIRouter()

# Your chat-related route definitions go here
@router.post("/")
async def chat_endpoint(query: str, document_ids: str):
    # Split document_ids into a list
    doc_ids = document_ids.split(',')

    # Get embedding for the query
    query_embedding = get_embedding(query)

    # Perform semantic search using Supabase's similarity_search function
    response = supabase.rpc(
        'match_documents',
        {
            'query_embedding': query_embedding,
            'document_ids': doc_ids,
            'match_threshold': 0.1,
            'match_count': 5
        }
    ).execute()

    # Extract the matched documents
    matched_documents = response.data

    # Prepare the context from matched documents
    context = "\n".join([doc['content'] for doc in matched_documents if 'content' in doc])
    print(context)
    # Pass the context and query to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question."
             "Only use the context provided to answer the question, do not use any other information."
             "Assume the user is asking a question about the documents provided."
            #  "If you cannot answer the question with the context provided, say 'From the context provided, I cannot answer that question.'"
             "If you cannot answer the question but the context might have information related, say something about the related information'"
             "Make your answers concise and to the point."
             "When possible, provide section numbers or clause ids in your answers."
             },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )
    return {"message": response.choices[0].message.content}

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# More routes as needed