from typing import List
from fastapi import APIRouter
from openai import OpenAI

client = OpenAI()


router = APIRouter()

# Your chat-related route definitions go here
@router.post("/")
async def chat_endpoint(query: str):

    # get top k embeddings for the query using cosine similarity/hybrid search
    




    # pass those to openai 
    # return the response
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query +
             "answer: "}
        ]
    )
    return {"message": response.choices[0].message.content}

# More routes as needed