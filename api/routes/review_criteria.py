from fastapi import APIRouter, HTTPException
from typing import List, Dict
from openai import OpenAI
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

router = APIRouter()

# Initialize Supabase client
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

client = OpenAI()

# create a new criteria group
@router.post("/criteria_groups")
async def create_criteria_group(name: str):
    # create new row in criteria_groups table
    response = supabase.table("criteria_groups").insert({"name": name}).execute()
    return {"message": "Criteria group created successfully", "criteria_group": response.data[0]}

# add a new clause to a criteria group
@router.post("/criteria_groups/{criteria_group_id}/clauses")
async def create_clause(criteria_group_id: str, name: str, description: str):
    # create new row in clauses table
    # takes name and description then in groups_clauses table they must be linked
    response = supabase.table("clauses").insert({"name": name, "description": description}).execute()
    supabase.table("groups_clauses").insert({"group_id": criteria_group_id, "clause_id": response.data[0]['id']}).execute()
    return {"message": "Clause created successfully", "clause": response.data[0]}

# get all criteria groups
@router.get("/criteria_groups")
async def get_criteria_groups():
    response = supabase.table("criteria_groups").select("*").execute()
    return {"criteria_groups": response.data}

# get all clauses for a criteria group
@router.get("/criteria_groups/{criteria_group_id}/clauses")
async def get_clauses(criteria_group_id: str):
    # Fetch criteria group information
    criteria_group_response = supabase.table("criteria_groups").select("*").eq("id", criteria_group_id).execute()
    if not criteria_group_response.data:
        raise HTTPException(status_code=404, detail="Criteria group not found")
    
    criteria_group = criteria_group_response.data[0]

    # Fetch clauses associated with the criteria group
    clauses_response = supabase.table("groups_clauses").select("*").eq("group_id", criteria_group_id).execute()
    clause_ids = [clause['clause_id'] for clause in clauses_response.data]
    
    # Fetch detailed information for each clause
    clauses_details_response = supabase.table("clauses").select("*").in_("id", clause_ids).execute()
    
    return {
        "criteria_group": criteria_group,
        "clauses": clauses_details_response.data
    }

# get all clauses
@router.get("/clauses")
async def get_clauses():
    response = supabase.table("clauses").select("*").execute()
    return {"clauses": response.data}

# add existing clause to a criteria group
@router.post("/criteria_groups/{criteria_group_id}/clauses/{clause_id}")
async def add_clause_to_criteria_group(criteria_group_id: str, clause_id: str):
    supabase.table("groups_clauses").insert({"group_id": criteria_group_id, "clause_id": clause_id}).execute()
    return {"message": "Clause added to criteria group successfully"}

# delete a clause from a criteria group
@router.delete("/criteria_groups/{criteria_group_id}/clauses/{clause_id}")
async def delete_clause_from_criteria_group(criteria_group_id: str, clause_id: str):
    supabase.table("groups_clauses").delete().eq("group_id", criteria_group_id).eq("clause_id", clause_id).execute()
    return {"message": "Clause deleted from criteria group successfully"}

# delete a criteria group
@router.delete("/criteria_groups/{criteria_group_id}")
async def delete_criteria_group(criteria_group_id: str):
    # Delete all associations in groups_clauses table
    supabase.table("groups_clauses").delete().eq("group_id", criteria_group_id).execute()
    # Delete the criteria group
    supabase.table("criteria_groups").delete().eq("id", criteria_group_id).execute()
    return {"message": "Criteria group deleted successfully"}

# delete a clause
@router.delete("/clauses/{clause_id}")
async def delete_clause(clause_id: str):
    # delete the relationships in groups_clauses table
    supabase.table("groups_clauses").delete().eq("clause_id", clause_id).execute()
    # delete the clause
    supabase.table("clauses").delete().eq("id", clause_id).execute()
    return {"message": "Clause deleted successfully"}

# remove a clause from a specific group
@router.delete("/criteria_groups/{criteria_group_id}/clauses/{clause_id}")
async def remove_clause_from_criteria_group(criteria_group_id: str, clause_id: str):
    supabase.table("groups_clauses").delete().eq("group_id", criteria_group_id).eq("clause_id", clause_id).execute()
    return {"message": "Clause removed from criteria group successfully"}

# get all criteria groups and all the clauses for each group
@router.get("/criteria_groups/all")
async def get_all_criteria_groups_and_clauses():
    # Fetch all criteria groups
    criteria_groups_response = supabase.table("criteria_groups").select("*").execute()
    criteria_groups = criteria_groups_response.data
    
    # Get all clauses for each group and return them all
    for group in criteria_groups:
        # Fetch clause IDs associated with the group
        clauses_response = supabase.table("groups_clauses").select("*").eq("group_id", group['id']).execute()
        clause_ids = [clause['clause_id'] for clause in clauses_response.data]
        
        # Fetch detailed information for each clause
        clauses_details_response = supabase.table("clauses").select("*").in_("id", clause_ids).execute()
        group['clauses'] = clauses_details_response.data
    
    return {"criteria_groups": criteria_groups}

# Update a clause
@router.put("/clauses/{clause_id}")
async def update_clause(clause_id: str, name: str, description: str):
    response = supabase.table("clauses").update({"name": name, "description": description}).eq("id", clause_id).execute()
    return {"message": "Clause updated successfully"}

# generate an ai recommendation for a clause description given the clause name
@router.post("/clauses/generate_description")
async def generate_clause_description(name: str):
    # call the openai api to generate a description
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates descriptions for clauses based on their names. These are for aerospace contract reviews."},
            {"role": "user", "content": f"Generate a brief description for the clause that will be used to semantically search for it in a document: {name}"}
        ],
        max_tokens=250
    )
    description = response.choices[0].message.content

    return {"description": description}

