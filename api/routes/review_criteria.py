from fastapi import APIRouter, HTTPException
from typing import List, Dict
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Initialize Supabase client
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

# Get all criteria groups
@router.get("/criteria_groups")
async def get_criteria_groups():
    response = supabase.table("criteria_groups").select("*").execute()
    return {"criteria_groups": response.data}

# Create a new criteria group
@router.post("/criteria_groups")
async def create_criteria_group(criteria_group: Dict):
    response = supabase.table("criteria_groups").insert(criteria_group).execute()
    return {"message": "Criteria group created successfully", "criteria_group": response.data[0]}

# Update a criteria group
@router.put("/criteria_groups/{criteria_group_id}")
async def update_criteria_group(criteria_group_id: str, criteria_group: Dict):
    response = supabase.table("criteria_groups").update(criteria_group).eq("id", criteria_group_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Criteria group not found")
    return {"message": "Criteria group updated successfully", "criteria_group": response.data[0]}

# Delete a criteria group
@router.delete("/criteria_groups/{criteria_group_id}")
async def delete_criteria_group(criteria_group_id: str):
    response = supabase.table("criteria_groups").delete().eq("id", criteria_group_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Criteria group not found")
    return {"message": "Criteria group deleted successfully", "criteria_group_id": criteria_group_id}

# Get all clauses
@router.get("/clauses")
async def get_clauses():
    response = supabase.table("clauses").select("*").execute()
    return {"clauses": response.data}
