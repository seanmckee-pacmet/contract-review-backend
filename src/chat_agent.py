from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from supabase import create_client, Client
import os
import json
load_dotenv()

# connect to supabase
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# tools

# tool definitions
@tool
def get_po_rules(doc_id: str) -> dict:
    """
    Get information about a Purchase Order (PO) including:
    - Whether it invokes the entire quality document
    - Specific clauses invoked from the quality document
    - Any other relevant detected information about the PO

    Args:
        doc_id (str): The unique identifier for the PO document

    Returns:
        dict: A dictionary containing the PO analysis
    """
    response = supabase.table("po_analysis").select("analysis").eq("id", doc_id).execute()
    
    if not response.data:
        raise ValueError(f"No analysis found for document ID: {doc_id}")
    
    return json.loads(response.data[0]["analysis"])

@tool
def semantic_search_docs(query: str, doc_ids: list[str]) -> list[dict]:
    """
    Perform a semantic search on a list of documents using a query.
    Use this tool when you need to search for specific information within a certain set of documents.

    Args:
        query (str): The search query to look for in the documents
        doc_ids (list[str]): A list of document IDs to search through

    Returns:
        list[dict]: A list of documents that match the query
    """
    # from the query generate phrases to perform multiple semantic searches on to cover all the nuances of the query

 

# tool instantiation
tools = [get_po_rules]

# get po analysis


# prompt
prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a helpful assistant. Make sure to use the tavily_search_results_json tool for information.

{input}
""",
)

# create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # run agent executor
# agent_executor.invoke({"input": "What is the weather in Tokyo?"})
