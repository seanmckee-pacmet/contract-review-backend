from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from supabase import create_client, Client
import os
import json
import difflib
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
    Perform a semantic search on a specific list of documents using a query.
    Use this tool when you need to search for specific information within a certain set of documents.

    Args:
        query (str): The search query to look for in the documents
        doc_ids (list[str]): A list of document IDs to search through

    Returns:
        list[dict]: A list of documents that match the query
    """
    # from the query generate phrases to perform multiple semantic searches on to cover all the nuances of the query

@tool
def search_for_specific_quality_clause(clause_id: str, doc_id: str) -> dict:
    """
    Search for a specific quality clause id that closely matches the header of a chunk.


    Args:
        query (str): The search query to look for in the documents
        clause_id (str): The ID of the clause to search for
        doc_id (str): The ID of the document to search in
    Returns:
        dict: A dictionary containing the clause analysis
    """
    clause_search = supabase.table('Chunks').select('header', 'content').eq('document_id', doc_id).ilike('header', f'%{clause_id}%').execute()
    if not clause_search.data:
        raise ValueError(f"No clause found for ID: {clause_id}")
    
    return json.loads(clause_search.data[0])

@tool
def check_if_clause_is_invoked(po_rules: dict, clause_id: str, doc_id: str) -> bool:
    """
    Check if a specific quality clause id is invoked in a document.

    Args:
        po_rules (dict): The rules of the PO
        clause_id (str): The ID of the clause to search for
        doc_id (str): The ID of the document to search in
    Returns:
        bool: True if the clause is invoked, False otherwise
    """


    def is_similar(str1, str2, threshold=0.8):
        return difflib.SequenceMatcher(None, str1, str2).ratio() >= threshold
    # if po_rules['clause_identifiers'] is not empty
    if len(po_rules['clause_identifiers']) > 0:
        for rule in po_rules['clause_identifiers']:
            if is_similar(clause_id, rule, threshold=0.8):
                return True
    return False


# tool instantiation
tools = [get_po_rules, semantic_search_docs, search_for_specific_quality_clause, check_if_clause_is_invoked]



# prompt
prompt = PromptTemplate(
    input_variables=["query", "doc_ids"],
    template="""
    You are a contract review assistant for an aerospce manufacturer. You are given a query and a list of document ids.
    Depending on the query and the document ids, you may need to use one or more of the tools provided to you.
    Think carefully about what tools will best answer the query.
    You must also think about whether the query can be answered by the documents provided.
    If the query can be answered by the documents, you must use the get_po_rules tool to get the rules of the PO.
    You may also use the semantic_search_docs tool to search for the query in the documents.
    You may use the search_for_specific_quality_clause tool to search for a specific quality clause in the quality document.
    You may use the check_if_clause_is_invoked tool to check if the query is invoked in the documents.
    You must return the results of the tools to the user.

    Here is the query:
    {query}

    Here is the list of document ids:
    {doc_ids}
""",
)

# create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # run agent executor
# agent_executor.invoke({"input": "What is the weather in Tokyo?"})
