from typing import List
from fastapi import APIRouter
from langchain_openai import ChatOpenAI
from openai import OpenAI
from supabase import create_client, Client
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
client = OpenAI()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

class QueryKeywords(BaseModel):
    keywords: list[str] = Field(description="A list of keywords extracted from the user's query")
    phrases: list[str] = Field(description="A list of phrases extracted from the user's query")
    clauses: list[str] = Field(description="A list of clauses extracted from the user's query")
    documents: list[str] = Field(description="A list of documents extracted from the user's query")



router = APIRouter()

@router.post("/test")
async def test_endpoint(query: str, document_ids: str):

    doc_ids = document_ids.split(',')
    context = ""
    llm = ChatOpenAI(model="gpt-4o-2024-08-06")
    structured_llm = llm.with_structured_output(QueryKeywords)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant well versed in aerospace manufacturing contracts."
         '''
         Before including any of the following ask yourself. Do I need this data to answer the question? and should this data be in a different category?
         clause IDs and documents do not go in phrases or keywords.
         Keywords:
         - Extract however many keywords are needed to fully express the main topic of the query.
         - Keywords should be short terms or variations of the main topic in the query.
         - keywords should reflect terms commonly found in aerospace contracts
         - Do not make keywords that will be very general, keywords should be specific words that will possibly return results that are relevant to the query.
         - do not mention clause ids or numbers in keywords.
         Phrases:
         - Extract however many phrases are needed to fully express the main topic of the query.
         - Phrases should be longer, complete thoughts or sentences that rephrase the main topic of the query.
         - Phrases should reflect how aerospace contracts are written and would be found in a quality document or terms and conditions.
         - Phrases are statements not questions.
         - Use keywords in the phrases to help with the semantic search.
         - Do not mention a contract just mimic how a clause would be written.
         - Do not mention clause ID or numbers.
         Clauses:
         - this is when the query specifically asks for a clause or a section of the contract.
         - specifically and explicitly mention the clause id or number.
         - Examples of clauses: WQR1, Clause A, 1.1, etc.
         - WQR1 = WQR1, Clause A = A, 1.1 = 1.1
         - do not include any if none are mentioned.
         Documents:
         - When user references a specifc document, extract the document type to reference so search can be filtered to only those documents.
         - Types of documents ( only these 3 ): QD (Quality Documents), TC (Terms and Conditions), PO (Purchase Order) 

         Example 1:
         Query: What does the contract say about subcontracts?
         Keywords: ['subcontracting', 'subcontract'] (note: 'subcontracting' is a keyword because it is a common term in aerospace contracts, 'contract' is not because it is too general)
         Phrases: ['buyer consent required for subcontracts', 'subcontracting is allowed']
         Clauses: []
         Documents: []

         Example 2:
         Query: What does the contract say about dfars?
         Keywords: ['dfars', 'defense federal acquisition regulation'] (note: 'dfars' is a keyword because it is a common term in aerospace contracts, 'defense' is not because it is too general)
         Phrases: ['dfars compliance is required', 'dfars clauses must be followed']
         Clauses: []
         Documents: []

         Example 3:
         Query: Is a Certificate of Conformance required?
         Keywords: ['certificate of conformance', 'CofC'] (note: 'certificate of conformance' is a keyword because it is a common term in aerospace contracts, 'conformance' is not because it is too general)
         Phrases: ['certificate of conformance is required', 'CofC must be provided with each shipment']
         Clauses: []
         Documents: []

         Example 4:
         Query: What does clause 1.1 say?
         Keywords: []
         Phrases: []
         Clauses: ['1.1']
         Documents: []

         Example 5:
         Query: What does TC document say about dfars?
         Keywords: ['dfars', 'defense federal acquisition regulation']
         Phrases: ['dfars compliance is required', 'dfars clauses must be followed']
         Clauses: []
         Documents: ['TC']
         '''),
        ("human", "{query}")
    ])
    formatted_prompt = prompt_template.format_messages(query=query)
    response = structured_llm.invoke(formatted_prompt)

    # get doc types from doc ids
    doc_types = []
    for doc_id in doc_ids:
        doc_type_response = supabase.table('Documents').select('doc_type').eq('id', doc_id).execute()
        if doc_type_response.data and len(doc_type_response.data) > 0:
            doc_types.append(doc_type_response.data[0]['doc_type'])
    print("doc_types:", doc_types)
    
    result = {
        "documents": response.documents,
        "clauses": response.clauses,
        "phrases": response.phrases,
        "keywords": response.keywords
    }

    print("result:", result)

    filter_by_doc_types = []
    po_analysis = ""

    # Conditional processing based on the content of each list
    if len(response.documents) > 0:
        # Process documents
        print("Documents:", response.documents)
        for document in response.documents:
            if document in doc_types:
                filter_by_doc_types.append(document)
            
            # Check if the document is a PO and fetch its analysis
            if document == 'PO':
                po_analysis_response = supabase.table('po_analysis').select('analysis').in_('doc_id', doc_ids).execute()
                if po_analysis_response.data and len(po_analysis_response.data) > 0:
                    po_analysis = "\n\nPO Analysis:\n" + "\n".join([item['analysis'] for item in po_analysis_response.data])
        
        print("filter_by_doc_types:", filter_by_doc_types)

    if len(response.clauses) > 0:
        # Process clauses
        # For example, you might want to fetch the full text of these clauses
        clause_search_results = []
        for clause in response.clauses:
            clause_search = supabase.table('Chunks').select('header', 'content').in_('document_id', doc_ids).ilike('header', f'%{clause}%').execute()
            if clause_search.data:
                clause_search_results.extend(clause_search.data)
        # print("Clause search results:", clause_search_results)
        # result["clause_search_results"] = clause_search_results
        context = "\n Found clauses: " + "\n".join([f"Header: {result['header']}\nContent: {result['content']}\n\n" for result in clause_search_results])
        

    if len(response.phrases) > 0:
        # Process phrases
        # For example, you might want to use these for additional context in your response
        # make embeddings for phrases
        embeddings = []
        for phrase in response.phrases:
            phrase_embedding = get_embedding(phrase)
            embeddings.append(phrase_embedding)

        # perform semantic search for phrases
        phrase_search_results = []
        print("response.phrases:", response.phrases)
        for i, phrase in enumerate(response.phrases):
            phrase_search = supabase.rpc(
                'match_documents',
                {
                    'query_embedding': embeddings[i],
                    'document_ids': doc_ids,
                    'match_threshold': -0.1,
                    'match_count': 10
                }
            ).execute()
            context += "\n".join([f"Header: {doc['header']}\nContent: {doc['content']}" for doc in phrase_search.data if 'content' in doc and 'header' in doc])
        
        # Format phrase search results
        formatted_phrase_results = []
        for result in phrase_search_results:
            formatted_result = f"Header: {result['header']}\nContent: {result['content']}\n\n"
            formatted_phrase_results.append(formatted_result)
            print("formatted_result:", formatted_result)
        context += "Found related chunks: " + "\n".join(formatted_phrase_results)

    if len(response.keywords) > 0:
        # Process keywords
        # For example, you might want to use these for additional semantic search
        print("Keywords:", response.keywords)

    print("context:", context)
    # Add PO analysis to the context
    context += po_analysis

    # pass context and query to openai using langchain
    llm = ChatOpenAI(model="gpt-4o-2024-08-06")
    # not using structured output because it is too complicated for the model
    prompt_template = ChatPromptTemplate.from_messages([    
        ("system", "You are an AI assistant well versed in aerospace manufacturing contracts. Using the following context, answer the user's question."
         "Only use the context provided to answer the question first and assume the user is asking a question about the documents provided"
         "You may use your own knowledge to answer the question but make sure to use the context provided to answer the question first."
         "Take all the information provided in the context to answer the question."
         "you may have to combine information from multiple chunks to answer the question."
         "you may have to infer information from the context to answer the question."
         "When possible, provide section numbers or clause ids in your answers."
         "Information from purchase order always overrides information from quality documents or terms and conditions."
         "{context}"),
        ("human", "{query}")
    ])
    formatted_prompt = prompt_template.format_messages(context=context, query=query)
    response = llm.invoke(formatted_prompt)
    return {"message": response.content}

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