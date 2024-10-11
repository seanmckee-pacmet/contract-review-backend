from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import os
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import traceback
import fitz  # PyMuPDF
from pydantic import BaseModel
from typing import List
from langchain_openai import ChatOpenAI
load_dotenv()
# new
# Set up LlamaParse
parser = LlamaParse(
    result_type="markdown",
    parsing_instruction="You are a helpful assistant that converts PDF documents to markdown. Focus on finding the correct overarching heading for each section.",
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
)

def get_plain_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return ""

def get_plain_text_from_tiff(tiff_path):
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"The file {tiff_path} does not exist.")
    
    try:
        full_text = ""
        with Image.open(tiff_path) as img:
            for i in range(img.n_frames):
                img.seek(i)
                text = pytesseract.image_to_string(img)
                full_text += f"{text}\n\n"
        return full_text
    except Exception as e:
        print(f"Error processing {tiff_path}: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return ""

def parse_pdf_to_markdown(pdf_path):
    print("Parsing PDF to Markdown")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    plain_text = get_plain_text_from_pdf(pdf_path)
    print("Got plain text")
    headers = get_headers(plain_text)
    print("Got headers")
    # create prompt (multiline string)
    parsing_instruction = f"""
    You are a helpful assistant that converts PDF documents to markdown. 
    Focus on finding the correct overarching heading for each section.
    The headers to focus on are: {headers}
    """

    # create instance of llamaparse with custom prompt 
    parser = LlamaParse(
        result_type="markdown",
        parsing_instruction=parsing_instruction,
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    )
    print("Created parser")
    try:
        file_extractor = {".pdf": parser}
        reader = SimpleDirectoryReader(input_files=[pdf_path], file_extractor=file_extractor)
        print("Created reader")
        documents = reader.load_data()
        print("Loaded documents")

        
        if documents:
            markdown_content = []
            for doc in documents:
                if hasattr(doc, 'text'):
                    markdown_content.append(doc.text)
                elif hasattr(doc, 'page_content'):
                    markdown_content.append(doc.page_content)
                else:
                    print(f"Warning: Document doesn't have 'text' or 'page_content' attribute: {doc}")

            return "\n\n".join(markdown_content)
        else:
            print(f"No documents were parsed from {pdf_path}")
            return ""
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return ""

def parse_tiff_to_markdown(tiff_path):
    print("Parsing TIFF to Markdown")
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"The file {tiff_path} does not exist.")
    
    full_text = ""
    with Image.open(tiff_path) as img:
        for i in range(img.n_frames):
            img.seek(i)
            text = pytesseract.image_to_string(img)
            full_text += f"Page {i+1}:\n{text}\n\n"

    headers = get_headers(tiff_path)
    # multiline string prompt
    parsing_instruction = f"""
    You are a helpful assistant that converts text documents to markdown. 
    Focus on finding the correct overarching heading for each section.
    The headers to focus on are: {headers}
    """
    parser = LlamaParse(
        result_type="markdown",
        parsing_instruction=parsing_instruction,
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    )
    
    # Use LlamaParse to convert the OCR text to markdown
    file_extractor = {".txt": parser}
    with open("temp.txt", "w", encoding="utf-8") as temp_file:
        temp_file.write(full_text)
    documents = SimpleDirectoryReader(input_files=["temp.txt"], file_extractor=file_extractor).load_data()
    os.remove("temp.txt")
    
    if documents:
        return "\n\n".join(doc.text for doc in documents)
    return ""

class Headers(BaseModel):
    headers: List[str]

def get_headers(plain_text):

    # get formatted text
    
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = model.with_structured_output(Headers)
    print("Created structured LLM")
    headers = structured_llm.invoke(plain_text)
    print("Got headers")
    return headers

def parse_document(doc_path):
    print("Parsing document")
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"The file {doc_path} does not exist.")

    if doc_path.lower().endswith(".pdf"):
        print("Parsing PDF")
        return parse_pdf_to_markdown(doc_path)
    elif doc_path.lower().endswith((".tiff", ".tif")):
        print("Parsing TIFF")
        return parse_tiff_to_markdown(doc_path)
    else:
        raise ValueError(f"Unsupported file type: {doc_path}")

def get_formatted_text(file_path):
    response = parse_document(file_path)
    return response

def get_plain_text(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    if file_path.lower().endswith(".pdf"):
        return get_plain_text_from_pdf(file_path)
    elif file_path.lower().endswith((".tiff", ".tif")):
        return get_plain_text_from_tiff(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")