import os
import re
import logging
from typing import List
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from openai import OpenAI
from collections import defaultdict

# Import DB functions
from db import init_db, insert_document, insert_chunks


#https://docs.python.org/3/howto/logging.html
#https://www.geeksforgeeks.org/python/how-to-log-queries-in-postgresql-using-python/
# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()] # GPT
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1500  
CHUNK_OVERLAP = 200 

client = OpenAI(api_key=OPENAI_API_KEY)

# GPT**
def clean_text(text: str) -> str:
    """
    Cleans the text by removing specific patterns and excessive whitespace.
    """
    # Example patterns to remove (customize as needed)
    patterns_to_remove = [
        r"Dr\s+([A-Z]\.?\s*[A-Za-z]+\s*){1,2}RDB\s+\d+", # Remove header/footer noise
        "Dr Renaud Richardet"
    ]
    
    cleaned_text = text
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, "", cleaned_text)
    
    # Normalize whitespace (replace multiple spaces/newlines with single space)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def load_documents(directory_path: str) -> List[Document]:
    """
    Loads documents from the specified directory using SimpleDirectoryReader.
    """
    logger.info(f"Loading documents from: {directory_path}")
    
    reader = SimpleDirectoryReader(directory_path, recursive=True)
    documents = reader.load_data()
    
    logger.info(f"Loaded {len(documents)} raw documents (pages).")
    return documents

def process_documents_to_chunks(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[TextNode]:
    """
    Cleans documents and splits them into chunks.
    """
    # Clean documents - Create new Document objects with cleaned text
    cleaned_docs = []
    for doc in documents:
        cleaned_text = clean_text(doc.text)
        # Create a new Document with cleaned text and preserve metadata
        new_doc = Document(text=cleaned_text, metadata=doc.metadata)
        cleaned_docs.append(new_doc)
        
    # Chunking
    logger.info(f"Splitting documents into chunks (Size: {chunk_size}, Overlap: {chunk_overlap}).")
    
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(cleaned_docs)
    
    logger.info(f"Created {len(nodes)} chunks.")
    
    return nodes

def generate_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using OpenAI's Batch API.
    """
    try:
        # Filter out empty texts to avoid API errors # GPT **
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []
            
        response = client.embeddings.create(
            model=model,
            input=valid_texts
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []

def ingest_data(directory_path: str):
    """
    Main ingestion function: Load -> Clean -> Chunk -> Embed -> Save to DB.
    """
    # 1. Initialize DB
    init_db()
    
    # 2. Load Documents
    raw_docs = load_documents(directory_path)
    
    # 3. Process (Clean & Chunk)
    nodes = process_documents_to_chunks(raw_docs)
    
    # 4. Group nodes by file for efficient DB insertion
    nodes_by_file = defaultdict(list)
    for node in nodes:
        file_path = node.metadata.get('file_path')
        if file_path:
            nodes_by_file[file_path].append(node)
            
    logger.info(f"Starting database insertion for {len(nodes_by_file)} files...")
    
    # 5. Process each file
    for file_path, file_nodes in nodes_by_file.items():
        try:
            # A. Insert Document Metadata
            first_node = file_nodes[0]
            filename = first_node.metadata.get('file_name', 'unknown')
            
            unique_pages = set()
            for n in file_nodes:
                page_label = n.metadata.get('page_label')
                if page_label:
                    unique_pages.add(page_label)
            total_pages = len(unique_pages) if unique_pages else 0
            
            # insert_document fonksiyonu file_path'ten kategoriyi otomatik çıkarır
            doc_id = insert_document(file_path, filename, total_pages)
            
            # B. Generate Embeddings for all chunks in this file
            texts = [node.text for node in file_nodes]
            embeddings = generate_embeddings(texts)
            
            if len(embeddings) != len(texts):
                logger.warning(f"Mismatch in embedding count for {filename}. Skipping chunks.")
                continue
                
            # C. Prepare Chunk Data
            chunks_data = []
            for i, node in enumerate(file_nodes):
                chunks_data.append({
                    'chunk_id': node.id_,
                    'text': node.text,
                    'embedding': embeddings[i],
                    'page_number': int(node.metadata.get('page_label', 0)) if str(node.metadata.get('page_label', '0')).isdigit() else 0,
                    'metadata': node.metadata
                })
            
            # D. Insert Chunks
            insert_chunks(doc_id, chunks_data)
            logger.info(f"Processed: {filename} ({len(chunks_data)} chunks)")
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    if os.path.exists(DATA_DIR):
        ingest_data(DATA_DIR)
    else:
        logger.error(f"Data directory not found: {DATA_DIR}")
        logger.error("Please update DATA_DIR in ingestion.py")
