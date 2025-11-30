"""
Sources :
https://www.ibm.com/think/topics/vector-database
https://www.psycopg.org/psycopg3/docs/basic/usage.html
https://www.crunchydata.com/blog/hnsw-indexes-with-postgres-and-pgvector
https://github.com/pgvector/pgvector?tab=readme-ov-file#installation ** Vector Index Settings


https://docs.python.org/3/howto/logging.html -> Logging

"""

import os
import psycopg
from psycopg.rows import dict_row
import json
import logging
import colorlog
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Configure Logging with Colors (For src.db)
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False  # Prevent duplicate logs

load_dotenv()

db_user = os.getenv("POSTGRES_USER")
db_password = os.getenv("POSTGRES_PASSWORD")
db_name = os.getenv("POSTGRES_DB")
db_host = os.getenv("POSTGRES_HOST", "localhost")
db_port = os.getenv("POSTGRES_PORT", "5432")

# GPT**
if not all([db_user, db_password, db_name]):
    logger.error("Database info is missing. Please check your .env file.")
    raise ValueError("Database info is missing.")

# Connection settings
DB_PARAMS = {
    "dbname": db_name,
    "user": db_user,
    "password": db_password,
    "host": db_host,
    "port": db_port
}

# Vector index settings
# Supported index types: "hnsw", "ivfflat"
VECTOR_INDEX_TYPE = "hnsw"

# Supported operator classes and their corresponding distance functions:
# - vector_l2_ops      -> <->  (L2 distance)
# - vector_ip_ops      -> <#>  (negative inner product)
# - vector_cosine_ops  -> <=>  (cosine distance) [CURRENT]
# - vector_l1_ops      -> <+>  (L1 distance)
# - vector_hamming_ops -> <~>  (Hamming distance, binary vectors)
# - vector_jaccard_ops -> <%>  (Jaccard distance, binary vectors)
VECTOR_OPERATOR_CLASS = "vector_cosine_ops"

# https://www.psycopg.org/psycopg3/docs/basic/usage.html
def get_db_connection():
    return psycopg.connect(**DB_PARAMS, row_factory=dict_row)

def init_db():
    """Creates database tables and enables pgvector."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension - https://github.com/pgvector/pgvector?tab=readme-ov-file#installation
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    total_pages INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Chunks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector(1536), 
                    page_number INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Indexes - GPT**
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_embedding 
                ON document_chunks USING {VECTOR_INDEX_TYPE} (embedding {VECTOR_OPERATOR_CLASS});
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_category ON documents(category);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON document_chunks(document_id);")

            conn.commit()
            logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error while creating database: {e}")
        conn.rollback()
    finally:
        conn.close()

# GPT**
def extract_category_from_path(file_path: str) -> str:
    try:
        normalized = file_path.replace("\\", "/")
        folder_path = "/".join(normalized.split("/")[:-1])
        category = folder_path.split("/")[-1] if folder_path else "uncategorized"
        return category if category else "uncategorized"
    except:
        return "uncategorized"

def insert_document(file_path: str, filename: str, total_pages: int):
    """Adds a document into the table. Returns ID. If exists, returns existing ID."""
    conn = get_db_connection()
    category = extract_category_from_path(file_path)
    doc_id = None

    try:
        with conn.cursor() as cur:
            # Check if document already exists
            cur.execute("SELECT id FROM documents WHERE file_path = %s", (file_path,))
            existing = cur.fetchone()

            if existing:
                doc_id = existing["id"]
            else:
                cur.execute("""
                    INSERT INTO documents (filename, file_path, category, total_pages)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """, (filename, file_path, category, total_pages))
                doc_id = cur.fetchone()["id"]

            conn.commit()
    except Exception as e:
        logger.error(f"Error inserting document: {e}")
        conn.rollback()
    finally:
        conn.close()

    return doc_id

def insert_chunks(doc_id: int, chunks):
    """Inserts chunk data for a specific document."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for ch in chunks:
                cur.execute("""
                    INSERT INTO document_chunks 
                    (document_id, chunk_id, text, embedding, page_number, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    doc_id,
                    ch["chunk_id"],
                    ch["text"],
                    ch["embedding"],
                    ch["page_number"],
                    json.dumps(ch["metadata"])
                ))

            conn.commit()
            logger.info(f"{len(chunks)} chunks inserted.")
    except Exception as e:
        logger.error(f"Error inserting chunks: {e}")
        conn.rollback()
    finally:
        conn.close()

# https://stackoverflow.com/questions/2254999/similarity-function-in-postgres-with-pg-trgm
def search_similar_chunks(query_embedding: List[float], top_k: int = 5, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Searches for the most similar chunks to the query embedding.
    
    Args:
        query_embedding: The vector representation of the user's query.
        top_k: Number of results to return.
        category_filter: Optional category name to filter results (e.g., 'sql').
        
    Returns:
        List of dictionaries containing text, metadata, similarity score, etc.
    """
    conn = get_db_connection()
    results = []
    try:
        with conn.cursor() as cur:

            sql = """
                SELECT 
                    c.text, 
                    c.metadata, 
                    c.page_number,
                    d.filename, 
                    d.category,
                    1 - (c.embedding <=> %s::vector) as similarity
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
            """
            params = [str(query_embedding)] # Pass as string for casting to vector
            
            # Add category filter if provided
            if category_filter:
                sql += " WHERE d.category = %s"
                params.append(category_filter)
                
            # Order by similarity (closest distance first)
            sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
            params.append(str(query_embedding))
            params.append(top_k)
            
            cur.execute(sql, params)
            results = cur.fetchall()
            
    except Exception as e:
        logger.error(f"Error searching similar chunks: {e}")
    finally:
        conn.close()
        
    return results
