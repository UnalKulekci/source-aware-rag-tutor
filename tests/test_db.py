"""
https://realpython.com/python-mock-library/
https://docs.python.org/3/library/unittest.mock.html
https://docs.pytest.org/en/7.1.x/how-to/monkeypatch.html


"""


import pytest
from unittest.mock import MagicMock, patch
from src.db import (
    extract_category_from_path,
    init_db,
    insert_document,
    search_similar_chunks,
    DB_PARAMS
)

# Unit Tests

def test_extract_category_standard_path():
    assert extract_category_from_path("data/raw/sql/lecture1.pdf") == "sql"

def test_extract_category_windows_path():
    assert extract_category_from_path("data\\raw\\python\\intro.pdf") == "python"

def test_extract_category_root_file():
    # Kök dizin veya tek dosya durumunda
    assert extract_category_from_path("manual.pdf") == "uncategorized"

def test_extract_category_empty_path():
    assert extract_category_from_path("") == "uncategorized"
    assert extract_category_from_path(None) == "uncategorized"

# Mock Integration Tests

@patch("src.db.psycopg.connect")
def test_init_db_success(mock_connect):
   
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    init_db()
    
    assert mock_cursor.execute.call_count >= 4  
    mock_conn.commit.assert_called_once() 
    mock_conn.close.assert_called_once() 

@patch("src.db.psycopg.connect")
def test_insert_document_new(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    

    mock_cursor.fetchone.side_effect = [None, {"id": 10}]
    
    doc_id = insert_document("data/sql/intro.pdf", "intro.pdf", 5)
    
    assert doc_id == 10
    assert "INSERT INTO documents" in mock_cursor.execute.call_args_list[1][0][0]

@patch("src.db.psycopg.connect")
def test_insert_document_existing(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    

    mock_cursor.fetchone.return_value = {"id": 5}
    
    doc_id = insert_document("data/sql/intro.pdf", "intro.pdf", 5)
    
    assert doc_id == 5
    assert mock_cursor.execute.call_count == 1
    assert "SELECT id FROM documents" in mock_cursor.execute.call_args[0][0]

@patch("src.db.psycopg.connect")
def test_search_similar_chunks(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    fake_results = [
        {"text": "SQL is great", "similarity": 0.95, "filename": "sql.pdf"},
        {"text": "Select * from", "similarity": 0.88, "filename": "sql.pdf"}
    ]
    mock_cursor.fetchall.return_value = fake_results
    
    query_vec = [0.1, 0.2, 0.3]
    results = search_similar_chunks(query_vec, top_k=2)
    
    assert len(results) == 2
    assert results[0]["text"] == "SQL is great"
    mock_cursor.execute.assert_called()
