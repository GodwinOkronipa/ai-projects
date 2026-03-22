"""
Unit tests for AI Web Scrapper loader and RAG vector search engine.
"""

import sys
import os
import pytest

# Ensure parent module directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.web_loader import WebScraperLoader
from rag.rag_chain import WebScraperRAGChain, VectorStoreIndex


def test_web_loader_chunking():
    loader = WebScraperLoader()
    text = "Word " * 500
    chunks = loader.chunk_text(text, max_chars=100, overlap=20)
    
    assert len(chunks) > 1
    assert isinstance(chunks, list)


def test_vector_store_search():
    store = VectorStoreIndex()
    passages = [
        "Python is a high level programming language for data science.",
        "Machine learning models predict future outcomes.",
        "Web scraping extracts structured HTML content."
    ]
    store.fit_index(passages)
    
    results = store.search_similar("machine learning prediction", top_k=1)
    assert len(results) == 1
    assert "Machine learning" in results[0]


def test_rag_chain_pipeline():
    rag = WebScraperRAGChain()
    data = rag.ingest_url("https://example.com")
    
    assert "num_chunks" in data
    assert data["num_chunks"] > 0
    
    res = rag.query("What is this website about?")
    assert "answer" in res
    assert "retrieved_context" in res
