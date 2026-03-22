"""
FastAPI REST microservice for AI Web Scrapper using LangChain RAG & Meta Llama 3.2.

Author: godmode-dev
License: MIT
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uvicorn

from rag.rag_chain import WebScraperRAGChain

app = FastAPI(
    title="AI Web Scrapper RAG API",
    description="REST microservice serving web page scraping, semantic vector search, and RAG Q&A using Meta Llama 3.2.",
    version="1.0.0"
)

rag = WebScraperRAGChain()


class IngestRequest(BaseModel):
    url: str = Field(..., example="https://en.wikipedia.org/wiki/Artificial_intelligence")


class QueryRequest(BaseModel):
    question: str = Field(..., example="What is artificial intelligence?")
    top_k: Optional[int] = Field(default=3, ge=1, le=10)


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "service": "AI Web Scrapper RAG Engine"}


@app.post("/scrape", tags=["Scraper"])
def scrape_and_ingest(req: IngestRequest):
    try:
        data = rag.ingest_url(req.url)
        return {
            "title": data["title"],
            "url": data["url"],
            "chunks_generated": data["num_chunks"],
            "status": data["status"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", tags=["RAG Chain"])
def query_scraped_content(req: QueryRequest):
    try:
        res = rag.query(req.question, top_k=req.top_k or 3)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/summarize", tags=["Summarization"])
def summarize_scraped_page():
    try:
        return rag.summarize_website()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
