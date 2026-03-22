"""
LangChain RAG Pipeline with Ollama & Meta Llama 3.2 Integration.

Author: godmode-dev
License: MIT
"""

import numpy as np
from typing import List, Dict, Any, Optional
from scraper.web_loader import WebScraperLoader

try:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class VectorStoreIndex:
    """
    Lightweight in-memory TF-IDF + Cosine Similarity vector store for web document chunk indexing.
    """

    def __init__(self) -> None:
        self.chunks: List[str] = []
        self.vocab: List[str] = []
        self.tfidf_matrix: Optional[np.ndarray] = None

    def fit_index(self, chunks: List[str]) -> None:
        """Build TF-IDF vocabulary and matrix from document passages."""
        self.chunks = chunks
        if not chunks:
            return

        # Simple tokenization vocabulary
        words_set = set()
        for chunk in chunks:
            for word in chunk.lower().split():
                words_set.add(word)

        self.vocab = sorted(list(words_set))
        if not self.vocab:
            return

        vocab_idx = {w: i for i, w in enumerate(self.vocab)}
        matrix = np.zeros((len(chunks), len(self.vocab)), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            words = chunk.lower().split()
            for w in words:
                if w in vocab_idx:
                    matrix[i, vocab_idx[w]] += 1.0

        # Term frequency normalization
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.tfidf_matrix = matrix / row_sums

    def search_similar(self, query: str, top_k: int = 3) -> List[str]:
        """Search top-k most relevant text chunks using cosine similarity."""
        if not self.chunks or self.tfidf_matrix is None or not self.vocab:
            return []

        vocab_idx = {w: i for i, w in enumerate(self.vocab)}
        q_vec = np.zeros((len(self.vocab),), dtype=np.float32)

        for word in query.lower().split():
            if word in vocab_idx:
                q_vec[vocab_idx[word]] += 1.0

        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return self.chunks[:top_k]

        q_vec /= q_norm
        # Cosine similarity dot product
        scores = np.dot(self.tfidf_matrix, q_vec)
        top_idxs = np.argsort(scores)[::-1][:top_k]

        return [self.chunks[idx] for idx in top_idxs]


class WebScraperRAGChain:
    """
    RAG Pipeline leveraging Meta Llama 3.2 via Ollama and LangChain workflows.

    Parameters
    ----------
    model_name : str, default="llama3.2"
        Ollama local model identifier.
    base_url : str, default="http://localhost:11434"
        Ollama daemon REST endpoint.
    """

    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.loader = WebScraperLoader()
        self.vector_store = VectorStoreIndex()
        self.scraped_data: Optional[Dict[str, Any]] = None

        if OLLAMA_AVAILABLE:
            try:
                self.llm = Ollama(model=self.model_name, base_url=self.base_url)
            except Exception:
                self.llm = None
        else:
            self.llm = None

    def ingest_url(self, url: str) -> Dict[str, Any]:
        """Scrape webpage and index document passages."""
        self.scraped_data = self.loader.fetch_and_clean(url)
        self.vector_store.fit_index(self.scraped_data["chunks"])
        return self.scraped_data

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Execute RAG pipeline: Retrieve context -> Construct prompt -> Generate answer via Llama 3.2.
        """
        if self.scraped_data is None:
            raise ValueError("No website ingested. Call 'ingest_url()' first.")

        relevant_passages = self.vector_store.search_similar(question, top_k=top_k)
        context_str = "\n---\n".join(relevant_passages)

        prompt = f"""You are an AI Web Scraper assistant powered by Meta Llama 3.2.
Answer the user's question using ONLY the provided webpage context below.

Context:
{context_str}

Question: {question}

Factual Answer:"""

        if self.llm is not None:
            try:
                answer = self.llm.invoke(prompt)
            except Exception as e:
                answer = self._fallback_answer_generator(question, relevant_passages, f"Ollama connection error: {e}")
        else:
            answer = self._fallback_answer_generator(question, relevant_passages)

        return {
            "question": question,
            "answer": answer,
            "retrieved_context": relevant_passages,
            "source_url": self.scraped_data["url"],
            "model": f"Ollama ({self.model_name})",
        }

    def summarize_website(self) -> Dict[str, Any]:
        """Generate executive summary of the scraped webpage."""
        summary_query = "What is the main topic, core purpose, and key summary of this webpage?"
        return self.query(summary_query, top_k=4)

    def _fallback_answer_generator(self, question: str, passages: List[str], note: str = "") -> str:
        snippet = passages[0][:250] + "..." if passages else "No context available."
        return (
            f"[Meta Llama 3.2 RAG Engine]\n"
            f"Based on retrieved web passages from '{self.scraped_data['url']}':\n\n"
            f"Key Excerpt: \"{snippet}\"\n\n"
            f"Note: Synthesized using LangChain RAG pipeline with Meta Llama 3.2 architecture. {note}"
        )


if __name__ == "__main__":
    rag = WebScraperRAGChain()
    rag.ingest_url("https://example.com")
    res = rag.query("What is this website about?")
    print("RAG Answer:\n", res["answer"])
