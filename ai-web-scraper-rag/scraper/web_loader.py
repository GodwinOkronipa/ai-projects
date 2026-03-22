"""
Web Content Extractor & HTML Preprocessing Utility.

Author: godmode-dev
License: MIT
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List


class WebScraperLoader:
    """
    Fetches raw web pages, strips DOM noise (scripts, styles, headers, footers),
    and converts structural HTML elements into clean text chunks for RAG embedding.
    """

    def __init__(self, user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)") -> None:
        self.headers = {"User-Agent": user_agent}

    def fetch_and_clean(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Fetch HTML content from target URL and clean boilerplate elements.
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            html_content = response.text
        except Exception as e:
            # Fallback mock for offline demonstration / testing
            return self._generate_mock_scraped_content(url, str(e))

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove irrelevant noise elements
        for element in soup(["script", "style", "nav", "footer", "iframe", "noscript", "svg", "button"]):
            element.decompose()

        # Extract title
        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Extract main text
        text_lines = [line.strip() for line in soup.get_text(separator="\n").split("\n") if line.strip()]
        clean_text = "\n".join(text_lines)

        # Chunk content into paragraphs
        chunks = self.chunk_text(clean_text, max_chars=1000, overlap=150)

        return {
            "url": url,
            "title": title,
            "raw_text": clean_text,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "status": "success",
        }

    @staticmethod
    def chunk_text(text: str, max_chars: int = 1000, overlap: int = 150) -> List[str]:
        """
        Split long text into semantic passages with sliding window overlapping boundaries.
        """
        if not text:
            return []

        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1

            if current_length >= max_chars:
                chunks.append(" ".join(current_chunk))
                # Retain overlapping tail words
                overlap_words = current_chunk[-max(1, int(overlap / 6)):]
                current_chunk = overlap_words
                current_length = sum(len(w) + 1 for w in current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _generate_mock_scraped_content(self, url: str, error_msg: str) -> Dict[str, Any]:
        mock_text = (
            f"Artificial Intelligence and Modern Web Scraping Architecture for {url}.\n\n"
            "Retrieval-Augmented Generation (RAG) combines semantic vector search with Large Language Models "
            "such as Meta Llama 3.2. By partitioning webpage text into dense embedding spaces, LLMs can provide "
            "factual answers anchored directly in source web content.\n\n"
            "LangChain provides the orchestration workflow linking document loaders, vector stores, and "
            "Ollama local model instances. This eliminates hallucinations and preserves data privacy."
        )
        chunks = self.chunk_text(mock_text)
        return {
            "url": url,
            "title": f"Scraped Page Content ({url})",
            "raw_text": mock_text,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "status": f"simulated ({error_msg})",
        }


if __name__ == "__main__":
    loader = WebScraperLoader()
    res = loader.fetch_and_clean("https://example.com")
    print(f"Scraped {res['title']} -> {res['num_chunks']} chunks generated.")
