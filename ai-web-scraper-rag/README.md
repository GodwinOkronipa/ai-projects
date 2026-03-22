# 🌐 AI Web Scrapper using LangChain RAG, Ollama & Meta Llama 3.2

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-LangChain%20%7C%20Ollama%20%7C%20Meta%20Llama%203.2%20%7C%20FastAPI-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Project Highlight**: Built an AI-powered web scraper using **Meta AI's Llama 3.2** model and **LangChain's** AI agent building toolkit that aggregates and summarizes website data, demonstrating the ability to apply ML to real-world information retrieval and analysis.

---

## 🏗️ Technical Architecture & Features

- **🌐 Automated Web Scraping & DOM Noise Filter**:
  - Fetches target HTML pages via `requests` and `BeautifulSoup4`, removing boilerplate tags (`script`, `style`, `nav`, `footer`).
  - Generates overlapping semantic text passages suitable for vector embedding.

- **🔍 Dense Vector Retrieval Index**:
  - In-memory TF-IDF and Cosine Similarity vector store performing dense passage retrieval.

- **🤖 LangChain & Ollama Meta Llama 3.2 RAG Pipeline**:
  - Constructs Retrieval-Augmented Generation (RAG) prompts passing retrieved web passages directly to local **Meta Llama 3.2** instances running via **Ollama**.
  - Eliminates hallucinations by grounding all answers directly in scraped site context.

- **🖥️ Interactive Streamlit UI**:
  - Allows users to paste any URL, scrape content, ask natural language questions, generate executive summaries, and inspect vector chunks.

- **⚡ FastAPI Microservice**:
  - Exposes REST endpoints (`/scrape`, `/query`, `/summarize`) for integration with external agents.

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
cd ai-web-scraper-rag
pip install -r requirements.txt
```

### 2. Launch Streamlit Web UI
```bash
streamlit run app.py
```

### 3. Launch FastAPI Microservice
```bash
python api.py
# Open Swagger docs at http://localhost:8002/docs
```

### 4. Run Pytest Suite
```bash
pytest tests/
```
