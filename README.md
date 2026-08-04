# 🧠 AI & Machine Learning Engineering Workspace

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20LangChain%20%7C%20Scikit--Learn%20%7C%20FastAPI%20%7C%20Streamlit-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

Welcome to my Machine Learning and Artificial Intelligence portfolio repository. This repository demonstrates core machine learning algorithms implemented from mathematical first principles, NLP intent classification systems, remote sensing PyTorch deep learning models, an enterprise AI retail analytics platform tailored for Ghanaian commerce (**Storeflow AI**), and an AI-powered Web Scrapper using **LangChain RAG** and **Meta Llama 3.2**.

---

## 🌟 Portfolio Overview

| Project Subfolder | Focus Area | Key Technologies & Architecture | Target Application |
| :--- | :--- | :--- | :--- |
| 🏬 [**`storeflow-retail-ai`**](./storeflow-retail-ai) | **Flagship AI Platform** | XGBoost, Scikit-Learn, Streamlit, FastAPI, Plotly, GHS Financial Metrics | Retail Demand Forecasting, Stockout Risk AI, Customer RFM Segmentation ([Storeflow Live Link](https://flywheel-storeflow.pages.dev)) |
| 🌐 [**`ai-web-scraper-rag`**](./ai-web-scraper-rag) | **RAG & Information Retrieval** | Meta Llama 3.2, Ollama, LangChain, BeautifulSoup4, Streamlit, FastAPI | AI-Powered Web Scraping, Semantic Vector QA, Automated Executive Summarization |
| 🧮 [**`ml-algorithms`**](./ml-algorithms) | **Algorithms from Scratch** | NumPy, Vectorization, Matrix Calculus, Matplotlib | First-principles implementation of Linear/Logistic Regression, Decision Trees, K-Means, k-NN, PCA |
| 💬 [**`NLPChatbot`**](./NLPChatbot) | **NLP & Conversational AI** | TF-IDF, Scikit-Learn, FastAPI, Intent Classification | Intelligent customer support bot with confidence thresholds and REST API |
| 🛰️ [**`RS-Flood-Mapper`**](./RS-Flood-Mapper) | **Deep Learning & Earth Observation** | PyTorch, U-Net, Multi-Spectral Remote Sensing, Sentinel-1/2 | Automated flood extent extraction and disaster response analytics |

---

## 🚀 Projects Deep-Dive

### 1. 🏬 Storeflow AI — Ghanaian Retail Analytics & Demand Intelligence (`/storeflow-retail-ai`)
- **Description**: An AI-powered decision support system built to optimize retail store operations in Ghana (Accra, Kumasi, Takoradi). Integrates time-series sales forecasting (GHS revenue & unit demand), customer Recency-Frequency-Monetary (RFM) clustering, and automated stockout risk prediction.
- **Demo & Deployment**: Built with **Streamlit** and **FastAPI**. Directly bridges to the [Flywheel Storeflow Platform](https://flywheel-storeflow.pages.dev).

### 2. 🌐 AI Web Scrapper using LangChain RAG & Meta Llama 3.2 (`/ai-web-scraper-rag`)
- **Description**: Built an AI-powered web scraper using Meta AI's Llama 3.2 model and LangChain's AI agent building toolkit that aggregates and summarizes website data, demonstrating the ability to apply ML to real-world information retrieval and analysis.
- **Features**: DOM noise cleaning, dense passage vector index, Ollama LLM integration, Streamlit Web App, and FastAPI endpoint.

### 3. 🧮 ML Algorithms from Scratch (`/ml-algorithms`)
- **Description**: Modular, production-style implementations of fundamental ML algorithms built using pure NumPy and linear algebra without relying on high-level estimator wrappers.

### 4. 💬 Intelligent Intent Chatbot (`/NLPChatbot`)
- **Description**: Production-ready NLP conversational assistant supporting JSON intent configuration, TF-IDF vectorization, confidence score thresholds, fallback loops, and FastAPI integration.

### 5. 🛰️ Remote Sensing Flood Mapper (`/RS-Flood-Mapper`)
- **Description**: Semantic segmentation deep learning pipeline in **PyTorch** using a U-Net architecture to map flood extents from synthetic/real Sentinel-1 SAR and Sentinel-2 optical imagery (NDWI).

---

## 🛠️ Quickstart & Environment Setup

```bash
git clone https://github.com/your-username/ai-projects.git
cd ai-projects

# Activate virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Run test suites across all projects
python -m pytest ml-algorithms/tests/
python -m pytest storeflow-retail-ai/tests/
python -m pytest ai-web-scraper-rag/tests/
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
