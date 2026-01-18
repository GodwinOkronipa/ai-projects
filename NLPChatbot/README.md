# 💬 NLP Intent Chatbot Microservice

A production-style Natural Language Processing intent classification engine and REST API.

---

## 🏗️ Architecture & Features

- **Preprocessing & Stemming**: Custom NLTK tokenization and Porter Stemmer pipeline.
- **Vectorizer & Classifier**: TF-IDF n-gram vectorization coupled with regularized Logistic Regression.
- **Confidence Thresholding**: Automatic fallback strategy when prediction confidence falls below threshold.
- **REST Microservice**: **FastAPI** service serving `/chat`, `/health`, and `/intents` endpoints.

---

## 🚀 Quickstart

### 1. Train the Model
```bash
python train_chatbot.py
```

### 2. Run Interactive Terminal Bot
```bash
python chatbot.py
```

### 3. Launch REST API Server
```bash
python api.py
# Open Swagger UI at http://localhost:8000/docs
```
