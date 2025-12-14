"""
Training pipeline for TF-IDF + LogisticRegression NLP Intent Classifier.

Author: godmode-dev
License: MIT
"""

import os
import json
import pickle
import numpy as np
import nltk
from typing import Tuple, List, Dict
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download required NLTK tokenizers safely
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

stemmer = PorterStemmer()


def tokenize_and_stem(text_or_tokens) -> List[str]:
    """Tokenize and stem text string or token list."""
    ignore_words = ["?", "!", ".", ",", ";", ":"]
    if isinstance(text_or_tokens, str):
        tokens = nltk.word_tokenize(text_or_tokens)
    else:
        tokens = text_or_tokens

    return [stemmer.stem(w.lower()) for w in tokens if w not in ignore_words]


def load_intents(json_path: str) -> Dict:
    """Load and validate intents JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Intents file not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataset(intents_dict: Dict) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer, List[str]]:
    """Preprocess patterns, build TF-IDF matrix and target vector."""
    corpus = []
    labels = []
    tags = []

    for intent in intents_dict["intents"]:
        tag = intent["tag"]
        if tag not in tags:
            tags.append(tag)

        for pattern in intent["patterns"]:
            stemmed_pattern = " ".join(tokenize_and_stem(pattern))
            corpus.append(stemmed_pattern)
            labels.append(tag)

    tags = sorted(tags)
    y_train = np.array([tags.index(tag) for tag in labels])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train = vectorizer.fit_transform(corpus).toarray()

    return X_train, y_train, vectorizer, tags


def train_and_save_model(
    intents_path: str = "intents.json",
    model_dir: str = "."
) -> Tuple[LogisticRegression, TfidfVectorizer, List[str]]:
    """Train intent classifier and dump artifacts."""
    intents_dict = load_intents(intents_path)
    X_train, y_train, vectorizer, tags = prepare_dataset(intents_dict)

    print(f"Training Intent Classifier on {len(X_train)} patterns across {len(tags)} intent tags...")
    model = LogisticRegression(solver="liblinear", C=1.5, random_state=42)
    model.fit(X_train, y_train)

    # Save artifacts
    model_path = os.path.join(model_dir, "chatbot_model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    tags_path = os.path.join(model_dir, "tags.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(tags_path, "wb") as f:
        pickle.dump(tags, f)

    print("Model artifacts successfully saved!")
    return model, vectorizer, tags


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    intents_file = os.path.join(script_dir, "intents.json")
    train_and_save_model(intents_file, script_dir)