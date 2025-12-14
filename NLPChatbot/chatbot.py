"""
Production Intent Chatbot Inference Engine.

Author: godmode-dev
License: MIT
"""

import os
import json
import pickle
import random
import numpy as np
from typing import Dict, Any, Tuple, Optional
from train_chatbot import tokenize_and_stem


class IntentChatbot:
    """
    NLP Conversational Assistant with confidence thresholding and fallback mechanisms.

    Parameters
    ----------
    model_dir : str
        Directory containing serialized model, vectorizer, and intents.json.
    min_confidence : float, default=0.25
        Minimum classification probability threshold required to accept prediction.
    """

    def __init__(self, model_dir: Optional[str] = None, min_confidence: float = 0.25) -> None:
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))

        self.model_dir = model_dir
        self.min_confidence = min_confidence

        # Paths
        self.intents_path = os.path.join(model_dir, "intents.json")
        self.model_path = os.path.join(model_dir, "chatbot_model.pkl")
        self.vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        self.tags_path = os.path.join(model_dir, "tags.pkl")

        # Load artifacts
        self.intents = self._load_json(self.intents_path)
        self.model = self._load_pickle(self.model_path)
        self.vectorizer = self._load_pickle(self.vectorizer_path)
        self.tags = self._load_pickle(self.tags_path)

    @staticmethod
    def _load_json(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_pickle(path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_intent(self, user_input: str) -> Tuple[str, float]:
        """
        Predict intent tag and confidence probability score for user message.
        """
        stemmed_input = " ".join(tokenize_and_stem(user_input))
        input_vec = self.vectorizer.transform([stemmed_input]).toarray()

        probabilities = self.model.predict_proba(input_vec)[0]
        max_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[max_idx])
        predicted_tag = self.tags[max_idx]

        return predicted_tag, confidence

    def get_response(self, user_input: str) -> Dict[str, Any]:
        """
        Process user message and return structured response object.
        """
        if not user_input or not user_input.strip():
            return {
                "response": "Please enter a message so I can help you.",
                "tag": "empty_input",
                "confidence": 1.0,
            }

        predicted_tag, confidence = self.predict_intent(user_input)

        if confidence < self.min_confidence:
            return {
                "response": "I'm sorry, I'm not completely sure I understood that. Could you rephrase your question?",
                "tag": "fallback",
                "confidence": confidence,
            }

        # Match intent tag to responses
        for intent in self.intents["intents"]:
            if intent["tag"] == predicted_tag:
                selected_response = random.choice(intent["responses"])
                return {
                    "response": selected_response,
                    "tag": predicted_tag,
                    "confidence": round(confidence, 4),
                }

        return {
            "response": "I apologize, but I couldn't process your inquiry at this moment.",
            "tag": "unknown",
            "confidence": 0.0,
        }


def start_terminal_chat():
    bot = IntentChatbot()
    print("=" * 60)
    print("  🤖 NLP Intent Chatbot initialized!")
    print("  Type 'quit' or 'exit' to stop the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_msg = input("You > ")
            if user_msg.lower().strip() in ["quit", "exit"]:
                print("Bot > Goodbye! Have a great day.")
                break

            result = bot.get_response(user_msg)
            print(f"Bot > {result['response']} (Tag: {result['tag']}, Confidence: {result['confidence']:.2f})\n")
        except KeyboardInterrupt:
            print("\nBot > Session terminated.")
            break


if __name__ == "__main__":
    start_terminal_chat()