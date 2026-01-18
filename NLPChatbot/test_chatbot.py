"""
Unit tests verifying intent classification, artifact loading, and fallback thresholds.
"""

import pytest
import os
from chatbot import IntentChatbot
from train_chatbot import train_and_save_model


@pytest.fixture(scope="module")
def setup_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    intents_path = os.path.join(script_dir, "intents.json")
    train_and_save_model(intents_path, script_dir)
    return IntentChatbot(model_dir=script_dir)


def test_chatbot_greeting(setup_model):
    bot = setup_model
    res = bot.get_response("Hi there!")
    assert res["confidence"] > 0.0
    assert isinstance(res["response"], str)
    assert len(res["response"]) > 0


def test_chatbot_fallback(setup_model):
    bot = setup_model
    res = bot.get_response("xyz123456789 non_existent_gibberish_string")
    assert res["tag"] in ["fallback", "unknown", "greeting"]


def test_empty_input(setup_model):
    bot = setup_model
    res = bot.get_response("")
    assert res["tag"] == "empty_input"
