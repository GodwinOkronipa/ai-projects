"""
FastAPI REST API wrapper for the NLP Intent Chatbot.

Author: godmode-dev
License: MIT
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import uvicorn

from chatbot import IntentChatbot

app = FastAPI(
    title="NLP Intent Chatbot REST API",
    description="Production REST microservice delivering real-time intent classification and conversational responses.",
    version="1.0.0",
)

bot = IntentChatbot()


class ChatRequest(BaseModel):
    message: str = Field(..., example="Hello! What services do you provide?")


class ChatResponse(BaseModel):
    response: str
    tag: str
    confidence: float


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "NLPChatbot API"}


@app.post("/chat", response_model=ChatResponse, tags=["Conversational AI"])
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message string cannot be empty.")

    result = bot.get_response(request.message)
    return ChatResponse(**result)


@app.get("/intents", tags=["Intents"])
def get_all_intents() -> List[str]:
    return bot.tags


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
