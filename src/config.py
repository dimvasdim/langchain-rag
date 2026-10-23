"""
Shared model configuration for all lessons.

Pick the provider with LLM_PROVIDER in .env:
- "ollama" (default): local models, no quota
- "google": Gemini API (needs GOOGLE_API_KEY, free tier is rate limited)
"""

import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b")
# Bigger local model, used as the "expensive" route in cost_optimization.py
OLLAMA_LARGE_CHAT_MODEL = os.getenv("OLLAMA_LARGE_CHAT_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")

GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.8-flash")
GOOGLE_EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIMENSIONS = 768

# Cosine distance everywhere: Gemini embeddings below 3072 dimensions are not
# normalized, so L2 distance would give misleading rankings.
CHROMA_METADATA = {"hnsw:space": "cosine"}


class GemmaOllamaEmbeddings(OllamaEmbeddings):
    """
    OllamaEmbeddings with the task prefixes embeddinggemma was trained on.
    Without them, retrieval quality is noticeably lower.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"title: none | text: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"task: search result | query: {text}")


def get_llm(temperature: float = 0, large: bool = False) -> BaseChatModel:
    """
    Return the chat model for the selected provider.
    large=True picks the bigger local model (Gemini uses the same model for both).
    """
    if PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GOOGLE_CHAT_MODEL,
            temperature=temperature,
            thinking_level="low",
        )
    model = OLLAMA_LARGE_CHAT_MODEL if large else OLLAMA_CHAT_MODEL
    return ChatOllama(model=model, temperature=temperature, reasoning=False)


def get_embeddings() -> Embeddings:
    """Return the embedding model for the selected provider."""
    if PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBED_MODEL,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        )
    if OLLAMA_EMBED_MODEL.startswith("embeddinggemma"):
        return GemmaOllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
