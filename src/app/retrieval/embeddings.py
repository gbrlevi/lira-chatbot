"""Embedding providers behind one small interface, so the index does not depend on a vendor."""

from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import get_settings

FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "models/gemini-embedding-2"


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Identifies the vector space. An index only works with the embedder that built it."""
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _unit_length(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class FastEmbedEmbedder:
    """Runs locally with ONNX: no key, no quota, and about 220 MB downloaded on first use."""

    def __init__(self, model: str = FASTEMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        cache_dir = get_settings().var_dir / "models"
        self._model = TextEmbedding(model_name=model, cache_dir=str(cache_dir))
        self._model_name = model

    @property
    def name(self) -> str:
        return f"fastembed:{self._model_name}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _unit_length(np.array(list(self._model.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class GeminiEmbedder:
    """Uses the same API key as the chat model, with a network call per query."""

    def __init__(self, model: str = GEMINI_MODEL) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._model = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=get_settings().ai_api_key
        )
        self._model_name = model

    @property
    def name(self) -> str:
        return f"gemini:{self._model_name}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _unit_length(np.array(self._model.embed_documents(texts), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _unit_length(np.array(self._model.embed_query(text), dtype=np.float32))


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if get_settings().embeddings_provider == "gemini":
        return GeminiEmbedder()
    return FastEmbedEmbedder()
