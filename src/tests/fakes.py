"""Test doubles that keep the suite offline and deterministic."""

import hashlib

import numpy as np

from app.text import normalize


class FakeEmbedder:
    """Hashes words into a fixed vector, so shared words produce similar vectors."""

    name = "fake:hashing"
    dimensions = 256

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in normalize(text).split():
            digest = hashlib.md5(word.encode(), usedforsecurity=False).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)
