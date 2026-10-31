"""Test doubles that keep the suite offline and deterministic."""

import hashlib
from uuid import uuid4

import numpy as np
from langchain_core.messages import AIMessage, BaseMessage

from app.text import normalize


def tool_call(name: str, **args) -> dict:
    return {"name": name, "args": args, "id": f"call-{uuid4().hex}", "type": "tool_call"}


def ai_with_tools(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


class ScriptedModel:
    """Answers with a fixed script and records what it received on each call.

    An exception in the script is raised instead of returned, to simulate provider failures.
    """

    def __init__(self, *responses: AIMessage | str | Exception) -> None:
        self._responses = list(responses)
        self.received: list[list[BaseMessage]] = []
        self.bound_tools: list = []

    def bind_tools(self, tools: list) -> "ScriptedModel":
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[BaseMessage], *_args, **_kwargs) -> AIMessage:
        self.received.append(list(messages))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=response) if isinstance(response, str) else response


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
