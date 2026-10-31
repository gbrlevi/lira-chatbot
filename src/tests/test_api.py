from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes import FakeEmbedder, ScriptedModel, ai_with_tools, tool_call
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_graph
from app.config import get_settings
from app.interfaces.api import MAX_REQUEST_CHARS, app
from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore
from app.service import UNAVAILABLE_REPLY, ChatService
from app.tools.context import ToolContext
from app.tools.registry import build_tools


@pytest.fixture(autouse=True)
def store_open(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REFERENCE_DATETIME", "2026-03-25T10:00:00")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("REFERENCE_DATETIME")
    get_settings.cache_clear()


@pytest.fixture
def tools(
    store_path: Path, embedder: FakeEmbedder, tmp_path_factory: pytest.TempPathFactory
) -> list:
    runtime = RuntimeStore(tmp_path_factory.mktemp("runtime") / "runtime.db")
    return build_tools(ToolContext(store_path, HybridIndex(store_path, embedder), runtime))


def _client(model: ScriptedModel, tools: list) -> Iterator[TestClient]:
    graph = build_graph(model, tools, checkpointer=InMemorySaver(), retry_interval=0.01)
    app.state.chat_service = ChatService(graph)
    with TestClient(app) as client:
        yield client
    del app.state.chat_service


def test_health(tools: list) -> None:
    for client in _client(ScriptedModel(), tools):
        assert client.get("/health").json() == {"status": "ok"}


def test_chat_returns_the_reply_and_a_new_thread(tools: list) -> None:
    for client in _client(ScriptedModel("Oi! Sou a Lira."), tools):
        body = client.post("/chat", json={"message": "Oi"}).json()
    assert body["reply"] == "Oi! Sou a Lira."
    assert body["thread_id"]
    assert body["degraded"] is False


def test_same_thread_keeps_the_context(tools: list) -> None:
    model = ScriptedModel("Oi, Ana!", "Você é a Ana.")
    for client in _client(model, tools):
        thread = client.post("/chat", json={"message": "Sou a Ana"}).json()["thread_id"]
        client.post("/chat", json={"message": "Quem sou eu?", "thread_id": thread})
    contents = [message.content for message in model.received[1][1:]]
    assert contents == ["Sou a Ana", "Oi, Ana!", "Quem sou eu?"]


def test_different_threads_do_not_share_context(tools: list) -> None:
    model = ScriptedModel("Oi!", "Olá!")
    for client in _client(model, tools):
        client.post("/chat", json={"message": "Conversa A"})
        client.post("/chat", json={"message": "Conversa B"})
    assert [message.content for message in model.received[1][1:]] == ["Conversa B"]


def test_response_lists_the_tools_used(tools: list) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="Yamaha C40")), "Custa R$ 599,90."
    )
    for client in _client(model, tools):
        body = client.post("/chat", json={"message": "Preço?"}).json()
    assert body["tools_used"] == [{"name": "buscar_produtos", "args": {"modelo": "Yamaha C40"}}]


def test_provider_failure_keeps_the_http_contract(tools: list) -> None:
    error = genai_errors.ClientError(400, {"error": {"code": 400, "message": "falha"}})
    for client in _client(ScriptedModel(error), tools):
        response = client.post("/chat", json={"message": "Oi"})
    assert response.status_code == 200
    assert response.json()["reply"] == UNAVAILABLE_REPLY
    assert response.json()["degraded"] is True


def test_missing_message_is_rejected(tools: list) -> None:
    for client in _client(ScriptedModel(), tools):
        assert client.post("/chat", json={}).status_code == 422


def test_oversized_request_is_rejected_before_the_agent(tools: list) -> None:
    model = ScriptedModel()
    for client in _client(model, tools):
        response = client.post("/chat", json={"message": "x" * (MAX_REQUEST_CHARS + 1)})
    assert response.status_code == 422
    assert model.received == []


def test_customer_sized_long_message_gets_a_polite_answer(tools: list) -> None:
    model = ScriptedModel()
    for client in _client(model, tools):
        body = client.post("/chat", json={"message": "x" * 3000}).json()
    assert "longa" in body["reply"]
    assert model.received == []
