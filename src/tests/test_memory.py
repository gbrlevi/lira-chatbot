from pathlib import Path

import pytest
from fakes import FakeEmbedder, ScriptedModel, ai_with_tools, tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import build_graph
from app.agent.history import MAX_TURNS_IN_CONTEXT, select_window
from app.agent.persistence import open_checkpointer
from app.config import get_settings
from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore
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


def _turn(number: int, with_tool: bool = False) -> list:
    messages = [HumanMessage(f"pergunta {number}")]
    if with_tool:
        call = tool_call("resumir_catalogo")
        messages += [
            AIMessage(content="", tool_calls=[call]),
            ToolMessage("{}", tool_call_id=call["id"], name="resumir_catalogo"),
        ]
    messages.append(AIMessage(f"resposta {number}"))
    return messages


def _config(thread: str = "t1") -> dict:
    return {"configurable": {"thread_id": thread}}


def test_short_conversation_is_untouched() -> None:
    messages = _turn(1) + _turn(2)
    assert select_window(messages) == messages


def test_long_conversation_keeps_only_the_latest_turns() -> None:
    messages = [message for number in range(30) for message in _turn(number)]
    window = select_window(messages)
    assert sum(isinstance(m, HumanMessage) for m in window) == MAX_TURNS_IN_CONTEXT
    assert window[0].content == f"pergunta {30 - MAX_TURNS_IN_CONTEXT}"
    assert window[-1].content == "resposta 29"


def test_window_never_separates_a_tool_call_from_its_result() -> None:
    messages = [message for number in range(30) for message in _turn(number, with_tool=True)]
    window = select_window(messages)
    assert isinstance(window[0], HumanMessage)
    call_ids = {call["id"] for m in window if isinstance(m, AIMessage) for call in m.tool_calls}
    result_ids = {m.tool_call_id for m in window if isinstance(m, ToolMessage)}
    assert call_ids == result_ids


def test_model_only_receives_the_window_while_the_checkpoint_keeps_everything(
    tools: list, tmp_path: Path
) -> None:
    turns = MAX_TURNS_IN_CONTEXT + 3
    model = ScriptedModel(*[f"resposta {number}" for number in range(turns)])
    graph = build_graph(model, tools, checkpointer=open_checkpointer(tmp_path / "memory.db"))
    for number in range(turns):
        result = graph.invoke({"messages": [HumanMessage(f"pergunta {number}")]}, _config())
    humans_sent = [m for m in model.received[-1] if isinstance(m, HumanMessage)]
    assert len(humans_sent) == MAX_TURNS_IN_CONTEXT
    assert sum(isinstance(m, HumanMessage) for m in result["messages"]) == turns


def test_conversation_survives_a_restart(tools: list, tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = build_graph(ScriptedModel("Oi, Ana!"), tools, checkpointer=open_checkpointer(path))
    first.invoke({"messages": [HumanMessage("Meu nome é Ana")]}, _config())

    model = ScriptedModel("Você é a Ana.")
    restarted = build_graph(model, tools, checkpointer=open_checkpointer(path))
    restarted.invoke({"messages": [HumanMessage("Qual é meu nome?")]}, _config())

    contents = [message.content for message in model.received[0][1:]]
    assert contents == ["Meu nome é Ana", "Oi, Ana!", "Qual é meu nome?"]


def test_threads_are_isolated(tools: list, tmp_path: Path) -> None:
    model = ScriptedModel("Oi!", "Olá!")
    graph = build_graph(model, tools, checkpointer=open_checkpointer(tmp_path / "memory.db"))
    graph.invoke({"messages": [HumanMessage("Sou da conversa A")]}, _config("a"))
    graph.invoke({"messages": [HumanMessage("Sou da conversa B")]}, _config("b"))
    assert [message.content for message in model.received[1][1:]] == ["Sou da conversa B"]


def test_tool_history_is_replayed_after_a_restart(tools: list, tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first_model = ScriptedModel(ai_with_tools(tool_call("resumir_catalogo")), "Temos violões.")
    first = build_graph(first_model, tools, checkpointer=open_checkpointer(path))
    first.invoke({"messages": [HumanMessage("O que vendem?")]}, _config())

    model = ScriptedModel("Isso mesmo.")
    restarted = build_graph(model, tools, checkpointer=open_checkpointer(path))
    restarted.invoke({"messages": [HumanMessage("Violões, certo?")]}, _config())
    assert any(isinstance(message, ToolMessage) for message in model.received[0])
