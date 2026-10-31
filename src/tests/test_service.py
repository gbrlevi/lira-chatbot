from pathlib import Path

import pytest
from fakes import FakeEmbedder, ScriptedModel, ai_with_tools, tool_call
from google.genai import errors as genai_errors
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import MODEL_ATTEMPTS, build_graph
from app.agent.resilience import is_transient
from app.config import get_settings
from app.interfaces import cli
from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore
from app.service import UNAVAILABLE_REPLY, ChatEvent, ChatService
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


def _service(model: ScriptedModel, tools: list) -> ChatService:
    return ChatService(build_graph(model, tools, checkpointer=InMemorySaver(), retry_interval=0.01))


def _provider_error(code: int) -> genai_errors.APIError:
    error_class = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return error_class(code, {"error": {"code": code, "message": "falha"}})


@pytest.mark.parametrize("code", [429, 500, 503, 504])
def test_temporary_provider_errors_are_transient(code: int) -> None:
    assert is_transient(_provider_error(code))


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_permanent_provider_errors_are_not_transient(code: int) -> None:
    assert not is_transient(_provider_error(code))


def test_transient_error_is_found_inside_a_wrapper() -> None:
    wrapper = RuntimeError("Error calling model")
    wrapper.__cause__ = _provider_error(503)
    assert is_transient(wrapper)


def test_timeouts_are_transient() -> None:
    assert is_transient(TimeoutError("timed out"))


def test_programming_errors_are_not_transient() -> None:
    assert not is_transient(ValueError("bug"))


def test_reply_returns_the_final_text(tools: list) -> None:
    reply = _service(ScriptedModel("Oi!"), tools).reply("t1", "Oi")
    assert reply.text == "Oi!"
    assert reply.degraded is False
    assert reply.tools_used == []


def test_reply_lists_the_tools_used(tools: list) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="Yamaha C40")), "Custa R$ 599,90."
    )
    reply = _service(model, tools).reply("t1", "Preço?")
    assert [(use.name, use.args) for use in reply.tools_used] == [
        ("buscar_produtos", {"modelo": "Yamaha C40"})
    ]


def test_reply_reports_progress_events(tools: list) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="Yamaha C40")), "Custa R$ 599,90."
    )
    events: list[ChatEvent] = []
    _service(model, tools).reply("t1", "Preço?", on_event=events.append)
    assert [event.kind for event in events] == ["tool_start", "tool_end"]
    assert "Yamaha C40" in events[1].output


def test_guard_replacement_is_what_the_customer_receives(tools: list) -> None:
    model = ScriptedModel("Custa R$ 1,00.", "Custa R$ 2,00.")
    reply = _service(model, tools).reply("t1", "Preço?")
    assert "R$ 1,00" not in reply.text
    assert "R$ 2,00" not in reply.text


def test_temporary_failure_is_retried(tools: list) -> None:
    model = ScriptedModel(_provider_error(503), "Oi!")
    assert _service(model, tools).reply("t1", "Oi").text == "Oi!"
    assert len(model.received) == 2


def test_persistent_failure_returns_a_friendly_message(tools: list) -> None:
    model = ScriptedModel(*[_provider_error(503) for _ in range(MODEL_ATTEMPTS)])
    reply = _service(model, tools).reply("t1", "Oi")
    assert reply.text == UNAVAILABLE_REPLY
    assert reply.degraded is True
    assert len(model.received) == MODEL_ATTEMPTS


def test_permanent_failure_is_not_retried(tools: list) -> None:
    model = ScriptedModel(_provider_error(400), "nunca chega aqui")
    reply = _service(model, tools).reply("t1", "Oi")
    assert reply.degraded is True
    assert len(model.received) == 1


def test_conversation_continues_after_a_failure(tools: list) -> None:
    model = ScriptedModel(_provider_error(400), "Agora foi!")
    service = _service(model, tools)
    assert service.reply("t1", "Primeira").degraded is True
    assert service.reply("t1", "Segunda").text == "Agora foi!"
    humans = [m.content for m in model.received[1] if isinstance(m, HumanMessage)]
    assert humans == ["Primeira", "Segunda"]


def test_model_stuck_in_a_loop_ends_in_the_friendly_message(tools: list) -> None:
    calls = [ai_with_tools(tool_call("resumir_catalogo")) for _ in range(30)]
    reply = _service(ScriptedModel(*calls), tools).reply("t1", "Oi")
    assert reply.degraded is True


def test_cli_explains_a_missing_key_without_a_traceback(capsys: pytest.CaptureFixture) -> None:
    def missing_key():
        raise cli.MissingApiKeyError("Defina AI_API_KEY")

    original = cli.create_chat_service
    cli.create_chat_service = missing_key
    try:
        assert cli.main(["-m", "Oi"]) == 1
    finally:
        cli.create_chat_service = original
    assert "AI_API_KEY" in capsys.readouterr().out


def test_cli_single_message_prints_the_answer(
    tools: list, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(ScriptedModel("Oi! Sou a Lira."), tools)
    monkeypatch.setattr(cli, "create_chat_service", lambda: service)
    assert cli.main(["-m", "Oi"]) == 0
    assert "Sou a Lira" in capsys.readouterr().out


def test_cli_debug_shows_tool_results(
    tools: list, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="Yamaha C40")), "Custa R$ 599,90."
    )
    monkeypatch.setattr(cli, "create_chat_service", lambda: _service(model, tools))
    assert cli.main(["-m", "Preço?", "--debug"]) == 0
    output = capsys.readouterr().out
    assert "consultando buscar_produtos" in output
    assert "retorno:" in output
