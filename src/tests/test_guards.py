from pathlib import Path

import pytest
from fakes import FakeEmbedder, ScriptedModel, ai_with_tools, tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import guards
from app.agent.graph import build_graph
from app.config import get_settings
from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore
from app.tools.context import ToolContext
from app.tools.registry import build_tools


@pytest.fixture
def tools(
    store_path: Path, embedder: FakeEmbedder, tmp_path_factory: pytest.TempPathFactory
) -> list:
    runtime = RuntimeStore(tmp_path_factory.mktemp("runtime") / "runtime.db")
    return build_tools(ToolContext(store_path, HybridIndex(store_path, embedder), runtime))


@pytest.fixture
def store_open(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REFERENCE_DATETIME", "2026-03-25T10:00:00")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("REFERENCE_DATETIME")
    get_settings.cache_clear()


@pytest.fixture
def store_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REFERENCE_DATETIME", "2026-03-28T15:00:00")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("REFERENCE_DATETIME")
    get_settings.cache_clear()


def _run(graph, text: str, thread: str = "t1") -> list:
    config = {"configurable": {"thread_id": thread}}
    return graph.invoke({"messages": [HumanMessage(text)]}, config=config)["messages"]


@pytest.mark.parametrize(
    ("text", "cents"),
    [
        ("R$ 2.199,00", {219900}),
        ("R$2199", {219900}),
        ("R$ 1000", {100000}),
        ("R$ 439,2", {43920}),
        ("de R$ 549,00 por R$ 439,20", {54900, 43920}),
        ("custa 2.199,00", set()),
        ("sem valores", set()),
    ],
)
def test_amounts_in_reais(text: str, cents: set[int]) -> None:
    assert guards.amounts_in_reais(text) == cents


@pytest.mark.parametrize(
    ("text", "cents"),
    [
        ("até 1000", {100000}),
        ("até R$1000", {100000}),
        ("até mil reais", {100000}),
        ("uns 2 mil", {200000}),
        ("tenho 1.500,50", {150050}),
    ],
)
def test_customer_amounts_do_not_need_the_symbol(text: str, cents: set[int]) -> None:
    assert cents <= guards.amounts_said_by_customer(text)


@pytest.mark.parametrize(
    "text",
    ["Takamine GD20", "em 10x", "meu telefone é 99654-7821", "pedido BRJL5544332BR"],
)
def test_numbers_that_are_not_amounts_do_not_count(text: str) -> None:
    assert guards.amounts_said_by_customer(text) == set()


def test_amount_from_a_tool_is_known() -> None:
    messages = [ToolMessage('{"preco": "R$ 2.199,00"}', tool_call_id="1")]
    assert guards.unverified_amounts("Custa R$ 2.199,00.", messages) == set()


def test_amount_from_the_customer_is_known() -> None:
    messages = [HumanMessage("Tem violão até 1000?")]
    assert guards.unverified_amounts("Até R$ 1.000,00 temos estes.", messages) == set()


def test_invented_amount_is_flagged() -> None:
    messages = [ToolMessage('{"preco": "R$ 2.199,00"}', tool_call_id="1")]
    assert guards.unverified_amounts("Custa R$ 1.999,00.", messages) == {199900}


def test_amount_from_an_answer_is_not_a_source() -> None:
    messages = [AIMessage("Custa R$ 1.999,00.")]
    assert guards.unverified_amounts("Custa R$ 1.999,00.", messages) == {199900}


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_input_gets_a_greeting(text: str) -> None:
    assert guards.rejected_input_reply(text) == guards.EMPTY_INPUT_REPLY


def test_oversized_input_is_rejected() -> None:
    assert guards.rejected_input_reply("a" * 2001) == guards.TOO_LONG_INPUT_REPLY
    assert guards.rejected_input_reply("a" * 2000) is None


def test_rejected_input_never_reaches_the_model(tools: list, store_open) -> None:
    model = ScriptedModel()
    messages = _run(build_graph(model, tools), "x" * 5000)
    assert messages[-1].content == guards.TOO_LONG_INPUT_REPLY
    assert model.received == []


def test_verified_answer_passes_untouched(tools: list, store_open) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="Takamine GD20")),
        "Custa R$ 2.199,00, ou R$ 2.089,05 no PIX.",
    )
    messages = _run(build_graph(model, tools), "Preço do GD20?")
    assert messages[-1].content == "Custa R$ 2.199,00, ou R$ 2.089,05 no PIX."
    assert len(model.received) == 2


def test_invented_price_triggers_one_regeneration(tools: list, store_open) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="Takamine GD20")),
        "Custa R$ 1.000,00.",
        "Custa R$ 2.199,00.",
    )
    messages = _run(build_graph(model, tools), "Preço do GD20?")
    assert messages[-1].content == "Custa R$ 2.199,00."
    correction_prompt = model.received[2][0].content
    assert "<correcao>" in correction_prompt
    assert "R$ 1.000,00" in correction_prompt


def test_discarded_answer_is_removed_from_the_history(tools: list, store_open) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="Takamine GD20")),
        "Custa R$ 1.000,00.",
        "Custa R$ 2.199,00.",
    )
    messages = _run(build_graph(model, tools), "Preço do GD20?")
    assert not any("1.000,00" in str(message.content) for message in messages)


def test_second_invented_price_ends_in_the_safe_answer(tools: list, store_open) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="Takamine GD20")),
        "Custa R$ 1.000,00.",
        "Custa R$ 1.500,00.",
    )
    messages = _run(build_graph(model, tools), "Preço do GD20?")
    assert messages[-1].content == guards.SAFE_ANSWER
    assert not any("1.500,00" in str(message.content) for message in messages)


def test_empty_answer_is_regenerated(tools: list, store_open) -> None:
    model = ScriptedModel("", "Oi! Como posso ajudar?")
    assert _run(build_graph(model, tools), "Oi")[-1].content == "Oi! Como posso ajudar?"


def test_customer_budget_can_be_repeated_in_the_answer(tools: list, store_open) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", categoria="Violões", preco_max=1000)),
        "Até R$ 1.000,00 temos ótimas opções.",
    )
    messages = _run(build_graph(model, tools), "Violão até R$1000?")
    assert messages[-1].content == "Até R$ 1.000,00 temos ótimas opções."


def test_regeneration_budget_resets_on_each_turn(tools: list, store_open) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    model = ScriptedModel("R$ 1,00", "Oi!", "R$ 2,00", "Tudo bem!")
    graph = build_graph(model, tools, checkpointer=InMemorySaver())
    assert _run(graph, "Primeira")[-1].content == "Oi!"
    assert _run(graph, "Segunda")[-1].content == "Tudo bem!"


def test_closed_store_notice_is_added_to_the_first_reply(tools: list, store_closed) -> None:
    messages = _run(build_graph(ScriptedModel("Temos sim!"), tools), "Tem violão?")
    assert messages[-1].content.startswith("Temos sim!")
    assert "volta segunda-feira, 30/03, às 09:00" in messages[-1].content


def test_closed_store_notice_appears_only_once(tools: list, store_closed) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(
        ScriptedModel("Temos sim!", "De nada!"), tools, checkpointer=InMemorySaver()
    )
    _run(graph, "Tem violão?")
    assert _run(graph, "Obrigado")[-1].content == "De nada!"


def test_open_store_has_no_notice(tools: list, store_open) -> None:
    assert _run(build_graph(ScriptedModel("Temos sim!"), tools), "Tem?")[-1].content == "Temos sim!"
