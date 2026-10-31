import json
from pathlib import Path

import pytest
from fakes import FakeEmbedder, ScriptedModel, ai_with_tools, tool_call
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.agent.graph import build_graph
from app.config import get_settings
from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore
from app.tools.context import ToolContext
from app.tools.registry import build_tools


@pytest.fixture(autouse=True)
def store_open(monkeypatch: pytest.MonkeyPatch):
    """Keeps answers free of the closed-store notice, whatever the real time is."""
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


def _run(graph, text: str, thread: str = "t1") -> list:
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 25}
    return graph.invoke({"messages": [HumanMessage(text)]}, config=config)["messages"]


def test_answer_without_tools_ends_right_away(tools: list) -> None:
    model = ScriptedModel("Oi! Como posso ajudar?")
    messages = _run(build_graph(model, tools), "Oi")
    assert messages[-1].content == "Oi! Como posso ajudar?"
    assert len(model.received) == 1


def test_system_prompt_is_sent_on_every_call(tools: list) -> None:
    model = ScriptedModel(ai_with_tools(tool_call("resumir_catalogo")), "Temos violões.")
    _run(build_graph(model, tools), "O que vocês vendem?")
    for received in model.received:
        assert isinstance(received[0], SystemMessage)
        assert "Lira" in received[0].content


def test_tool_result_goes_back_to_the_model(tools: list) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="Takamine GD20")),
        "Custa R$ 2.199,00.",
    )
    messages = _run(build_graph(model, tools), "Quanto custa o GD20?")
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert json.loads(tool_message.content)["produtos"][0]["id"] == 95
    assert isinstance(model.received[1][-1], ToolMessage)


def test_two_tools_in_sequence_for_a_mixed_question(tools: list) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="Yamaha C40")),
        ai_with_tools(tool_call("simular_parcelamento", valor=599.9, parcelas_desejadas=12)),
        "Em até 6x.",
    )
    messages = _run(build_graph(model, tools), "Parcela em 12x?")
    assert [m.name for m in messages if isinstance(m, ToolMessage)] == [
        "buscar_produtos",
        "simular_parcelamento",
    ]


def test_guessed_price_in_a_parallel_call_is_refused_and_recovered(tools: list) -> None:
    # Reproduces a real failure: the model asked for the installments together with the
    # price search, inventing the amount before seeing any price.
    model = ScriptedModel(
        ai_with_tools(
            tool_call("buscar_produtos", modelo="Takamine GD20"),
            tool_call("simular_parcelamento", valor=2490, parcelas_desejadas=10),
        ),
        ai_with_tools(tool_call("simular_parcelamento", valor=2199, parcelas_desejadas=10)),
        "Custa R$ 2.199,00, em 10x de R$ 219,90.",
    )
    messages = _run(build_graph(model, tools), "GD20 em 10x?")
    simulations = [m for m in messages if isinstance(m, ToolMessage) and m.name.startswith("sim")]
    assert "não foi confirmado" in json.loads(simulations[0].content)["erro"]
    assert "R$ 219,90" in simulations[1].content
    assert not any("249,00" in str(message.content) for message in messages)
    assert messages[-1].content == "Custa R$ 2.199,00, em 10x de R$ 219,90."


def test_parallel_tool_calls_all_run(tools: list) -> None:
    model = ScriptedModel(
        ai_with_tools(
            tool_call("buscar_produtos", modelo="Yamaha C40"),
            tool_call("consultar_politicas", pergunta="PIX"),
        ),
        "Pronto.",
    )
    messages = _run(build_graph(model, tools), "Preço e PIX?")
    assert len([m for m in messages if isinstance(m, ToolMessage)]) == 2


def test_invalid_arguments_return_to_the_model_as_a_correctable_error(tools: list) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("simular_parcelamento", valor=-5)),
        "Desculpe, pode confirmar o valor?",
    )
    messages = _run(build_graph(model, tools), "Parcela?")
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert "Parâmetros inválidos" in tool_message.content


def test_unexpected_tool_failure_does_not_leak_details() -> None:
    @tool
    def buscar_produtos(modelo: str) -> str:
        """Busca."""
        raise RuntimeError("sqlite3 file C:/segredo/emporio.db is locked")

    model = ScriptedModel(
        ai_with_tools(tool_call("buscar_produtos", modelo="x")), "Não consegui consultar agora."
    )
    messages = _run(build_graph(model, [buscar_produtos]), "Preço?")
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert "segredo" not in tool_message.content
    assert "contatos da loja" in tool_message.content
    assert isinstance(messages[-1], AIMessage)


def test_tool_receives_the_thread_of_the_conversation(tools: list) -> None:
    model = ScriptedModel(
        ai_with_tools(tool_call("registrar_chamado", motivo="reclamacao", resumo="Quebrou")),
        "Registrado.",
    )
    messages = _run(build_graph(model, tools), "Quebrou!", thread="conversa-42")
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert json.loads(tool_message.content)["protocolo"].startswith("EM-")
