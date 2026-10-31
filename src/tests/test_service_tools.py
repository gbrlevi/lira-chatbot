import json
from datetime import datetime
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from app.domain.clock import STORE_TIMEZONE
from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore
from app.tools.context import ToolContext
from app.tools.registry import build_tools
from app.tools.service import open_ticket, search_policies, simulate_installments


@pytest.fixture
def context(
    store_path: Path, embedder: FakeEmbedder, tmp_path_factory: pytest.TempPathFactory
) -> ToolContext:
    runtime = RuntimeStore(tmp_path_factory.mktemp("runtime") / "runtime.db")
    return ToolContext(store_path, HybridIndex(store_path, embedder), runtime)


def test_policy_search_cites_the_section(context: ToolContext) -> None:
    result = search_policies(context, "Jadlog")
    first = result["trechos_da_politica"][0]
    assert first["secao"] == "5.2"
    assert "Jadlog" in first["texto"]


def test_policy_search_warns_that_excerpts_may_not_answer(context: ToolContext) -> None:
    result = search_policies(context, "tem estacionamento?")
    assert "não tem essa informação" in result["aviso"]
    assert "sem acrescentar condições" in result["aviso"]


def test_installments_list_only_allowed_options() -> None:
    result = simulate_installments(429.90, 6)
    assert result["valor"] == "R$ 429,90"
    assert result["maximo_de_parcelas"] == 5
    assert result["parcelas_desejadas_permitidas"] is False
    assert result["opcoes_sem_juros"][-1] == {"parcelas": 5, "valor_de_cada": "R$ 85,98"}


def test_installments_handle_float_values_exactly() -> None:
    assert simulate_installments(599.9)["valor"] == "R$ 599,90"


def test_allowed_installment_is_confirmed() -> None:
    assert simulate_installments(1200, 12)["parcelas_desejadas_permitidas"] is True


def test_ticket_returns_protocol_and_response_time(context: ToolContext) -> None:
    result = open_ticket(context, thread_id="t1", motivo="reclamacao", resumo="Chegou quebrado")
    assert result["protocolo"].startswith("EM-")
    assert result["prazo_de_retorno"] == "24 horas úteis"


def test_ticket_is_persisted_for_the_team(context: ToolContext) -> None:
    open_ticket(
        context, thread_id="t1", motivo="reclamacao", resumo="Chegou quebrado", contato="6799"
    )
    ticket = context.runtime.list_tickets()[0]
    assert ticket["reason"] == "reclamacao"
    assert ticket["summary"] == "Chegou quebrado"
    assert ticket["contact"] == "6799"


def test_repeated_call_does_not_duplicate_the_ticket(context: ToolContext) -> None:
    first = open_ticket(context, thread_id="t1", motivo="reclamacao", resumo="Primeira versão")
    second = open_ticket(context, thread_id="t1", motivo="reclamacao", resumo="Segunda versão")
    assert first == second
    assert len(context.runtime.list_tickets()) == 1


def test_distinct_reasons_and_conversations_open_new_tickets(context: ToolContext) -> None:
    open_ticket(context, thread_id="t1", motivo="reclamacao", resumo="a")
    open_ticket(context, thread_id="t1", motivo="pedido_atrasado", resumo="b")
    open_ticket(context, thread_id="t2", motivo="reclamacao", resumo="c")
    protocols = {row["protocol"] for row in context.runtime.list_tickets()}
    assert len(protocols) == 3


def test_protocol_uses_the_store_date(tmp_path: Path) -> None:
    runtime = RuntimeStore(tmp_path / "runtime.db")
    ticket = runtime.open_ticket(
        thread_id="t",
        reason="reclamacao",
        summary="s",
        contact=None,
        now=datetime(2026, 3, 25, tzinfo=STORE_TIMEZONE),
    )
    assert ticket.protocol == "EM-20260325-0001"


def test_runtime_data_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    RuntimeStore(path).record_failed_lookup("t")
    assert RuntimeStore(path).failed_lookups("t") == 1


def test_registry_exposes_the_seven_tools(context: ToolContext) -> None:
    names = {tool.name for tool in build_tools(context)}
    assert names == {
        "buscar_produtos",
        "detalhar_produtos",
        "resumir_catalogo",
        "consultar_pedidos",
        "consultar_politicas",
        "simular_parcelamento",
        "registrar_chamado",
    }


def test_every_tool_describes_when_to_use_it(context: ToolContext) -> None:
    for tool in build_tools(context):
        assert "Use" in tool.description or "chame" in tool.description, tool.name
        for name, field in tool.args_schema.model_fields.items():
            assert field.description, f"{tool.name}.{name}"


def test_tools_return_json_through_the_tool_interface(context: ToolContext) -> None:
    tools = {tool.name: tool for tool in build_tools(context)}
    assert json.loads(tools["simular_parcelamento"].invoke({"valor": 500.0}))["valor"]
    assert json.loads(tools["consultar_politicas"].invoke({"pergunta": "PIX"}))[
        "trechos_da_politica"
    ]
    ticket = tools["registrar_chamado"].invoke(
        {"motivo": "duvida_sem_resposta", "resumo": "Sem resposta"},
        config={"configurable": {"thread_id": "t9"}},
    )
    assert json.loads(ticket)["protocolo"]
