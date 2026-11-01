import json
from pathlib import Path
from uuid import uuid4

import pytest
from fakes import FakeEmbedder

from app.config import get_settings
from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore
from app.tools.context import ToolContext
from app.tools.orders import (
    LOCKED,
    MAX_FAILED_LOOKUPS,
    NOT_FOUND,
    build_order_tools,
    lookup_orders,
    normalize_phone,
)

ANA_PHONE = "(67) 99654-7821"  # customer 2, owner of order 8 (shipped)
PEDRO_PHONE = "(67) 98432-1098"  # customer 3, owner of orders 1 and 19
BRUNO_C_PHONE = "(67) 98543-2109"  # customer 11, shares an e-mail with customer 43
BRUNO_M_PHONE = "(67) 99456-3456"  # customer 43, no orders
SHARED_EMAIL = "bruno.martins@jmail.com"


@pytest.fixture
def context(
    store_path: Path, embedder: FakeEmbedder, tmp_path_factory: pytest.TempPathFactory
) -> ToolContext:
    runtime = RuntimeStore(tmp_path_factory.mktemp("runtime") / "runtime.db")
    return ToolContext(store_path, HybridIndex(store_path, embedder), runtime)


@pytest.fixture
def frozen_today(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REFERENCE_DATETIME", "2026-03-25T10:00:00")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("REFERENCE_DATETIME")
    get_settings.cache_clear()


def _lookup(context: ToolContext, **kwargs) -> dict:
    return lookup_orders(context, thread_id=kwargs.pop("thread_id", str(uuid4())), **kwargs)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("(67) 99654-7821", "67996547821"),
        ("67996547821", "67996547821"),
        ("+55 67 99654-7821", "67996547821"),
        ("(67) 3341-4444", "6733414444"),
        ("", None),
        ("99654-7821", None),
        ("abc", None),
    ],
)
def test_normalize_phone(raw: str, expected: str | None) -> None:
    assert normalize_phone(raw) == expected


def test_phone_and_order_number_return_the_order(context: ToolContext) -> None:
    result = _lookup(context, telefone=ANA_PHONE, numero_pedido=8)
    order = result["pedidos"][0]
    assert result["primeiro_nome_do_cliente"] == "Ana"
    assert order["numero"] == 8
    assert order["codigo_de_rastreio"] == "BRJL5544332BR"
    assert order["total"] == "R$ 349,90"


def test_internal_codes_are_translated(context: ToolContext) -> None:
    order = _lookup(context, telefone=ANA_PHONE, numero_pedido=8)["pedidos"][0]
    assert order["status"] == "enviado"
    assert order["pagamento"] == "cartão de débito"
    assert "shipped" not in json.dumps(order)


def test_phone_formatting_does_not_matter(context: ToolContext) -> None:
    assert "pedidos" in _lookup(context, telefone="+55 67 99654 7821", numero_pedido=8)


def test_phone_and_email_list_all_orders(context: ToolContext) -> None:
    result = _lookup(context, telefone=PEDRO_PHONE, email="PEDRO.OLIVEIRA@jmail.com")
    assert {order["numero"] for order in result["pedidos"]} == {1, 19}


def test_order_of_another_customer_is_not_found(context: ToolContext) -> None:
    assert _lookup(context, telefone=ANA_PHONE, numero_pedido=1) == {"erro": NOT_FOUND}


def test_wrong_phone_gets_the_same_message_as_a_missing_order(context: ToolContext) -> None:
    wrong_phone = _lookup(context, telefone="(67) 90000-0000", numero_pedido=8)
    missing_order = _lookup(context, telefone=ANA_PHONE, numero_pedido=999)
    wrong_email = _lookup(context, telefone=ANA_PHONE, email="outra@pessoa.com")
    assert wrong_phone == missing_order == wrong_email == {"erro": NOT_FOUND}


def test_failure_reveals_no_personal_data(context: ToolContext) -> None:
    result = json.dumps(_lookup(context, telefone="(67) 90000-0000", numero_pedido=8))
    assert "Ana" not in result
    assert "BRJL" not in result


def test_shared_email_only_returns_orders_of_the_matching_phone(context: ToolContext) -> None:
    owner = _lookup(context, telefone=BRUNO_C_PHONE, email=SHARED_EMAIL)
    assert {order["numero"] for order in owner["pedidos"]} == {10}
    other = _lookup(context, telefone=BRUNO_M_PHONE, email=SHARED_EMAIL)
    assert other == {"erro": NOT_FOUND}


def test_empty_phone_asks_for_it_and_does_not_count_as_a_failure(context: ToolContext) -> None:
    thread = str(uuid4())
    result = _lookup(context, thread_id=thread, telefone="", numero_pedido=8)
    assert "Telefone" in result["erro"]
    assert context.runtime.failed_lookups(thread) == 0


def test_phone_alone_asks_for_a_second_factor(context: ToolContext) -> None:
    result = _lookup(context, telefone=ANA_PHONE)
    assert "número do pedido ou o e-mail" in result["erro"]


def test_enumeration_is_blocked_after_the_limit(context: ToolContext) -> None:
    thread = str(uuid4())
    for order_number in range(1, MAX_FAILED_LOOKUPS + 1):
        result = _lookup(context, thread_id=thread, telefone=ANA_PHONE, numero_pedido=order_number)
        assert result == {"erro": NOT_FOUND}
    # Even the correct data is refused from now on.
    assert _lookup(context, thread_id=thread, telefone=ANA_PHONE, numero_pedido=8) == {
        "erro": LOCKED
    }


def test_limit_is_counted_per_conversation(context: ToolContext) -> None:
    thread = str(uuid4())
    for _ in range(MAX_FAILED_LOOKUPS):
        _lookup(context, thread_id=thread, telefone="(67) 90000-0000", numero_pedido=1)
    assert "pedidos" in _lookup(context, telefone=ANA_PHONE, numero_pedido=8)


def test_recorded_total_prevails_over_the_sum_of_items(context: ToolContext) -> None:
    order = _lookup(context, telefone="(67) 99678-9012", numero_pedido=3)["pedidos"][0]
    assert order["total"] == "R$ 3.450,00"


def test_dates_reach_the_agent_ready_to_be_repeated(context: ToolContext) -> None:
    # The model once rewrote 2026-02-28 as 25/02/2026 when converting the format by itself.
    order = _lookup(context, telefone=ANA_PHONE, numero_pedido=8)["pedidos"][0]
    assert order["previsao_de_entrega"] == "28/02/2026"
    assert order["data_do_pedido"] == "15/02/2026"


def test_late_shipment_gets_the_days_ready(context: ToolContext, frozen_today) -> None:
    order = _lookup(context, telefone=ANA_PHONE, numero_pedido=8)["pedidos"][0]
    assert order["situacao_do_prazo"] == "Previsão de entrega vencida há 25 dias."
    assert "chamado" in order["sugestao"]
    assert "código de rastreio" in order["sugestao"]


def test_delivered_order_states_the_withdrawal_window_and_its_caveat(
    context: ToolContext, frozen_today
) -> None:
    order = _lookup(context, telefone="(67) 99789-0123", numero_pedido=7)["pedidos"][0]
    assert order["prazo_de_arrependimento_de_7_dias"] == (
        "provavelmente vencido: a previsão de entrega foi há 36 dias"
    )
    assert "provavelmente já venceu" in order["orientacao_se_o_cliente_quiser_devolver"]
    assert "data real" in order["ressalva"]
    # The model once presented the estimated date as the day the order arrived.
    assert order["ressalva"].startswith(f"{order['previsao_de_entrega']} é a previsão")


def test_recent_delivery_is_reported_as_inside_the_window(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REFERENCE_DATETIME", "2026-02-20T10:00:00")
    get_settings.cache_clear()
    try:
        order = _lookup(context, telefone="(67) 99789-0123", numero_pedido=7)["pedidos"][0]
    finally:
        monkeypatch.delenv("REFERENCE_DATETIME")
        get_settings.cache_clear()
    assert order["prazo_de_arrependimento_de_7_dias"].startswith("provavelmente vigente")
    assert "dentro do prazo" in order["orientacao_se_o_cliente_quiser_devolver"]


def test_cancellation_reason_is_available(context: ToolContext) -> None:
    order = _lookup(context, telefone="(67) 98098-7654", numero_pedido=18)["pedidos"][0]
    assert order["status"] == "cancelado"
    assert "fora de estoque" in order["observacao"]


def test_output_never_contains_contact_data(context: ToolContext) -> None:
    result = json.dumps(_lookup(context, telefone=ANA_PHONE, numero_pedido=8))
    assert "99654" not in result
    assert "@" not in result


def test_tool_reads_the_thread_from_the_config(context: ToolContext) -> None:
    lookup = build_order_tools(context)[0]
    thread = str(uuid4())
    output = lookup.invoke(
        {"telefone": "(67) 90000-0000", "numero_pedido": 8},
        config={"configurable": {"thread_id": thread}},
    )
    assert json.loads(output) == {"erro": NOT_FOUND}
    assert context.runtime.failed_lookups(thread) == 1


def test_identifiers_of_the_session_are_not_in_the_tool_schema(context: ToolContext) -> None:
    lookup = build_order_tools(context)[0]
    assert set(lookup.args) == {"telefone", "numero_pedido", "email"}
