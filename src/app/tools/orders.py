"""Order lookup. Identity is verified here, in code, because a prompt cannot enforce it."""

import sqlite3
from datetime import date

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.db import connect_readonly
from app.domain import clock
from app.domain.order_labels import payment_label, status_label
from app.domain.pricing import format_brl
from app.text import digits_only
from app.tools.context import ToolContext, thread_id_of, to_json

MAX_FAILED_LOOKUPS = 3
WITHDRAWAL_DAYS = 7
_BRAZIL_COUNTRY_CODE = "55"

# The same text for every mismatch, so a failure never reveals which data was wrong or
# whether an order number exists.
NOT_FOUND = (
    "Não localizei pedidos com os dados informados. Peça ao cliente para conferir o telefone "
    "da compra e o número do pedido ou o e-mail."
)
LOCKED = (
    "Limite de tentativas atingido nesta conversa. Não tente novamente: oriente o cliente a "
    "falar com a equipe pelos canais oficiais da loja."
)


def normalize_phone(raw: str) -> str | None:
    """Return the digits of a Brazilian phone with area code, or None if it cannot be one."""
    digits = digits_only(raw)
    if len(digits) in (12, 13) and digits.startswith(_BRAZIL_COUNTRY_CODE):
        digits = digits[len(_BRAZIL_COUNTRY_CODE) :]
    return digits if len(digits) in (10, 11) else None


def _format_date(iso_date: str) -> str:
    """Dates go out ready to repeat, because the model can slip when converting the format."""
    return date.fromisoformat(iso_date).strftime("%d/%m/%Y")


def _delivery_status(order: sqlite3.Row, today: date) -> dict:
    """Date arithmetic the agent receives ready, based on the estimated date only."""
    estimated = order["estimated_delivery"]
    if not estimated:
        return {}
    days_since = (today - date.fromisoformat(estimated)).days
    if order["status"] == "shipped" and days_since > 0:
        return {
            "situacao_do_prazo": f"Previsão de entrega vencida há {days_since} dias.",
            "sugestao": (
                "Informe o código de rastreio e ofereça abrir um chamado para a equipe "
                "verificar a entrega."
                if order["tracking_code"]
                else "Ofereça abrir um chamado para a equipe verificar a entrega."
            ),
        }
    if order["status"] == "delivered":
        within_window = days_since <= WITHDRAWAL_DAYS
        situation = "vigente" if within_window else "vencido"
        return {
            "prazo_de_arrependimento_de_7_dias": (
                f"provavelmente {situation}: a previsão de entrega foi há {days_since} dias"
            ),
            "orientacao_se_o_cliente_quiser_devolver": (
                "Diga que, pela previsão de entrega, ainda parece estar dentro do prazo de 7 dias."
                if within_window
                else "Diga claramente que, pela previsão de entrega, o prazo de 7 dias "
                "provavelmente já venceu, e ofereça abrir um chamado para a equipe avaliar."
            ),
            "ressalva": (
                f"{_format_date(estimated)} é a previsão, não a data em que o pedido chegou: "
                "o sistema não registra a data real de recebimento. "
                "Deixe isso claro e, se houver dúvida, abra um chamado."
            ),
        }
    return {}


def _present_order(connection: sqlite3.Connection, order: sqlite3.Row, today: date) -> dict:
    items = connection.execute(
        """
        SELECT p.name, i.quantity FROM order_items i
        JOIN products p ON p.product_id = i.product_id
        WHERE i.order_id = ?
        """,
        (order["order_id"],),
    ).fetchall()
    result = {
        "numero": order["order_id"],
        "data_do_pedido": _format_date(order["order_date"]),
        "status": status_label(order["status"]),
        "itens": [{"produto": item["name"], "quantidade": item["quantity"]} for item in items],
        "total": format_brl(order["total_cents"]),
        "pagamento": payment_label(order["payment_method"]),
    }
    if order["tracking_code"]:
        result["codigo_de_rastreio"] = order["tracking_code"]
    if order["estimated_delivery"]:
        result["previsao_de_entrega"] = _format_date(order["estimated_delivery"])
    if order["notes"]:
        result["observacao"] = order["notes"]
    result.update(_delivery_status(order, today))
    return result


def lookup_orders(
    context: ToolContext,
    *,
    thread_id: str,
    telefone: str,
    numero_pedido: int | None = None,
    email: str | None = None,
) -> dict:
    if context.runtime.failed_lookups(thread_id) >= MAX_FAILED_LOOKUPS:
        return {"erro": LOCKED}

    # Incomplete data is a request to ask the customer, not a failed attempt.
    phone = normalize_phone(telefone)
    if phone is None:
        return {"erro": "Telefone ausente ou incompleto. Peça ao cliente o telefone com DDD."}
    normalized_email = email.strip().lower() if email and email.strip() else None
    if numero_pedido is None and normalized_email is None:
        return {"erro": "Peça ao cliente o número do pedido ou o e-mail da compra."}

    with connect_readonly(context.store_path) as connection:
        customer = connection.execute(
            "SELECT customer_id, name, email FROM customers WHERE phone_digits = ?", (phone,)
        ).fetchone()
        orders: list[sqlite3.Row] = []
        if customer and normalized_email in (None, customer["email"]):
            query = "SELECT * FROM orders WHERE customer_id = ?"
            params: list = [customer["customer_id"]]
            if numero_pedido is not None:
                query += " AND order_id = ?"
                params.append(numero_pedido)
            orders = connection.execute(f"{query} ORDER BY order_date DESC", params).fetchall()

        if not orders:
            context.runtime.record_failed_lookup(thread_id)
            return {"erro": NOT_FOUND}

        today = clock.now().date()
        return {
            "primeiro_nome_do_cliente": customer["name"].split()[0],
            "pedidos": [_present_order(connection, order, today) for order in orders],
        }


class OrderLookupInput(BaseModel):
    telefone: str = Field(
        description="Telefone com DDD informado pelo cliente nesta conversa. Nunca invente."
    )
    numero_pedido: int | None = Field(
        default=None, description="Número do pedido, se o cliente informou."
    )
    email: str | None = Field(
        default=None, description="E-mail da compra, para listar pedidos sem saber o número."
    )


def build_order_tools(context: ToolContext) -> list[BaseTool]:
    @tool("consultar_pedidos", args_schema=OrderLookupInput)
    def consultar_pedidos(
        telefone: str,
        config: RunnableConfig,
        numero_pedido: int | None = None,
        email: str | None = None,
    ) -> str:
        """Consulta status, itens, total, pagamento e rastreio de pedidos do próprio cliente.

        Só chame depois que o cliente tiver informado o telefone E também o número do pedido
        ou o e-mail. Se faltar algum dado, peça ao cliente em vez de chamar a ferramenta.
        Com os dados informados, consulte antes de responder sobre o pedido, inclusive em
        dúvidas de devolução ou troca, porque o prazo depende dele.
        """
        return to_json(
            lookup_orders(
                context,
                thread_id=thread_id_of(config),
                telefone=telefone,
                numero_pedido=numero_pedido,
                email=email,
            )
        )

    return [consultar_pedidos]
