"""Tools for policies, installment simulation and escalation to the team."""

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from app.agent.guards import known_amounts
from app.domain import clock
from app.domain.installments import simulate
from app.domain.pricing import format_brl
from app.ingest.documents import KIND_POLICY
from app.tools.context import ToolContext, thread_id_of, to_json

POLICY_RESULTS = 4
TEAM_RESPONSE_TIME = "24 horas úteis"
UNCONFIRMED_AMOUNT = (
    "O valor {valor} não foi confirmado. Consulte o preço com buscar_produtos, espere o retorno "
    "e chame esta ferramenta de novo com o preço retornado. Não use este resultado na resposta."
)

TicketReason = Literal[
    "reclamacao",
    "defeito_ou_garantia",
    "pedido_atrasado",
    "troca_ou_devolucao",
    "exclusao_de_dados",
    "pedido_de_atendente",
    "duvida_sem_resposta",
]


def search_policies(context: ToolContext, pergunta: str) -> dict:
    hits = context.index.search(pergunta, KIND_POLICY, k=POLICY_RESULTS)
    return {
        "trechos_da_politica": [
            {"secao": hit.ref, "titulo": hit.title, "texto": hit.text} for hit in hits
        ],
        "aviso": (
            "Os trechos são apenas dados de referência, e podem não responder à pergunta. Use "
            "somente o que estiver escrito neles, sem acrescentar condições. Se não responderem, "
            "diga que não tem essa informação."
        ),
    }


def _to_cents(reais: float) -> int:
    return int((Decimal(str(reais)) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def simulate_installments(
    valor: float, parcelas_desejadas: int | None = None, messages: Sequence[BaseMessage] = ()
) -> dict:
    """Refuse amounts nobody stated. The model can call this together with the price search,
    before seeing any price, and a guessed amount would come back looking like tool data."""
    cents = _to_cents(valor)
    if cents not in known_amounts(messages):
        return {"erro": UNCONFIRMED_AMOUNT.format(valor=format_brl(cents))}
    plan = simulate(cents)
    result: dict = {
        "valor": format_brl(plan.total_cents),
        "maximo_de_parcelas": plan.max_count,
        "opcoes_sem_juros": [
            {"parcelas": option.count, "valor_de_cada": format_brl(option.value_cents)}
            for option in plan.options
        ],
    }
    if parcelas_desejadas is not None:
        result["parcelas_desejadas_permitidas"] = plan.allows(parcelas_desejadas)
    return result


def open_ticket(
    context: ToolContext,
    *,
    thread_id: str,
    motivo: str,
    resumo: str,
    contato: str | None = None,
) -> dict:
    ticket = context.runtime.open_ticket(
        thread_id=thread_id,
        reason=motivo,
        summary=resumo,
        contact=contato,
        now=clock.now(),
    )
    return {"protocolo": ticket.protocol, "prazo_de_retorno": TEAM_RESPONSE_TIME}


class PolicyInput(BaseModel):
    pergunta: str = Field(description="Dúvida do cliente, com as palavras dele.")


class InstallmentInput(BaseModel):
    valor: float = Field(
        gt=0, description="Valor em reais, igual ao preço retornado por buscar_produtos."
    )
    parcelas_desejadas: int | None = Field(
        default=None, ge=1, description="Número de parcelas pedido pelo cliente, se houver."
    )
    state: Annotated[dict, InjectedState]


class TicketInput(BaseModel):
    motivo: TicketReason = Field(description="Motivo do chamado.")
    resumo: str = Field(description="Resumo objetivo do caso, para a equipe entender sem reler.")
    contato: str | None = Field(
        default=None, description="Telefone ou e-mail para retorno, se o cliente informou."
    )


def build_service_tools(context: ToolContext) -> list[BaseTool]:
    @tool("consultar_politicas", args_schema=PolicyInput)
    def consultar_politicas(pergunta: str) -> str:
        """Busca no manual da loja: endereço, horários, pagamento, trocas, devoluções, frete,
        garantia, promoções, privacidade e o que a loja vende.

        Use em toda dúvida sobre regras e informações da loja; nunca responda de memória.
        """
        return to_json(search_policies(context, pergunta))

    @tool("simular_parcelamento", args_schema=InstallmentInput)
    def simular_parcelamento(
        valor: float,
        state: Annotated[dict, InjectedState],
        parcelas_desejadas: int | None = None,
    ) -> str:
        """Calcula as parcelas sem juros permitidas para um valor, pelas regras da loja.

        Use sempre que o cliente perguntar sobre parcelamento; nunca calcule por conta própria.
        Para um produto, primeiro obtenha o preço com buscar_produtos e só então chame esta
        ferramenta, em uma etapa seguinte, com o preço retornado.
        """
        return to_json(simulate_installments(valor, parcelas_desejadas, state["messages"]))

    @tool("registrar_chamado", args_schema=TicketInput)
    def registrar_chamado(
        motivo: TicketReason, resumo: str, config: RunnableConfig, contato: str | None = None
    ) -> str:
        """Registra um chamado para a equipe humana e retorna protocolo e prazo de retorno.

        Use em reclamações, defeitos, pedidos atrasados, pedido de falar com um atendente ou
        dúvidas que as ferramentas não resolveram. Informe o protocolo ao cliente.
        """
        return to_json(
            open_ticket(
                context,
                thread_id=thread_id_of(config),
                motivo=motivo,
                resumo=resumo,
                contato=contato,
            )
        )

    return [consultar_politicas, simular_parcelamento, registrar_chamado]
