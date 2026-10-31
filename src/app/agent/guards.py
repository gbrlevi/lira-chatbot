"""Code guardrails around the model, with no extra LLM calls.

The output check only covers amounts in reais: it cannot catch an invented condition, which
is a limitation stated in the README and a case for the production gatekeeper.
"""

import re
from collections.abc import Iterable

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

MAX_INPUT_CHARS = 2000

EMPTY_INPUT_REPLY = "Oi! Sou a Lira, assistente virtual da Empório da Música. Como posso te ajudar?"
TOO_LONG_INPUT_REPLY = (
    "Sua mensagem ficou um pouco longa para mim. Pode resumir o que você precisa em poucas frases?"
)
SAFE_ANSWER = (
    "Desculpa, não consegui confirmar essa informação no sistema agora. Para não te passar nada "
    "errado, fale com a equipe pelo WhatsApp (67) 3321-4500 ou pelo telefone (67) 3341-4444."
)
UNVERIFIED_AMOUNT_CORRECTION = (
    "A resposta anterior foi descartada porque citou valores que não constam nos retornos das "
    "ferramentas: {amounts}. Responda de novo usando somente valores retornados, sem fazer contas. "
    "Se precisar de um valor, consulte a ferramenta."
)
EMPTY_ANSWER_CORRECTION = "A resposta anterior veio vazia. Responda à última mensagem do cliente."

_AMOUNT_IN_REAIS = re.compile(r"R\$\s*(\d[\d.]*)(?:,(\d{1,2}))?")
_PLAIN_NUMBER = re.compile(r"(?<![\w.,-])(\d[\d.]*)(?:,(\d{1,2}))?(?![\w-])")
_THOUSANDS = re.compile(r"(?:(\d+)\s*)?\bmil\b", re.IGNORECASE)


def _to_cents(whole: str, fraction: str | None) -> int | None:
    digits = whole.replace(".", "")
    if not digits.isdigit():
        return None
    return int(digits) * 100 + int((fraction or "0").ljust(2, "0"))


def amounts_in_reais(text: str) -> set[int]:
    """Amounts written with the currency symbol, in cents."""
    found = (_to_cents(*match.groups()) for match in _AMOUNT_IN_REAIS.finditer(text))
    return {amount for amount in found if amount is not None}


def amounts_said_by_customer(text: str) -> set[int]:
    """Customers rarely write the symbol: 'até 1000' and 'até mil reais' are amounts too."""
    found = {_to_cents(*match.groups()) for match in _PLAIN_NUMBER.finditer(text)}
    for match in _THOUSANDS.finditer(text):
        found.add(int(match.group(1) or 1) * 1000 * 100)
    return {amount for amount in found if amount is not None}


def known_amounts(messages: Iterable[BaseMessage]) -> set[int]:
    known: set[int] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            known |= amounts_in_reais(str(message.content))
        elif isinstance(message, HumanMessage):
            known |= amounts_said_by_customer(str(message.content))
    return known


def unverified_amounts(answer: str, messages: Iterable[BaseMessage]) -> set[int]:
    return amounts_in_reais(answer) - known_amounts(messages)


def rejected_input_reply(text: str) -> str | None:
    """Return the fixed reply for an unusable message, or None to let it through."""
    stripped = text.strip()
    if not stripped:
        return EMPTY_INPUT_REPLY
    if len(stripped) > MAX_INPUT_CHARS:
        return TOO_LONG_INPUT_REPLY
    return None
