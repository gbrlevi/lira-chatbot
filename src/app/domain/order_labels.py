"""Translates internal order codes, so they never reach the customer as raw values."""

import re

_STATUS_LABELS = {
    "pending": "pendente",
    "confirmed": "confirmado",
    "shipped": "enviado",
    "delivered": "entregue",
    "cancelled": "cancelado",
}

_PAYMENT_LABELS = {
    "pix": "PIX",
    "debit": "cartão de débito",
    "boleto": "boleto bancário",
}

_CREDIT_PATTERN = re.compile(r"credit_(\d+)x")


def status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, "em processamento")


def payment_label(method: str) -> str:
    credit = _CREDIT_PATTERN.fullmatch(method)
    if credit:
        return f"cartão de crédito em {credit.group(1)}x"
    return _PAYMENT_LABELS.get(method, "não informado")
