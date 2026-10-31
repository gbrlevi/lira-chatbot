"""Decides how a product is presented. Status always prevails over stock (policy 7.3)."""

from enum import StrEnum

# Below this the customer sees "últimas unidades". Exact counts are internal data.
LOW_STOCK_THRESHOLD = 2


class Availability(StrEnum):
    AVAILABLE = "disponível"
    LOW_STOCK = "últimas unidades"
    OUT_OF_STOCK = "temporariamente indisponível"
    DISCONTINUED = "fora de catálogo"
    COMING_SOON = "em breve"

    @property
    def purchasable(self) -> bool:
        return self in (Availability.AVAILABLE, Availability.LOW_STOCK)


def classify(status: str, stock_quantity: int) -> Availability:
    if status == "discontinued":
        return Availability.DISCONTINUED
    if status == "coming_soon":
        return Availability.COMING_SOON
    if status != "active":
        raise ValueError(f"Status de produto desconhecido: {status!r}")
    if stock_quantity <= 0:
        return Availability.OUT_OF_STOCK
    if stock_quantity <= LOW_STOCK_THRESHOLD:
        return Availability.LOW_STOCK
    return Availability.AVAILABLE
