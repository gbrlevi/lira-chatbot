"""Price rules from the store policy (sections 3 and 6.2). Money is always integer cents."""

from dataclasses import dataclass

PIX_DISCOUNT_PERCENT = 5


@dataclass(frozen=True)
class Price:
    original_cents: int
    final_cents: int
    discount_percent: int = 0
    promotion_name: str | None = None
    pix_cents: int | None = None


def format_brl(cents: int) -> str:
    """Format as Brazilian currency: 219900 -> 'R$ 2.199,00'."""
    reais, centavos = divmod(cents, 100)
    return f"R$ {reais:,}".replace(",", ".") + f",{centavos:02d}"


def _apply_discount(cents: int, percent: int) -> int:
    # Integer half-up rounding, which binary floats cannot guarantee for money.
    return (cents * (100 - percent) + 50) // 100


def compute_price(
    price_cents: int,
    promotion_percent: int | None = None,
    promotion_name: str | None = None,
) -> Price:
    """Apply an active promotion, or else offer the PIX discount. They never accumulate."""
    if promotion_percent:
        return Price(
            original_cents=price_cents,
            final_cents=_apply_discount(price_cents, promotion_percent),
            discount_percent=promotion_percent,
            promotion_name=promotion_name,
        )
    return Price(
        original_cents=price_cents,
        final_cents=price_cents,
        pix_cents=_apply_discount(price_cents, PIX_DISCOUNT_PERCENT),
    )
