import pytest

from app.domain.installments import simulate
from app.domain.pricing import compute_price, format_brl


@pytest.mark.parametrize(
    ("cents", "text"),
    [
        (219900, "R$ 2.199,00"),
        (1149900, "R$ 11.499,00"),
        (43920, "R$ 439,20"),
        (5, "R$ 0,05"),
        (100000000, "R$ 1.000.000,00"),
    ],
)
def test_format_brl(cents: int, text: str) -> None:
    assert format_brl(cents) == text


def test_regular_price_offers_pix_discount() -> None:
    price = compute_price(219900)
    assert price.final_cents == 219900
    assert price.pix_cents == 208905
    assert price.discount_percent == 0


@pytest.mark.parametrize(
    ("cents", "percent", "final"),
    [(54900, 20, 43920), (35900, 10, 32310), (239900, 18, 196718), (599900, 8, 551908)],
)
def test_active_promotions_of_the_catalog(cents: int, percent: int, final: int) -> None:
    assert compute_price(cents, percent).final_cents == final


def test_pix_does_not_accumulate_with_promotion() -> None:
    price = compute_price(54900, 20, "Desconto Exclusivo")
    assert price.pix_cents is None
    assert price.promotion_name == "Desconto Exclusivo"
    assert price.original_cents == 54900


def test_discount_rounds_half_up() -> None:
    # 5% off 19,90 is 18,905, which rounds to 18,91.
    assert compute_price(1990).pix_cents == 1891


@pytest.mark.parametrize(
    ("total", "maximum"),
    [
        (42990, 5),  # 6x would be R$ 71,65, below the R$ 80,00 minimum
        (59990, 6),  # 7x would be R$ 85,70, below the R$ 100,00 minimum
        (120000, 12),  # exactly R$ 100,00 each
        (119999, 11),  # one cent short of 12x
        (15990, 3),  # 4x would be below R$ 80,00
        (15000, 3),  # exactly R$ 50,00 each
        (14999, 2),
        (9000, 1),  # 2x would be below R$ 50,00
    ],
)
def test_installment_tiers(total: int, maximum: int) -> None:
    assert simulate(total).max_count == maximum


def test_installment_values_are_rounded_to_the_cent() -> None:
    plan = simulate(59990)
    assert plan.options[-1].count == 6
    assert plan.options[-1].value_cents == 9998


def test_paying_once_is_allowed_for_any_value() -> None:
    plan = simulate(100)
    assert plan.allows(1)
    assert not plan.allows(2)


def test_gap_in_tiers_does_not_create_holes() -> None:
    # 3x is allowed at R$ 50,00, but 4x needs R$ 80,00 each, so the list stops at 3.
    plan = simulate(20000)
    assert [option.count for option in plan.options] == [1, 2, 3]


@pytest.mark.parametrize("total", [0, -100])
def test_invalid_total_is_rejected(total: int) -> None:
    with pytest.raises(ValueError):
        simulate(total)
