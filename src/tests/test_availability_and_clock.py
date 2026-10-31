from datetime import UTC, datetime, timedelta

import pytest

from app.config import get_settings
from app.domain import clock
from app.domain.availability import Availability, classify
from app.domain.order_labels import payment_label, status_label

TZ = clock.STORE_TIMEZONE


def _at(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TZ)


@pytest.mark.parametrize(
    ("status", "stock", "expected", "purchasable"),
    [
        ("active", 12, Availability.AVAILABLE, True),
        ("active", 3, Availability.AVAILABLE, True),
        ("active", 2, Availability.LOW_STOCK, True),
        ("active", 0, Availability.OUT_OF_STOCK, False),
        ("discontinued", 13, Availability.DISCONTINUED, False),
        ("discontinued", 0, Availability.DISCONTINUED, False),
        ("coming_soon", 2, Availability.COMING_SOON, False),
    ],
)
def test_status_prevails_over_stock(
    status: str, stock: int, expected: Availability, purchasable: bool
) -> None:
    result = classify(status, stock)
    assert result is expected
    assert result.purchasable is purchasable


def test_unknown_status_fails_loudly() -> None:
    with pytest.raises(ValueError):
        classify("archived", 5)


@pytest.mark.parametrize(
    ("code", "label"),
    [
        ("shipped", "enviado"),
        ("delivered", "entregue"),
        ("pending", "pendente"),
        ("confirmed", "confirmado"),
        ("cancelled", "cancelado"),
    ],
)
def test_status_labels(code: str, label: str) -> None:
    assert status_label(code) == label


@pytest.mark.parametrize(
    ("code", "label"),
    [
        ("pix", "PIX"),
        ("debit", "cartão de débito"),
        ("boleto", "boleto bancário"),
        ("credit_12x", "cartão de crédito em 12x"),
        ("credit_3x", "cartão de crédito em 3x"),
    ],
)
def test_payment_labels(code: str, label: str) -> None:
    assert payment_label(code) == label


def test_unknown_codes_do_not_leak_raw_values() -> None:
    assert "weird" not in status_label("weird")
    assert "weird" not in payment_label("weird")


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (_at(2026, 3, 25, 10), True),  # Wednesday
        (_at(2026, 3, 25, 8, 59), False),
        (_at(2026, 3, 25, 18), False),  # closing time is exclusive
        (_at(2026, 3, 28, 12, 59), True),  # Saturday
        (_at(2026, 3, 28, 13), False),
        (_at(2026, 3, 29, 10), False),  # Sunday
    ],
)
def test_is_open(moment: datetime, expected: bool) -> None:
    assert clock.is_open(moment) is expected


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (_at(2026, 3, 25, 7), _at(2026, 3, 25, 9)),  # before opening, same day
        (_at(2026, 3, 25, 19), _at(2026, 3, 26, 9)),  # after closing, next day
        (_at(2026, 3, 27, 19), _at(2026, 3, 28, 9)),  # Friday night, Saturday morning
        (_at(2026, 3, 28, 14), _at(2026, 3, 30, 9)),  # Saturday afternoon skips Sunday
        (_at(2026, 3, 29, 10), _at(2026, 3, 30, 9)),  # Sunday
    ],
)
def test_next_opening(moment: datetime, expected: datetime) -> None:
    assert clock.next_opening(moment) == expected


def test_describe_when_open() -> None:
    text = clock.describe(_at(2026, 3, 28, 10, 30))
    assert "sábado, 28/03/2026, 10:30" in text
    assert "aberta até 13:00" in text


def test_describe_when_closed_says_when_the_team_returns() -> None:
    text = clock.describe(_at(2026, 3, 28, 15))
    assert "fechada" in text
    assert "segunda-feira, 30/03, às 09:00" in text


def test_reference_datetime_overrides_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFERENCE_DATETIME", "2026-03-25T10:00:00")
    get_settings.cache_clear()
    try:
        assert clock.now() == _at(2026, 3, 25, 10)
    finally:
        monkeypatch.delenv("REFERENCE_DATETIME")
        get_settings.cache_clear()


def test_reference_with_other_offset_is_converted_to_store_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REFERENCE_DATETIME", "2026-03-25T14:00:00+00:00")
    get_settings.cache_clear()
    try:
        moment = clock.now()
        assert moment.utcoffset() == timedelta(hours=-4)
        assert moment.hour == 10
        assert moment == datetime(2026, 3, 25, 14, tzinfo=UTC)
    finally:
        monkeypatch.delenv("REFERENCE_DATETIME")
        get_settings.cache_clear()


def test_real_clock_uses_store_timezone() -> None:
    assert clock.now().tzinfo == TZ
