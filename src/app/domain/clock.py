"""Store time: the current moment in Campo Grande and the opening hours (policy section 2)."""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.config import get_settings

STORE_TIMEZONE = ZoneInfo("America/Campo_Grande")

_WEEKDAYS = (
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
)

# Indexed by weekday, Monday first. None means closed. Holidays are also closed, but no
# holiday calendar exists in the source data, which the README lists as a limitation.
_OPENING_HOURS: tuple[tuple[time, time] | None, ...] = (
    (time(9), time(18)),
    (time(9), time(18)),
    (time(9), time(18)),
    (time(9), time(18)),
    (time(9), time(18)),
    (time(9), time(13)),
    None,
)


def now() -> datetime:
    """Current time at the store, or the configured reference for reproducible runs."""
    reference = get_settings().reference_datetime
    if reference is None:
        return datetime.now(STORE_TIMEZONE)
    if reference.tzinfo is None:
        return reference.replace(tzinfo=STORE_TIMEZONE)
    return reference.astimezone(STORE_TIMEZONE)


def hours_of(day: date) -> tuple[time, time] | None:
    return _OPENING_HOURS[day.weekday()]


def is_open(moment: datetime) -> bool:
    hours = hours_of(moment.date())
    return hours is not None and hours[0] <= moment.time() < hours[1]


def next_opening(moment: datetime) -> datetime:
    hours = hours_of(moment.date())
    if hours is not None and moment.time() < hours[0]:
        return datetime.combine(moment.date(), hours[0], moment.tzinfo)
    day = moment.date() + timedelta(days=1)
    while (hours := hours_of(day)) is None:
        day += timedelta(days=1)
    return datetime.combine(day, hours[0], moment.tzinfo)


def describe(moment: datetime) -> str:
    """Short status line given to the agent on every turn."""
    current = f"{_WEEKDAYS[moment.weekday()]}, {moment:%d/%m/%Y}, {moment:%H:%M}"
    if is_open(moment):
        closing = hours_of(moment.date())[1]
        return f"Agora é {current}. A loja está aberta até {closing:%H:%M}."
    reopening = next_opening(moment)
    return (
        f"Agora é {current}. A loja está fechada e a equipe volta "
        f"{_WEEKDAYS[reopening.weekday()]}, {reopening:%d/%m}, às {reopening:%H:%M}."
    )
