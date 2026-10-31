"""Text normalization shared by ingestion and search."""

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def normalize(value: str) -> str:
    """Lowercase, remove accents and collapse punctuation: 'GD-20 Aço' -> 'gd 20 aco'."""
    return _NON_ALNUM.sub(" ", strip_accents(value).lower()).strip()


def compact(value: str) -> str:
    """Normalize and drop spaces, so 'GD-20', 'GD 20' and 'GD20' become the same key."""
    return normalize(value).replace(" ", "")


def digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)
