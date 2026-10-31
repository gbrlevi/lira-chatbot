"""Composes the system prompt from the XML modules in src/prompts."""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

from app.config import get_settings
from app.domain import clock

MODULE_ORDER = ("persona", "scope", "service_rules", "tool_usage", "output_format")


@lru_cache(maxsize=1)
def _load_static_prompt(prompts_dir: Path) -> str:
    modules = (
        (prompts_dir / f"{name}.xml").read_text(encoding="utf-8").strip() for name in MODULE_ORDER
    )
    return "\n\n".join(modules)


def build_system_prompt(moment: datetime | None = None) -> str:
    """Static modules first, dynamic data last, which keeps the prefix stable for caching."""
    status = clock.describe(moment or clock.now())
    static = _load_static_prompt(get_settings().prompts_dir)
    return f"{static}\n\n<contexto_atual>{escape(status)}</contexto_atual>"
