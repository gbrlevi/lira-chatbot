import xml.etree.ElementTree as ET
from datetime import datetime

import pytest

from app.agent.prompts import MODULE_ORDER, build_system_prompt
from app.config import get_settings
from app.domain.clock import STORE_TIMEZONE

OPEN_MOMENT = datetime(2026, 3, 25, 10, tzinfo=STORE_TIMEZONE)
CLOSED_MOMENT = datetime(2026, 3, 28, 15, tzinfo=STORE_TIMEZONE)

CANONICAL_SECTIONS = {
    "task_context",
    "tone_context",
    "context_data",
    "detailed_task_description",
    "examples",
    "conversation_history",
    "immediate_task",
    "reasoning_trigger",
    "output_format",
    "prefilled_response",
}


@pytest.mark.parametrize("name", MODULE_ORDER)
def test_each_module_is_well_formed_xml(name: str) -> None:
    ET.parse(get_settings().prompts_dir / f"{name}.xml")


def test_composed_prompt_is_well_formed_and_has_every_section() -> None:
    root = ET.fromstring(f"<root>{build_system_prompt(OPEN_MOMENT)}</root>")
    assert CANONICAL_SECTIONS <= {element.tag for element in root.iter()}


def test_critical_rules_open_and_close_the_prompt() -> None:
    prompt = build_system_prompt(OPEN_MOMENT)
    assert prompt.index("Regras críticas") < prompt.index("<scope>")
    assert prompt.index("<critical_reminders>") > prompt.index("<examples>")


def test_dynamic_context_is_the_last_block() -> None:
    prompt = build_system_prompt(OPEN_MOMENT)
    assert prompt.endswith("A loja está aberta até 18:00.</contexto_atual>")


def test_context_reflects_the_moment_and_static_part_stays_identical() -> None:
    opened = build_system_prompt(OPEN_MOMENT)
    closed = build_system_prompt(CLOSED_MOMENT)
    assert "fechada" in closed
    assert opened.split("<contexto_atual>")[0] == closed.split("<contexto_atual>")[0]


def test_negative_examples_are_explicitly_marked() -> None:
    root = ET.fromstring(f"<root>{build_system_prompt(OPEN_MOMENT)}</root>")
    negatives = [e for e in root.iter("example") if e.get("type") == "negative"]
    assert negatives
    assert all(example.get("label", "").startswith("NÃO FAÇA ISSO") for example in negatives)


def test_store_contacts_use_the_resolved_values() -> None:
    prompt = build_system_prompt(OPEN_MOMENT)
    assert "contato@emporiodamusica.com.br" in prompt
    assert "empóriodamusica" not in prompt
