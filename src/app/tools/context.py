"""What the tools need at runtime, passed explicitly so tests can run them offline."""

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from app.retrieval.index import HybridIndex
from app.runtime_store import RuntimeStore


@dataclass(frozen=True)
class ToolContext:
    store_path: Path
    index: HybridIndex
    runtime: RuntimeStore


def to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def thread_id_of(config: RunnableConfig) -> str:
    return str(config.get("configurable", {}).get("thread_id", "sem-conversa"))
