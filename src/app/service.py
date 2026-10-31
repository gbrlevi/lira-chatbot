"""The single entry point for conversations. CLI and API are thin layers over it."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from app.agent.graph import NODE_AGENT, NODE_TOOLS, RECURSION_LIMIT, build_graph
from app.agent.persistence import open_checkpointer
from app.config import get_settings
from app.llm import build_chat_model
from app.tools.registry import build_tools, default_context

logger = logging.getLogger(__name__)

UNAVAILABLE_REPLY = (
    "Desculpa, estou com uma instabilidade e não consegui consultar o sistema agora. Tenta de "
    "novo em instantes ou fala com a equipe pelo WhatsApp (67) 3321-4500."
)


class StoreNotBuiltError(RuntimeError):
    """The store database does not exist yet."""


@dataclass(frozen=True)
class ToolUse:
    name: str
    args: dict


@dataclass(frozen=True)
class ChatReply:
    text: str
    tools_used: list[ToolUse] = field(default_factory=list)
    degraded: bool = False


@dataclass(frozen=True)
class ChatEvent:
    """Progress of a turn. 'output' is only set when a tool finishes."""

    kind: str
    name: str
    args: dict | None = None
    output: str | None = None


EventHandler = Callable[[ChatEvent], None]


class ChatService:
    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def reply(self, thread_id: str, text: str, on_event: EventHandler | None = None) -> ChatReply:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        tools_used: list[ToolUse] = []
        try:
            updates = self._graph.stream(
                {"messages": [HumanMessage(text)]}, config=config, stream_mode="updates"
            )
            for update in updates:
                self._handle_update(update, tools_used, on_event)
            answer = self._graph.get_state(config).values["messages"][-1]
        # Last line of defense for the customer: whatever failed, they get a useful message.
        except Exception:
            logger.exception("Falha ao processar a mensagem da conversa %s", thread_id)
            return ChatReply(UNAVAILABLE_REPLY, tools_used, degraded=True)
        return ChatReply(answer.text, tools_used)

    @staticmethod
    def _handle_update(
        update: dict, tools_used: list[ToolUse], on_event: EventHandler | None
    ) -> None:
        for node, payload in update.items():
            messages = (payload or {}).get("messages", [])
            if node == NODE_AGENT:
                for call in messages[-1].tool_calls:
                    tools_used.append(ToolUse(call["name"], call["args"]))
                    if on_event:
                        on_event(ChatEvent("tool_start", call["name"], args=call["args"]))
            elif node == NODE_TOOLS and on_event:
                for message in messages:
                    if isinstance(message, ToolMessage):
                        on_event(ChatEvent("tool_end", message.name, output=str(message.content)))


def create_chat_service() -> ChatService:
    settings = get_settings()
    if not settings.store_db_path.exists():
        raise StoreNotBuiltError("Base não encontrada. Execute primeiro: uv run lira-ingest")
    tools = build_tools(default_context())
    checkpointer = open_checkpointer(settings.runtime_db_path)
    return ChatService(build_graph(build_chat_model(), tools, checkpointer=checkpointer))
