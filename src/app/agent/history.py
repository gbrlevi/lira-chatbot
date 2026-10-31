"""Limits what the model sees of a long conversation. The checkpoint keeps everything."""

from langchain_core.messages import BaseMessage, HumanMessage

MAX_TURNS_IN_CONTEXT = 12


def select_window(
    messages: list[BaseMessage], max_turns: int = MAX_TURNS_IN_CONTEXT
) -> list[BaseMessage]:
    """Keep the most recent turns, counted from each customer message.

    Cutting only at customer messages keeps every tool call together with its result, which
    providers reject when separated.
    """
    turn_starts = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    if len(turn_starts) <= max_turns:
        return messages
    return messages[turn_starts[-max_turns] :]
