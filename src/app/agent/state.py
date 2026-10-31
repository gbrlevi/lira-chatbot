"""State carried through the agent graph."""

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    # Answers discarded by the output guard during the current turn.
    regenerations: int
    # Instruction for the next model call, after a discarded answer.
    correction: str | None
