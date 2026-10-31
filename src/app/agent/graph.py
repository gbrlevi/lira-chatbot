"""The agent graph: the model and its tools in an explicit cycle, between code guardrails.

START -> input_guard -> agent <-> tools
                          |
                          v
                     output_guard -> agent (one regeneration) | store_hours_notice -> END
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError

from app.agent import guards
from app.agent.history import select_window
from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.domain import clock
from app.domain.pricing import format_brl

NODE_INPUT_GUARD = "input_guard"
NODE_AGENT = "agent"
NODE_TOOLS = "tools"
NODE_OUTPUT_GUARD = "output_guard"
NODE_STORE_HOURS = "store_hours_notice"

# Enough for several tools per turn, while stopping a model stuck in a loop.
RECURSION_LIMIT = 25
MAX_REGENERATIONS = 1


def _handle_tool_error(error: Exception) -> str:
    """Invalid arguments can be corrected by the model; anything else must not leak details."""
    if isinstance(error, ToolInvocationError):
        return f"Parâmetros inválidos para a ferramenta: {error}. Corrija e tente de novo."
    return (
        "Não foi possível consultar agora. Avise o cliente, ofereça os contatos da loja e "
        "não responda de memória."
    )


def _input_guard(state: AgentState) -> dict:
    update: dict = {"regenerations": 0, "correction": None}
    reply = guards.rejected_input_reply(str(state["messages"][-1].content))
    if reply:
        update["messages"] = [AIMessage(reply)]
    return update


def _route_after_input(state: AgentState) -> str:
    return END if isinstance(state["messages"][-1], AIMessage) else NODE_AGENT


def _route_after_agent(state: AgentState) -> str:
    return NODE_TOOLS if state["messages"][-1].tool_calls else NODE_OUTPUT_GUARD


def _output_guard(state: AgentState) -> dict:
    """Fail closed: an answer that cannot be verified never reaches the customer."""
    answer = state["messages"][-1]
    text = answer.text
    unverified = guards.unverified_amounts(text, state["messages"])
    if text.strip() and not unverified:
        return {"correction": None}

    remove = RemoveMessage(id=answer.id)
    if state.get("regenerations", 0) >= MAX_REGENERATIONS:
        return {"messages": [remove, AIMessage(guards.SAFE_ANSWER)], "correction": None}

    correction = guards.EMPTY_ANSWER_CORRECTION
    if unverified:
        amounts = ", ".join(format_brl(amount) for amount in sorted(unverified))
        correction = guards.UNVERIFIED_AMOUNT_CORRECTION.format(amounts=amounts)
    return {
        "messages": [remove],
        "regenerations": state.get("regenerations", 0) + 1,
        "correction": correction,
    }


def _route_after_output_guard(state: AgentState) -> str:
    return NODE_AGENT if state.get("correction") else NODE_STORE_HOURS


def _store_hours_notice(state: AgentState) -> dict:
    """Policy 7.1 asks for the return time. A fixed rule, so code applies it, not the model."""
    moment = clock.now()
    first_reply = sum(isinstance(m, HumanMessage) for m in state["messages"]) == 1
    if clock.is_open(moment) or not first_reply:
        return {}
    answer = state["messages"][-1]
    notice = clock.team_return_notice(moment)
    return {"messages": [AIMessage(content=f"{answer.text}\n\n{notice}", id=answer.id)]}


def build_graph(model: BaseChatModel, tools: Sequence[BaseTool], checkpointer: Any = None) -> Any:
    model_with_tools = model.bind_tools(list(tools))

    def agent(state: AgentState) -> dict:
        prompt = build_system_prompt()
        if state.get("correction"):
            prompt = f"{prompt}\n\n<correcao>{state['correction']}</correcao>"
        window = select_window(state["messages"])
        return {"messages": [model_with_tools.invoke([SystemMessage(prompt), *window])]}

    workflow = StateGraph(AgentState)
    workflow.add_node(NODE_INPUT_GUARD, _input_guard)
    workflow.add_node(NODE_AGENT, agent)
    workflow.add_node(NODE_TOOLS, ToolNode(tools, handle_tool_errors=_handle_tool_error))
    workflow.add_node(NODE_OUTPUT_GUARD, _output_guard)
    workflow.add_node(NODE_STORE_HOURS, _store_hours_notice)

    workflow.add_edge(START, NODE_INPUT_GUARD)
    workflow.add_conditional_edges(NODE_INPUT_GUARD, _route_after_input, [NODE_AGENT, END])
    workflow.add_conditional_edges(NODE_AGENT, _route_after_agent, [NODE_TOOLS, NODE_OUTPUT_GUARD])
    workflow.add_edge(NODE_TOOLS, NODE_AGENT)
    workflow.add_conditional_edges(
        NODE_OUTPUT_GUARD, _route_after_output_guard, [NODE_AGENT, NODE_STORE_HOURS]
    )
    workflow.add_edge(NODE_STORE_HOURS, END)
    return workflow.compile(checkpointer=checkpointer)
