from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.nodes import end_node, rag_node, router_node, tools_node
from app.graph.state import AgentState


def route_after_router(state: AgentState) -> str:
    return state.get("next_action", "rag")


def route_after_rag(state: AgentState) -> str:
    documents = state.get("documents", [])
    if documents:
        return "end"
    retry_count = state.get("retry_count", 0) + 1
    state["retry_count"] = retry_count
    if retry_count >= state.get("max_retries", 3):
        return "end"
    return "tools"


def route_after_tools(state: AgentState) -> str:
    tool_calls = state.get("tool_calls", [])
    if tool_calls:
        return "end"
    retry_count = state.get("retry_count", 0) + 1
    state["retry_count"] = retry_count
    if retry_count >= state.get("max_retries", 3):
        return "end"
    return "rag"


def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("end", end_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag": "rag",
            "tools": "tools",
            "end": "end",
        },
    )

    workflow.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "end": "end",
            "tools": "tools",
        },
    )

    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "end": "end",
            "rag": "rag",
        },
    )

    workflow.add_edge("end", END)

    return workflow.compile()
