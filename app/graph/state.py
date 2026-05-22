from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    question: str
    conversation_id: Optional[str]
    documents: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    retry_count: int
    max_retries: int
    next_action: str
    final_answer: str
