from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import Anthropic

from app.config import settings
from app.graph.state import AgentState
from app.mcp.client import execute_mcp_tools
from app.rag.store import get_vector_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful technical Q&A assistant. Your job is to answer technical questions accurately.

You have access to the following capabilities:
1. A knowledge base (RAG) that contains internal technical documentation.
2. External tools (MCP) such as GitHub search and web fetch.

When answering:
- If the internal knowledge base has relevant information, prefer and cite it.
- If the answer is not in the knowledge base, use external tools.
- Always cite your sources.
- Be concise and accurate.
"""


def router_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count >= max_retries:
        logger.warning("Max retries reached, proceeding to generate final answer")
        return {"next_action": "end"}

    message = {
        "role": "system",
        "content": (
            "Analyze the user's question and determine the next step. "
            "Respond with ONLY one word:\n"
            "- 'rag' if the question likely needs internal knowledge base search\n"
            "- 'tools' if the question requires external tools (GitHub, web search)\n"
            "- 'end' if the question is simple and can be answered directly\n"
            f"Question: {question}"
        ),
    }

    try:
        client = Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=50,
            messages=[message],
        )
        action = response.content[0].text.strip().lower()
        logger.info("Router decided action: %s", action)

        valid_actions = {"rag", "tools", "end"}
        if action not in valid_actions:
            action = "rag"

        return {"next_action": action}
    except Exception as e:
        logger.error("Router node failed: %s", e)
        return {"next_action": "rag"}


def rag_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    if documents:
        return {}

    try:
        store = get_vector_store()
        results = store.similarity_search_with_score(question, k=5)

        docs = []
        for doc, score in results:
            docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            })

        logger.info("RAG found %d documents", len(docs))
        return {"documents": docs}
    except Exception as e:
        logger.error("RAG node failed: %s", e)
        return {"documents": []}


def tools_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]

    try:
        client = Anthropic(api_key=settings.anthropic_api_key)

        tools = [
            {
                "name": "github_search",
                "description": "Search GitHub repositories for code and documentation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "web_fetch",
                "description": "Fetch content from a URL",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to fetch"},
                    },
                    "required": ["url"],
                },
            },
        ]

        message = {
            "role": "user",
            "content": (
                f"Use available tools to find information that helps answer: {question}\n"
                "Call the most appropriate tool(s) to gather information."
            ),
        }

        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[message],
            tools=tools,
        )

        tool_calls = []
        executed_results = []

        for block in response.content:
            if block.type == "tool_use":
                tool_call = {
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
                tool_calls.append(tool_call)

                result = execute_mcp_tools(block.name, block.input)
                executed_results.append({
                    "name": block.name,
                    "input": block.input,
                    "result": result,
                })

        logger.info("Tools node executed %d tool calls", len(tool_calls))
        return {"tool_calls": tool_calls + state.get("tool_calls", [])}

    except Exception as e:
        logger.error("Tools node failed: %s", e)
        return {}


def end_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    tool_results = state.get("tool_calls", [])

    context_parts = []

    if documents:
        context_parts.append("## Internal Knowledge Base\n")
        for i, doc in enumerate(documents[:3], 1):
            context_parts.append(f"{i}. {doc['content'][:500]}")

    if tool_results:
        context_parts.append("\n## External Tool Results\n")
        for tc in tool_results:
            context_parts.append(
                f"- {tc.get('name', 'tool')}: {json.dumps(tc, ensure_ascii=False)[:500]}"
            )

    context = "\n".join(context_parts) if context_parts else "No additional context available."

    messages = [
        {
            "role": "user",
            "content": (
                f"Answer the user's question based on the available context.\n\n"
                f"## Question\n{question}\n\n"
                f"## Context\n{context}\n\n"
                "Provide a clear, accurate answer. Cite sources where applicable."
            ),
        }
    ]

    try:
        client = Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        answer = response.content[0].text
        return {"final_answer": answer}
    except Exception as e:
        logger.error("End node failed: %s", e)
        return {"final_answer": f"抱歉，处理请求时发生错误: {str(e)}"}
