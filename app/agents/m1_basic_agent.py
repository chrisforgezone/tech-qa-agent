from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import Anthropic

from app.config import settings

logger = logging.getLogger(__name__)


def calculate(expression: str) -> str:
    allowed = "0123456789+-*/(). "
    if not all(c in allowed for c in expression):
        return "Error: invalid characters"

    try:
        return str(eval(expression))
    except Exception:
        return f"Error: cannot evaluate '{expression}'"


def get_current_time() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


TOOL_FUNCTIONS: dict[str, Any] = {
    "calculate": calculate,
    "get_current_time": get_current_time,
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "calculate",
        "description": "Evaluate a math expression. Use this when the user asks math questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression like '2+3*4'",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current server time in ISO format.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def run_agent(user_input: str, max_steps: int = 10) -> str:
    client = Anthropic(api_key=settings.anthropic_api_key)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_input}]

    for step in range(max_steps):
        logger.info("Step %d/%d: Calling Claude...", step + 1, max_steps)

        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            system="You are a helpful technical assistant.",
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return text

        if response.stop_reason == "tool_use":
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "tool_use":
                    logger.info("Tool call: %s(%s)", block.name, json.dumps(block.input))
                    func = TOOL_FUNCTIONS[block.name]
                    result = func(**block.input)
                    logger.info("Result: %s", result)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        }
                    )
            messages.append({"role": "user", "content": tool_results})

    return "Max steps reached"


if __name__ == "__main__":
    result = run_agent("现在几点？另外帮我算 (123 + 456) * 789")
    logger.info("Final answer: %s", result)
