from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

TOOLS_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(
    name: str,
    description: str,
    input_schema: dict,
    handler: callable,
) -> None:
    TOOLS_REGISTRY[name] = {
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "handler": handler,
    }


def list_tools() -> list[dict]:
    return [
        {
            "name": v["name"],
            "description": v["description"],
            "input_schema": v["input_schema"],
        }
        for v in TOOLS_REGISTRY.values()
    ]


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    tool = TOOLS_REGISTRY.get(name)
    if not tool:
        return json.dumps({"error": f"Tool not found: {name}"})
    try:
        result = tool["handler"](**arguments)
        return result
    except Exception as e:
        logger.error("Tool %s execution failed: %s", name, e)
        return json.dumps({"error": str(e)})
