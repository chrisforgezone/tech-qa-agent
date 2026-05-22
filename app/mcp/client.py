from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


def execute_mcp_tools(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name == "github_search":
        return _github_search(tool_input.get("query", ""))
    elif tool_name == "web_fetch":
        return _web_fetch(tool_input.get("url", ""))
    else:
        logger.warning("Unknown tool: %s", tool_name)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _github_search(query: str) -> str:
    if not query:
        return json.dumps({"error": "Missing search query"})

    url = f"https://api.github.com/search/repositories?q={quote(query)}&per_page=5"
    try:
        response = httpx.get(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "tech-qa-agent",
            },
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()

        items = []
        for item in data.get("items", [])[:5]:
            items.append({
                "name": item.get("full_name"),
                "description": item.get("description"),
                "url": item.get("html_url"),
                "stars": item.get("stargazers_count"),
            })

        return json.dumps(items, ensure_ascii=False)
    except Exception as e:
        logger.error("GitHub search failed: %s", e)
        return json.dumps({"error": str(e)})


def _web_fetch(url: str) -> str:
    if not url:
        return json.dumps({"error": "Missing URL"})

    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            text = response.text[:5000]
            return json.dumps({"content": text, "url": url}, ensure_ascii=False)
        return json.dumps({"content": response.text[:5000], "url": url}, ensure_ascii=False)
    except Exception as e:
        logger.error("Web fetch failed: %s", e)
        return json.dumps({"error": str(e), "url": url})


class MCPClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._client = httpx.AsyncClient(timeout=30.0)

    async def list_tools(self) -> list[dict]:
        try:
            response = await self._client.get(f"{self.endpoint}/tools/list")
            response.raise_for_status()
            return response.json().get("tools", [])
        except Exception:
            return []

    async def call_tool(self, name: str, arguments: dict) -> str:
        try:
            response = await self._client.post(
                f"{self.endpoint}/tools/call",
                json={"name": name, "arguments": arguments},
            )
            response.raise_for_status()
            return json.dumps(response.json(), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def close(self) -> None:
        await self._client.aclose()
