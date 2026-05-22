from __future__ import annotations

import os
import uuid

import pytest

os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "test-key")

from app.graph.state import AgentState
from app.models import ChatRequest, ChatResponse, SourceDocument


class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(question="What is Docker?")
        assert req.question == "What is Docker?"
        assert req.max_retries == 3
        assert req.conversation_id is None

    def test_missing_question(self):
        with pytest.raises(ValueError):
            ChatRequest()

    def test_empty_question(self):
        with pytest.raises(ValueError):
            ChatRequest(question="")

    def test_custom_max_retries(self):
        req = ChatRequest(question="test", max_retries=5)
        assert req.max_retries == 5

    def test_max_retries_out_of_range(self):
        with pytest.raises(ValueError):
            ChatRequest(question="test", max_retries=0)
        with pytest.raises(ValueError):
            ChatRequest(question="test", max_retries=11)


class TestChatResponse:
    def test_response_model(self):
        conv_id = str(uuid.uuid4())
        resp = ChatResponse(
            answer="Test answer",
            sources=[SourceDocument(content="source content", metadata={})],
            conversation_id=conv_id,
        )
        assert resp.answer == "Test answer"
        assert len(resp.sources) == 1
        assert resp.conversation_id == conv_id

    def test_response_json(self):
        resp = ChatResponse(
            answer="Hello",
            sources=[],
            tool_calls=[],
            conversation_id="abc",
        )
        data = resp.model_dump()
        assert data["answer"] == "Hello"
        assert data["sources"] == []
        assert data["conversation_id"] == "abc"


class TestAgentState:
    def test_initial_state(self):
        state: AgentState = {
            "messages": [],
            "question": "test",
            "conversation_id": "abc",
            "documents": [],
            "tool_calls": [],
            "retry_count": 0,
            "max_retries": 3,
            "next_action": "",
            "final_answer": "",
        }
        assert state["question"] == "test"
        assert state["retry_count"] == 0
        assert state["max_retries"] == 3


class TestHealthEndpoint:
    async def test_health_check(self):
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestChatEndpointValidation:
    def test_chat_missing_api_key(self):
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.post("/chat", json={"question": "test"})
        assert response.status_code in [500]


class TestConfig:
    def test_settings_defaults(self):
        from app.config import Settings

        s = Settings()
        assert s.llm_model is not None
        assert s.chroma_collection_name == "tech_qa_knowledge"


class TestVectorStore:
    def test_store_creation(self):
        from app.rag.store import VectorStore

        store = VectorStore()
        assert store.count() >= 0


class TestMCP:
    def test_github_search_no_query(self):
        from app.mcp.client import _github_search

        result = _github_search("")
        assert "error" in result.lower()

    def test_web_fetch_no_url(self):
        from app.mcp.client import _web_fetch

        result = _web_fetch("")
        assert "error" in result.lower()
