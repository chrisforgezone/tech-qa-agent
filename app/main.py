from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.graph.workflow import create_workflow
from app.graph.state import AgentState
from app.models import ChatRequest, ChatResponse, ErrorResponse, SourceDocument

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_workflow = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = create_workflow()
    return _workflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting tech-qa-agent on %s:%s", settings.host, settings.port)
    get_workflow()
    yield
    logger.info("Shutting down tech-qa-agent")


app = FastAPI(
    title="Tech QA Agent",
    description="Production-grade AI assistant with RAG, MCP tools, and LangGraph orchestration",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Chat with the QA agent",
    description="Send a technical question and get an AI-powered response with source citations.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ANTHROPIC_API_KEY not configured",
        )

    conversation_id = request.conversation_id or str(uuid.uuid4())

    initial_state: AgentState = {
        "messages": [],
        "question": request.question,
        "conversation_id": conversation_id,
        "documents": [],
        "tool_calls": [],
        "retry_count": 0,
        "max_retries": request.max_retries,
        "next_action": "",
        "final_answer": "",
    }

    try:
        workflow = get_workflow()
        result = workflow.invoke(initial_state, config={"recursion_limit": 10})

        sources = []
        for doc in result.get("documents", []):
            sources.append(
                SourceDocument(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=doc.get("score", 0.0),
                )
            )

        tool_calls = result.get("tool_calls", [])

        return ChatResponse(
            answer=result.get("final_answer", "No answer generated"),
            sources=sources,
            tool_calls=tool_calls,
            conversation_id=conversation_id,
        )

    except Exception as e:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/health", summary="Health check")
async def health() -> dict[str, Any]:
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=True,
    )
