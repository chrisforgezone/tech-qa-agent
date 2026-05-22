from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000, description="用户问题")
    conversation_id: Optional[str] = Field(None, description="会话 ID，用于多轮对话")
    max_retries: int = Field(3, ge=1, le=10, description="最大重试次数")


class SourceDocument(BaseModel):
    content: str = Field(..., description="文档片段内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    score: float = Field(0.0, description="相似度分数")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI 回答")
    sources: list[SourceDocument] = Field(default_factory=list, description="引用来源")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="工具调用记录")
    conversation_id: Optional[str] = Field(None, description="会话 ID")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="详细错误描述")
