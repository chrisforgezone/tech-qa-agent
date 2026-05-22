## 项目概述
构建一个生产级的 AI 助手，基于内部知识库+大模型，能回答技术问题。

## 项目目标
先查内部知识库（RAG）——找到答案就直接返回
找不到就调用 MCP 工具（如 GitHub 搜索、web fetch）
复杂问题自动拆解多步处理
全部以 REST API 暴露，可容器化部署

## 整体架构
┌──────────────────────────────────────────────────────────┐
│  Docker Container                                         │
│                                                            │
│  ┌─────────────┐                                          │
│  │  FastAPI    │  ← REST API 入口（POST /chat）          │
│  └──────┬──────┘                                          │
│         │                                                  │
│  ┌──────▼──────────────────────────────────┐             │
│  │  LangGraph StateGraph                    │             │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌─────┐ │             │
│  │  │router│→ │ rag  │→ │tools │→ │ end │ │             │
│  │  └──────┘  └──────┘  └──────┘  └─────┘ │             │
│  └─────┬──────────┬──────────┬─────────────┘             │
│        │          │          │                            │
│   ┌────▼───┐ ┌────▼───┐ ┌────▼────┐                      │
│   │Anthropic│ │ Chroma │ │  MCP   │                      │
│   │  SDK    │ │   DB   │ │ Server │                      │
│   └─────────┘ └────────┘ └────────┘                      │
└──────────────────────────────────────────────────────────┘
每个组件的职责：
组件职责类比 Java 概念FastAPIHTTP 入口、请求验证Spring Boot ControllerLangGraph编排 Agent 工作流Spring StateMachineAnthropic SDKLLM 调用OkHttp 客户端Chroma向量存储 + 检索ElasticsearchMCP Server标准化工具协议gRPC 服务Docker部署封装不用解释

## 项目结构
tech-qa-agent/
├── pyproject.toml          # 依赖管理（类比 pom.xml）
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI 入口
│   ├── config.py          # 配置（类比 application.yml）
│   ├── models.py          # Pydantic 数据模型
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py       # LangGraph 状态定义
│   │   ├── nodes.py       # 各个节点的逻辑
│   │   └── workflow.py    # 工作流编排
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── store.py       # Chroma 封装
│   │   └── ingest.py      # 文档导入脚本
│   └── mcp/
│       ├── __init__.py
│       ├── client.py      # MCP 客户端
│       └── server.py      # 自定义 MCP Server
├── data/
│   └── docs/              # 知识库源文件
└── tests/
    └── test_agent.py

## 分阶段实现路线
我把整个项目拆成 6 个里程碑，每个里程碑都能独立运行、独立验证。先把第一个跑通再做下一个，不要一次性全写完。
里程碑总览
M1 (Day 1):  Anthropic SDK 基础 Agent (单文件，无依赖)
M2 (Day 2):  Chroma RAG 知识库 (能查文档)
M3 (Day 3):  LangGraph 工作流编排 (节点 + 状态机)
M4 (Day 4):  MCP 集成 (接入标准化工具)
M5 (Day 5):  FastAPI 暴露 REST API
M6 (Day 6):  Docker 容器化部署