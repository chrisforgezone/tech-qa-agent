from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.api.types import QueryResult
from langchain_anthropic import AnthropicEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)

_store: Optional["VectorStore"] = None


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.persist_dir)
        self._embedding_function = AnthropicEmbeddings(
            api_key=settings.anthropic_api_key,
            model="claude-sonnet-4-20250514",
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self, documents: list[str], metadatas: Optional[list[dict]] = None, ids: Optional[list[str]] = None
    ) -> None:
        if not documents:
            return

        embeddings = self._embedding_function.embed_documents(documents)

        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in documents]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("Added %d documents to collection", len(documents))

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple["Document", float]]:
        query_embedding = self._embedding_function.embed_query(query)

        results: QueryResult = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        if results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = (
                    results["metadatas"][0][i]
                    if results["metadatas"] and results["metadatas"][0]
                    else {}
                )
                distance = (
                    results["distances"][0][i]
                    if results["distances"] and results["distances"][0]
                    else 0.0
                )
                docs.append((Document(doc_text, metadata), distance))

        return docs

    def count(self) -> int:
        return self._collection.count()


class Document:
    def __init__(self, page_content: str, metadata: dict) -> None:
        self.page_content = page_content
        self.metadata = metadata


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
