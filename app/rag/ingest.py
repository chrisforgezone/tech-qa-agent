#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import List

from app.config import settings
from app.rag.store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind("。")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                end = start + break_point + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap if end < len(text) else end

    return chunks


def load_documents(docs_dir: str) -> list[dict]:
    documents = []
    path = Path(docs_dir)

    if not path.exists():
        logger.warning("Documents directory not found: %s", docs_dir)
        return documents

    supported_extensions = {".txt", ".md", ".rst", ".py", ".java", ".js", ".ts", ".go", ".yaml", ".yml", ".json"}

    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix in supported_extensions:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                continue

            chunks = split_text(content)
            for chunk in chunks:
                documents.append({
                    "content": chunk,
                    "metadata": {
                        "source": str(file_path.relative_to(path)),
                        "file_type": file_path.suffix,
                    },
                })
            logger.info("Loaded %s: %d chunks", file_path.name, len(chunks))

    return documents


def ingest_documents(docs_dir: str | None = None) -> int:
    docs_dir = docs_dir or os.path.join(Path(__file__).parent.parent.parent, settings.data_docs_dir)
    store = VectorStore()

    documents = load_documents(docs_dir)
    if not documents:
        logger.warning("No documents found to ingest")
        return 0

    texts = [d["content"] for d in documents]
    metadatas = [d["metadata"] for d in documents]
    ids = [str(uuid.uuid4()) for _ in documents]

    store.add_documents(texts, metadatas, ids)
    logger.info("Ingested %d documents", len(documents))
    return len(documents)


if __name__ == "__main__":
    count = ingest_documents()
    print(f"Ingested {count} documents")
