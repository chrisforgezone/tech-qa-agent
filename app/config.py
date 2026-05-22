import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "tech_qa_knowledge")

    data_docs_dir: str = os.getenv("DATA_DOCS_DIR", "data/docs")

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")

    @property
    def persist_dir(self) -> str:
        return os.path.join(Path(__file__).parent.parent, self.chroma_persist_dir)


settings = Settings()
