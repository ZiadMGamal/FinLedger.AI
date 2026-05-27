from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True)

    project_name: str = Field(default="FinLedger.AI", alias="PROJECT_NAME")
    project_env: Literal["development", "staging", "production"] = Field(default="development", alias="PROJECT_ENV")
    project_version: str = Field(default="0.1.0", alias="PROJECT_VERSION")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embed_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBED_MODEL")

    vector_backend: Literal["qdrant", "faiss"] = Field(default="qdrant", alias="VECTOR_BACKEND")
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(default="finledger_docfinqa", alias="QDRANT_COLLECTION")
    qdrant_use_https: bool = Field(default=False, alias="QDRANT_USE_HTTPS")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    faiss_index_path: str = Field(default="data/processed/faiss/docfinqa.index", alias="FAISS_INDEX_PATH")

    docfinqa_train_path: str = Field(default="data/raw/docfinqa_train.json", alias="DOCFINQA_TRAIN_PATH")
    docfinqa_validation_path: str = Field(default="data/raw/docfinqa_validation.json", alias="DOCFINQA_VALIDATION_PATH")
    docfinqa_test_path: str = Field(default="data/raw/docfinqa_test.json", alias="DOCFINQA_TEST_PATH")

    chunk_parent_size: int = Field(default=3000, alias="CHUNK_PARENT_SIZE")
    chunk_child_size: int = Field(default=600, alias="CHUNK_CHILD_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    max_contexts: int = Field(default=6, alias="MAX_CONTEXTS")
    top_k: int = Field(default=20, alias="TOP_K")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    retrieval_mode: Literal["parent_child", "contextual_compression"] = Field(default="parent_child", alias="RETRIEVAL_MODE")
    retrieval_vector_weight: float = Field(default=0.75, alias="RETRIEVAL_VECTOR_WEIGHT")
    retrieval_keyword_weight: float = Field(default=0.25, alias="RETRIEVAL_KEYWORD_WEIGHT")
    compression_top_n: int = Field(default=8, alias="COMPRESSION_TOP_N")
    parent_chunks_path: str = Field(default="data/processed/chunks/parent_chunks.jsonl", alias="PARENT_CHUNKS_PATH")
    child_chunks_path: str = Field(default="data/processed/chunks/child_chunks.jsonl", alias="CHILD_CHUNKS_PATH")
    evaluation_sample_size: int = Field(default=50, alias="EVALUATION_SAMPLE_SIZE")
    evaluation_output_path: str = Field(default="data/processed/evaluation/latest_report.json", alias="EVALUATION_OUTPUT_PATH")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")


settings = Settings()
Path(settings.evaluation_output_path).parent.mkdir(parents=True, exist_ok=True)
