from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.core.config import settings
from src.core.logging import get_logger
from src.retrieval.base import BaseRAGEngine
from src.retrieval.retriever_engine import HybridRetrieverEngine


class FinancialRAGService(BaseRAGEngine):
    def __init__(self) -> None:
        self.logger = get_logger("finledger.retrieval.rag_service")
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for RAG service.")
        self.retriever = HybridRetrieverEngine()
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            temperature=settings.temperature,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are FinLedger.AI, an expert financial analysis assistant. "
                    "Answer only using the provided context. "
                    "If context is insufficient, say that clearly. "
                    "Each material claim must cite one or more context chunk ids in square brackets like [C1]. "
                    "When calculations are needed, show concise calculation steps and cite supporting chunks.",
                ),
                (
                    "human",
                    "Question:\n{question}\n\nContext:\n{context}\n\n"
                    "Return a concise financial answer with citations.",
                ),
            ]
        )

    def answer(self, question: str) -> dict[str, Any]:
        try:
            docs = self.retriever.retrieve(question)
            context = self._format_context(docs)
            chain = self.prompt | self.llm
            response = chain.invoke({"question": question, "context": context})
            result = {
                "question": question,
                "answer": response.content,
                "mode": settings.retrieval_mode,
                "sources": self._serialize_sources(docs),
            }
            self.logger.info(
                "rag_answer_generated",
                extra={"question_length": len(question), "sources_count": len(result["sources"])},
            )
            return result
        except Exception as exc:
            self.logger.exception("rag_answer_failed", extra={"error": str(exc)})
            raise

    def _format_context(self, docs: list[Document]) -> str:
        blocks: list[str] = []
        for index, doc in enumerate(docs, start=1):
            block = f"[C{index}] {doc.page_content}"
            blocks.append(block)
        return "\n\n".join(blocks)

    def _serialize_sources(self, docs: list[Document]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for index, doc in enumerate(docs, start=1):
            payload.append(
                {
                    "chunk_id": f"C{index}",
                    "record_id": doc.metadata.get("record_id"),
                    "source_file": doc.metadata.get("source_file"),
                    "parent_chunk_index": doc.metadata.get("parent_chunk_index"),
                    "child_chunk_index": doc.metadata.get("child_chunk_index"),
                    "chunk_level": doc.metadata.get("chunk_level"),
                    "content": doc.page_content,
                }
            )
        return payload
