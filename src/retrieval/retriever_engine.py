from __future__ import annotations

from pathlib import Path

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS, Qdrant
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.core.config import settings
from src.core.logging import get_logger
from src.pipeline.artifact_store import ChunkArtifactStore
from src.retrieval.base import BaseRetrieverEngine


class HybridRetrieverEngine(BaseRetrieverEngine):
    def __init__(self) -> None:
        self.logger = get_logger("finledger.retrieval.engine")
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for retrieval.")
        self.embeddings = OpenAIEmbeddings(model=settings.openai_embed_model, api_key=settings.openai_api_key)
        self.chunk_store = ChunkArtifactStore()
        self.parent_docs = self.chunk_store.load(settings.parent_chunks_path)
        self.parent_lookup = self._build_parent_lookup(self.parent_docs)
        self.base_retriever = self._build_hybrid_retriever()

    def retrieve(self, query: str) -> list[Document]:
        try:
            if settings.retrieval_mode == "contextual_compression":
                retriever = self._build_compression_retriever()
                docs = retriever.invoke(query)
            else:
                docs = self.base_retriever.invoke(query)
                docs = self._expand_to_parent_context(docs)
            deduplicated = self._deduplicate_documents(docs)
            result = deduplicated[: settings.max_contexts]
            self.logger.info(
                "retrieval_completed",
                extra={"query_length": len(query), "documents_count": len(result), "mode": settings.retrieval_mode},
            )
            return result
        except Exception as exc:
            self.logger.exception("retrieval_failed", extra={"error": str(exc), "mode": settings.retrieval_mode})
            raise

    def _build_hybrid_retriever(self) -> EnsembleRetriever:
        vectorstore = self._load_vectorstore()
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
        keyword_docs = self.chunk_store.load(settings.child_chunks_path)
        if not keyword_docs:
            keyword_docs = self.parent_docs
        bm25_retriever = BM25Retriever.from_documents(keyword_docs)
        bm25_retriever.k = settings.top_k
        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[settings.retrieval_vector_weight, settings.retrieval_keyword_weight],
        )

    def _build_compression_retriever(self) -> ContextualCompressionRetriever:
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            temperature=0.0,
        )
        compressor = LLMChainExtractor.from_llm(llm=llm)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=self.base_retriever)

    def _load_vectorstore(self) -> Qdrant | FAISS:
        if settings.vector_backend == "qdrant":
            return Qdrant.from_existing_collection(
                embedding=self.embeddings,
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                collection_name=settings.qdrant_collection,
                https=settings.qdrant_use_https,
                api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
            )
        index_path = Path(settings.faiss_index_path)
        return FAISS.load_local(
            folder_path=str(index_path.parent),
            embeddings=self.embeddings,
            index_name=index_path.stem,
            allow_dangerous_deserialization=True,
        )

    def _build_parent_lookup(self, parent_docs: list[Document]) -> dict[str, Document]:
        lookup: dict[str, Document] = {}
        for document in parent_docs:
            key = self._parent_key(
                str(document.metadata.get("record_id", "")),
                int(document.metadata.get("parent_chunk_index", -1)),
            )
            lookup[key] = document
        return lookup

    def _expand_to_parent_context(self, docs: list[Document]) -> list[Document]:
        expanded: list[Document] = []
        for doc in docs:
            record_id = str(doc.metadata.get("record_id", ""))
            parent_index = int(doc.metadata.get("parent_chunk_index", -1))
            if record_id and parent_index >= 0:
                key = self._parent_key(record_id, parent_index)
                parent_doc = self.parent_lookup.get(key)
                if parent_doc is not None:
                    expanded.append(parent_doc)
                    continue
            expanded.append(doc)
        return expanded

    def _deduplicate_documents(self, docs: list[Document]) -> list[Document]:
        unique: dict[str, Document] = {}
        for doc in docs:
            record_id = str(doc.metadata.get("record_id", "unknown"))
            parent_index = int(doc.metadata.get("parent_chunk_index", -1))
            child_index = int(doc.metadata.get("child_chunk_index", -1))
            key = f"{record_id}:{parent_index}:{child_index}:{len(doc.page_content)}"
            unique[key] = doc
        return list(unique.values())

    def _parent_key(self, record_id: str, parent_chunk_index: int) -> str:
        return f"{record_id}:{parent_chunk_index}"
