"""Chunk dataclass and singleton model factory functions.

All model instances are cached at process level (module-level singletons)
so they are loaded once and reused across calls.
"""

import os
from dataclasses import dataclass

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from .config import DEFAULT_OPENAI_MODEL, DEFAULT_SPARSE_MODEL


@dataclass(frozen=True)
class Chunk:
    """A text or table chunk extracted from a document.

    Attributes:
        chunk_id:    Unique identifier, e.g. ``"doc_slug::p0001_tb001"``.
        page:        1-based page (or slide) number.
        text:        Chunk content as a plain string.
        source_type: ``"text"`` for plain-text chunks, ``"table"`` for table chunks.
    """

    chunk_id: str
    page: int
    text: str
    source_type: str


# ── Process-level singleton instances ────────────────────────────────────────

_qdrant_client_singleton: QdrantClient | None = None
_sparse_embedder_singleton = None
_embed_model_singleton: HuggingFaceEmbeddings | None = None
_reranker_singleton: HuggingFaceCrossEncoder | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a cached :class:`QdrantClient` using ``QDRANT_URL`` / ``QDRANT_API_KEY``."""
    global _qdrant_client_singleton
    if _qdrant_client_singleton is None:
        url = os.environ.get("QDRANT_URL", "").strip()
        api_key = os.environ.get("QDRANT_API_KEY", "").strip()
        if not url:
            raise RuntimeError("Missing QDRANT_URL (set in .env)")
        if not api_key or api_key == "PASTE_YOUR_QDRANT_API_KEY_HERE":
            raise RuntimeError("Missing QDRANT_API_KEY (set in .env)")
        _qdrant_client_singleton = QdrantClient(url=url, api_key=api_key)
    return _qdrant_client_singleton


def load_sparse_embedder():
    """Return a cached BM25 sparse embedder (requires ``fastembed``)."""
    global _sparse_embedder_singleton
    if _sparse_embedder_singleton is None:
        from langchain_qdrant import FastEmbedSparse

        _sparse_embedder_singleton = FastEmbedSparse(model_name=DEFAULT_SPARSE_MODEL)
    return _sparse_embedder_singleton


def load_embed_model() -> HuggingFaceEmbeddings:
    """Return a cached BGE-M3 dense embedding model."""
    global _embed_model_singleton
    if _embed_model_singleton is None:
        _embed_model_singleton = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embed_model_singleton


def load_reranker() -> HuggingFaceCrossEncoder:
    """Return a cached BGE cross-encoder reranker."""
    global _reranker_singleton
    if _reranker_singleton is None:
        _reranker_singleton = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    return _reranker_singleton


def get_llm(model: str = DEFAULT_OPENAI_MODEL) -> ChatOpenAI:
    """Return a :class:`ChatOpenAI` instance using ``OPENAI_API_KEY``."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set in .env)")
    return ChatOpenAI(model=model, temperature=0.1, api_key=api_key)


def is_fastembed_available() -> bool:
    """Return True when ``fastembed`` can be imported (required for BM25 hybrid)."""
    try:
        import fastembed  # noqa: F401

        return True
    except ImportError:
        return False


def collection_is_hybrid(qdrant: QdrantClient, collection: str) -> bool:
    """Return True when the given Qdrant collection has sparse (BM25) vectors."""
    info = qdrant.get_collection(collection_name=collection)
    params = info.config.params
    sparse = getattr(params, "sparse_vectors", None) or {}
    return bool(sparse)
