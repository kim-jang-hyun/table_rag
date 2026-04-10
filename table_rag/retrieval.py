"""Hybrid vector search with optional cross-encoder reranking."""

import logging
import time
from typing import List, Tuple

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from .config import DEFAULT_COLLECTION
from .models import (
    collection_is_hybrid,
    get_qdrant_client,
    is_fastembed_available,
    load_embed_model,
    load_reranker,
    load_sparse_embedder,
)

_logger = logging.getLogger("table_rag.perf")


def search_and_rerank(
    *,
    query: str,
    collection: str = DEFAULT_COLLECTION,
    qdrant_top_k: int = 20,
    rerank_top_k: int = 5,
    use_reranker: bool = True,
    use_hybrid: bool = True,
    embed_model=None,
    reranker=None,
    sparse_embedder=None,
    qdrant=None,
) -> Tuple[List[dict], str]:
    """Search a Qdrant collection and optionally rerank results with a cross-encoder.

    Args:
        query:          User query string.
        collection:     Qdrant collection name.
        qdrant_top_k:   Number of candidate chunks retrieved from Qdrant before
                        reranking.
        rerank_top_k:   Number of chunks returned after reranking (or direct
                        vector-score cutoff when *use_reranker* is ``False``).
        use_reranker:   Apply BGE cross-encoder reranking when ``True``.
        use_hybrid:     Use BM25 + dense hybrid retrieval (RRF) when ``True``
                        and the collection supports it.
        embed_model:    Pre-loaded :class:`HuggingFaceEmbeddings` (optional).
        reranker:       Pre-loaded :class:`HuggingFaceCrossEncoder` (optional).
        sparse_embedder: Unused; kept for API compatibility.
        qdrant:         Pre-loaded :class:`QdrantClient` (optional).

    Returns:
        A ``(docs, hybrid_warning)`` tuple where:

        - *docs* is a list of result dicts with keys:
          ``score``, ``rerank_score``, ``doc``, ``page``, ``chunk_id``,
          ``source_type``, ``text``.
        - *hybrid_warning* is a non-empty string when hybrid was requested
          but could not be used (e.g. fastembed missing, or dense-only collection).
    """
    if embed_model is None:
        embed_model = load_embed_model()
    if qdrant is None:
        qdrant = get_qdrant_client()

    hybrid_warning = ""
    is_hybrid_col = collection_is_hybrid(qdrant, collection)

    if use_hybrid and is_hybrid_col and is_fastembed_available():
        retrieval_mode = RetrievalMode.HYBRID
        sparse = load_sparse_embedder()
    elif use_hybrid and is_hybrid_col and not is_fastembed_available():
        hybrid_warning = (
            "하이브리드 collection이지만 fastembed가 없어 BM25 없이 벡터 검색만 했습니다. "
            "`pip install fastembed` 하거나 Python 3.10~3.12 가상환경을 쓰면 RRF 하이브리드를 쓸 수 있습니다."
        )
        retrieval_mode = RetrievalMode.DENSE
        sparse = None
    elif use_hybrid and not is_hybrid_col:
        hybrid_warning = (
            "하이브리드 검색을 켰지만 이 collection은 BM25 sparse 없이 인덱싱되었습니다. "
            "벡터 검색만 사용했습니다. 하이브리드 인덱싱을 켠 뒤 다시 인덱싱하세요."
        )
        retrieval_mode = RetrievalMode.DENSE
        sparse = None
    else:
        retrieval_mode = RetrievalMode.DENSE
        sparse = None

    vs_kwargs: dict = dict(
        client=qdrant,
        collection_name=collection,
        embedding=embed_model,
        retrieval_mode=retrieval_mode,
    )
    if sparse is not None:
        vs_kwargs["sparse_embedding"] = sparse

    vectorstore = QdrantVectorStore(**vs_kwargs)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": qdrant_top_k})

    if use_reranker:
        if reranker is None:
            reranker = load_reranker()
        compressor = CrossEncoderReranker(model=reranker, top_n=rerank_top_k)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    else:
        retriever = base_retriever

    _logger.info(
        f"[search] 시작 (top_k={qdrant_top_k}, reranker={use_reranker}, hybrid={retrieval_mode})"
    )
    t0 = time.time()
    results = retriever.invoke(query)
    _logger.info(f"[search] 완료 ({time.time() - t0:.1f}s) — 결과 {len(results)}개")

    docs = []
    for doc in results:
        meta = doc.metadata or {}
        rerank_score = float(meta.get("relevance_score", meta.get("_relevance_score", 0.0)))
        docs.append(
            {
                "score": float(meta.get("_relevance_score", 0.0)),
                "rerank_score": rerank_score,
                "doc": meta.get("doc", ""),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
                "source_type": meta.get("source_type", "text"),
                "text": doc.page_content,
            }
        )

    if not use_reranker:
        docs = docs[:rerank_top_k]

    return docs, hybrid_warning
