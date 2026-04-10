"""table_rag — Table-aware RAG pipeline for PDF/PPTX documents.

This package provides:

1. **Table utilities** (``table_rag.table``) — pure PDF/PPTX table parsing with
   no RAG dependencies; suitable for standalone use.
2. **Document loaders** (``table_rag.document``) — convert PDF/PPTX files into
   text and table :class:`Chunk` objects.
3. **Indexing** (``table_rag.indexing``) — ingest documents into a Qdrant
   vector store with optional BM25 hybrid mode.
4. **Retrieval** (``table_rag.retrieval``) — hybrid search + cross-encoder
   reranking against Qdrant.
5. **QA** (``table_rag.qa``) — LLM-based answer generation from retrieved chunks.

Quick start (full RAG pipeline)::

    from table_rag import (
        ingest_pdfs_to_qdrant,
        search_and_rerank,
        answer_with_openai,
    )
    from table_rag.models import get_llm

    ingest_pdfs_to_qdrant(pdf_paths=["report.pdf"], collection="my_col")
    docs, _ = search_and_rerank(query="영업이익은?", collection="my_col")
    print(answer_with_openai(query="영업이익은?", docs=docs, llm=get_llm()))

Table utilities only (no Qdrant / LLM required)::

    import fitz
    from table_rag.table import (
        extract_raw_tables_from_doc,
        merge_cross_page_raw_tables,
        table_to_text,
    )

    doc    = fitz.open("report.pdf")
    raw    = extract_raw_tables_from_doc(doc)
    merged = merge_cross_page_raw_tables(raw)
    for t in merged:
        print(table_to_text(t["rows"], table_title=t["title"]))

CLI::

    python -m table_rag          # interactive Q&A (reads .env)
"""

from .config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_SPARSE_MODEL,
)
from .document import load_document_chunks, load_pdf_chunks, load_pptx_chunks
from .indexing import ingest_pdf_to_qdrant, ingest_pdfs_to_qdrant, upsert_document_to_qdrant
from .models import Chunk, is_fastembed_available
from .qa import answer_with_openai, build_context_from_docs
from .retrieval import search_and_rerank

__all__ = [
    # config
    "DEFAULT_COLLECTION",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_SPARSE_MODEL",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    # models
    "Chunk",
    "is_fastembed_available",
    # document
    "load_document_chunks",
    "load_pdf_chunks",
    "load_pptx_chunks",
    # indexing
    "ingest_pdfs_to_qdrant",
    "ingest_pdf_to_qdrant",
    "upsert_document_to_qdrant",
    # retrieval
    "search_and_rerank",
    # qa
    "answer_with_openai",
    "build_context_from_docs",
]
