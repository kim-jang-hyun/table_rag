"""Qdrant document ingestion.

Provides two ingestion strategies:

- :func:`ingest_pdfs_to_qdrant`      — full collection recreate from multiple docs
- :func:`upsert_document_to_qdrant`  — add/replace a single doc without recreating
"""

import os
from pathlib import Path
from typing import List, Sequence, Union

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import PayloadSchemaType

from .config import DEFAULT_COLLECTION
from .document import assign_doc_slugs, load_document_chunks, sanitize_doc_stem
from .models import (
    Chunk,
    get_qdrant_client,
    is_fastembed_available,
    load_embed_model,
    load_sparse_embedder,
)


def _chunks_to_lc_docs(chunks: List[Chunk], doc_name: str) -> List[Document]:
    return [
        Document(
            page_content=c.text,
            metadata={
                "doc": doc_name,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "source_type": c.source_type,
            },
        )
        for c in chunks
    ]


def ingest_pdfs_to_qdrant(
    *,
    pdf_paths: Sequence[Union[Path, str]],
    collection: str = DEFAULT_COLLECTION,
    embed_model=None,
    extract_table_chunks: bool = True,
    enable_hybrid: bool = True,
    merge_cross_page_tables: bool = True,
) -> int:
    """Ingest one or more documents into a Qdrant collection (full recreate).

    The collection is **deleted and recreated** on every call.  To add a
    single document without touching existing data use
    :func:`upsert_document_to_qdrant` instead.

    Args:
        pdf_paths:             Paths to PDF / PPTX documents.
        collection:            Qdrant collection name.
        embed_model:           Pre-loaded :class:`HuggingFaceEmbeddings` (optional;
                               loaded automatically when ``None``).
        extract_table_chunks:  Whether to extract table-specific chunks.
        enable_hybrid:         When ``True`` and ``fastembed`` is available, adds
                               BM25 sparse vectors for hybrid retrieval.
        merge_cross_page_tables: Whether to merge cross-page tables (PDF only).

    Returns:
        Total number of chunks indexed.

    Raises:
        ValueError:       If *pdf_paths* is empty.
        FileNotFoundError: If any path does not exist.
        RuntimeError:     If a document yields no extractable text.
    """
    paths = [Path(p).expanduser().resolve() for p in pdf_paths]
    if not paths:
        raise ValueError("No document paths provided")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("파일을 찾을 수 없습니다:\n" + "\n".join(missing))

    slugs = assign_doc_slugs(paths)
    all_chunks: List[Chunk] = []
    all_sources: List[Path] = []

    for p in paths:
        doc_chunks = load_document_chunks(
            p,
            doc_slug=slugs[p],
            extract_table_chunks=extract_table_chunks,
            merge_cross_page_tables=merge_cross_page_tables,
        )
        if not doc_chunks:
            raise RuntimeError(
                f"'{p.name}'에서 텍스트를 추출하지 못했습니다. "
                "(스캔 이미지 PDF이거나 내용이 없는 파일일 수 있음)"
            )
        all_chunks.extend(doc_chunks)
        all_sources.extend([p] * len(doc_chunks))

    if embed_model is None:
        embed_model = load_embed_model()

    do_hybrid = bool(enable_hybrid) and is_fastembed_available()
    if enable_hybrid and not do_hybrid:
        print(
            "주의: 하이브리드 인덱싱을 요청했지만 fastembed를 불러올 수 없어 밀집 벡터만 저장합니다. "
            "`pip install fastembed` 또는 Python 3.10~3.12 가상환경을 사용하세요."
        )

    lc_docs = [
        Document(
            page_content=c.text,
            metadata={
                "doc": all_sources[i].name,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "source_type": c.source_type,
            },
        )
        for i, c in enumerate(all_chunks)
    ]

    url = os.environ.get("QDRANT_URL", "").strip()
    api_key = os.environ.get("QDRANT_API_KEY", "").strip()

    vs_kwargs: dict = dict(
        documents=lc_docs,
        embedding=embed_model,
        collection_name=collection,
        url=url,
        api_key=api_key,
        force_recreate=True,
    )
    if do_hybrid:
        vs_kwargs["sparse_embedding"] = load_sparse_embedder()
        vs_kwargs["retrieval_mode"] = RetrievalMode.HYBRID
    else:
        vs_kwargs["retrieval_mode"] = RetrievalMode.DENSE

    QdrantVectorStore.from_documents(**vs_kwargs)

    get_qdrant_client().create_payload_index(
        collection_name=collection,
        field_name="metadata.doc",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    names = ", ".join(p.name for p in paths)
    mode = "hybrid dense+BM25 sparse" if do_hybrid else "dense only"
    print(f"Upserted {len(lc_docs)} chunks ({mode}) from {len(paths)} doc(s) into '{collection}': {names}")
    return len(lc_docs)


def ingest_pdf_to_qdrant(
    *,
    pdf_path: Union[Path, str],
    collection: str = DEFAULT_COLLECTION,
    embed_model=None,
    extract_table_chunks: bool = True,
    enable_hybrid: bool = True,
    merge_cross_page_tables: bool = True,
) -> int:
    """Convenience wrapper around :func:`ingest_pdfs_to_qdrant` for a single document."""
    return ingest_pdfs_to_qdrant(
        pdf_paths=[pdf_path],
        collection=collection,
        embed_model=embed_model,
        extract_table_chunks=extract_table_chunks,
        enable_hybrid=enable_hybrid,
        merge_cross_page_tables=merge_cross_page_tables,
    )


def upsert_document_to_qdrant(
    *,
    doc_path: Union[Path, str],
    collection: str = DEFAULT_COLLECTION,
    embed_model=None,
    extract_table_chunks: bool = True,
    enable_hybrid: bool = True,
    merge_cross_page_tables: bool = True,
) -> int:
    """Add or replace a single document in an existing Qdrant collection.

    Unlike :func:`ingest_pdfs_to_qdrant`, this does **not** recreate the
    collection.  It deletes the existing chunks for the document (matched by
    filename stored in ``metadata.doc``) and then inserts fresh chunks.
    If the collection does not exist yet it is created automatically.

    Args:
        doc_path:              Path to the document (PDF / PPTX).
        collection:            Qdrant collection name.
        embed_model:           Pre-loaded embedding model (optional).
        extract_table_chunks:  Whether to extract table-specific chunks.
        enable_hybrid:         Whether to add BM25 sparse vectors.
        merge_cross_page_tables: Whether to merge cross-page tables (PDF only).

    Returns:
        Number of chunks upserted.
    """
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    doc_path = Path(doc_path).expanduser().resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {doc_path}")

    doc_slug = sanitize_doc_stem(doc_path.stem)
    chunks = load_document_chunks(
        doc_path,
        doc_slug=doc_slug,
        extract_table_chunks=extract_table_chunks,
        merge_cross_page_tables=merge_cross_page_tables,
    )
    if not chunks:
        raise RuntimeError(
            f"'{doc_path.name}'에서 텍스트를 추출하지 못했습니다. "
            "(스캔 이미지 PDF이거나 내용이 없는 파일일 수 있음)"
        )

    if embed_model is None:
        embed_model = load_embed_model()

    do_hybrid = bool(enable_hybrid) and is_fastembed_available()
    url = os.environ.get("QDRANT_URL", "").strip()
    api_key = os.environ.get("QDRANT_API_KEY", "").strip()
    qdrant = get_qdrant_client()

    existing = [c.name for c in qdrant.get_collections().collections]
    if collection in existing:
        qdrant.create_payload_index(
            collection_name=collection,
            field_name="metadata.doc",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        ids_to_delete: list = []
        offset = None
        while True:
            results, offset = qdrant.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.doc",
                            match=MatchValue(value=doc_path.name),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids_to_delete.extend(r.id for r in results)
            if offset is None:
                break
        if ids_to_delete:
            qdrant.delete(collection_name=collection, points_selector=ids_to_delete)

    lc_docs = _chunks_to_lc_docs(chunks, doc_path.name)

    if collection not in existing:
        vs_kwargs: dict = dict(
            documents=lc_docs,
            embedding=embed_model,
            collection_name=collection,
            url=url,
            api_key=api_key,
            force_recreate=False,
        )
        if do_hybrid:
            vs_kwargs["sparse_embedding"] = load_sparse_embedder()
            vs_kwargs["retrieval_mode"] = RetrievalMode.HYBRID
        else:
            vs_kwargs["retrieval_mode"] = RetrievalMode.DENSE
        QdrantVectorStore.from_documents(**vs_kwargs)
    else:
        vs_kwargs = dict(
            client=qdrant,
            collection_name=collection,
            embedding=embed_model,
        )
        if do_hybrid:
            vs_kwargs["sparse_embedding"] = load_sparse_embedder()
            vs_kwargs["retrieval_mode"] = RetrievalMode.HYBRID
        else:
            vs_kwargs["retrieval_mode"] = RetrievalMode.DENSE
        vs = QdrantVectorStore(**vs_kwargs)
        vs.add_documents(lc_docs)

    mode = "hybrid dense+BM25 sparse" if do_hybrid else "dense only"
    print(f"Upserted {len(lc_docs)} chunks ({mode}) for '{doc_path.name}' into '{collection}'")
    return len(lc_docs)
