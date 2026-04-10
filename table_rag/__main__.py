"""CLI entry point: python -m table_rag

Reads configuration from environment variables / .env file and runs an
interactive question-answering loop over indexed Qdrant documents.

Key environment variables
-------------------------
QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
OPENAI_API_KEY, OPENAI_MODEL
PDF_PATH        — single document path
PDF_PATHS       — comma/semicolon/newline-separated list (takes precedence)
INGEST_ON_START — set to 0 to skip ingestion and use an existing collection
USE_RERANKER    — set to 0 to skip cross-encoder reranking
USE_HYBRID      — set to 0 to disable BM25 hybrid search
EXTRACT_TABLE_CHUNKS    — set to 0 to disable table chunk extraction
MERGE_CROSS_PAGE_TABLES — set to 0 to disable cross-page table merging
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import DEFAULT_COLLECTION, DEFAULT_OPENAI_MODEL, DEFAULT_PDF
from .indexing import ingest_pdfs_to_qdrant
from .models import (
    get_llm,
    get_qdrant_client,
    is_fastembed_available,
    load_embed_model,
    load_reranker,
)
from .qa import answer_with_openai
from .retrieval import search_and_rerank


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_dotenv()

    collection = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    openai_model = (
        os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    )

    def _flag(key: str, default: str = "1") -> bool:
        return os.environ.get(key, default).strip().lower() not in {"0", "false", "no"}

    ingest_on_start = _flag("INGEST_ON_START")
    extract_table_chunks = _flag("EXTRACT_TABLE_CHUNKS")
    merge_cross_page_tables = _flag("MERGE_CROSS_PAGE_TABLES")
    use_reranker = _flag("USE_RERANKER")
    want_hybrid = _flag("USE_HYBRID")

    pdf_paths_env = os.environ.get("PDF_PATHS", "").strip()
    if pdf_paths_env:
        parts = re.split(r"[,;\n]+", pdf_paths_env)
        pdf_paths = [Path(p.strip()) for p in parts if p.strip()]
    else:
        pdf_paths = [Path(os.environ.get("PDF_PATH", DEFAULT_PDF))]

    missing = [str(p) for p in pdf_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "PDF not found:\n" + "\n".join(missing) + "\n"
            "Place files in this folder or set PDF_PATH / PDF_PATHS in .env"
        )

    use_hybrid = want_hybrid and is_fastembed_available()
    if want_hybrid and not use_hybrid:
        print(
            "(알림) USE_HYBRID는 켜져 있지만 fastembed를 불러올 수 없습니다. "
            "밀집 벡터만 사용합니다. `pip install fastembed` 또는 Python 3.10~3.12 venv를 권장합니다."
        )

    embed_model = load_embed_model()
    reranker = load_reranker() if use_reranker else None
    qdrant = get_qdrant_client()
    llm = get_llm(model=openai_model)

    if ingest_on_start:
        mode = "bge-m3 + BM25 sparse (hybrid)" if want_hybrid else "bge-m3 (dense only)"
        print(f"Ingesting {len(pdf_paths)} document(s) into Qdrant ({mode})...")
        ingest_pdfs_to_qdrant(
            pdf_paths=pdf_paths,
            collection=collection,
            embed_model=embed_model,
            extract_table_chunks=extract_table_chunks,
            enable_hybrid=want_hybrid,
            merge_cross_page_tables=merge_cross_page_tables,
        )
    else:
        print("Skipping ingest (INGEST_ON_START=0).  Using existing Qdrant collection.")

    print("\n질문을 입력하세요.  종료하려면 'exit' 또는 'quit' 입력.")
    while True:
        user_q = input("\nQ> ").strip()
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            break

        top, hybrid_warn = search_and_rerank(
            query=user_q,
            collection=collection,
            qdrant_top_k=20,
            rerank_top_k=5,
            use_reranker=use_reranker,
            use_hybrid=want_hybrid,
            embed_model=embed_model,
            reranker=reranker,
            qdrant=qdrant,
        )
        if hybrid_warn:
            print(f"(알림) {hybrid_warn}")
        if not top:
            print("A> 검색 결과가 없습니다.")
            continue

        rank_label = "rerank" if use_reranker else "vector"
        print(f"\nA> ({rank_label} 기준 상위 {len(top)}개 청크)")
        for i, r in enumerate(top, start=1):
            print(
                f"\n[{i}] page={r['page']} chunk_id={r['chunk_id']} "
                f"type={r['source_type']} {rank_label}={r['rerank_score']:.4f}"
            )
            print(r["text"])

        llm_answer = answer_with_openai(
            query=user_q,
            docs=top,
            llm=llm,
            model=openai_model,
            use_reranker=use_reranker,
        )
        print("\nA> (OpenAI 답변)")
        print(llm_answer or "답변 생성에 실패했습니다.")


if __name__ == "__main__":
    main()
