import os
import re
import sys
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import Distance, PointStruct, VectorParams


DEFAULT_PDF = "[POSCO홀딩스]임원ㆍ주요주주특정증권등소유상황보고서(2026.03.10).pdf"
DEFAULT_COLLECTION = "posco_holdings_report_2026_03_10"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
HYBRID_DENSE = "dense"
HYBRID_SPARSE = "sparse"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    text: str
    source_type: str


def _normalize_cell(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _table_to_text(table_rows: List[List[str]], table_title: str = "") -> str:
    if not table_rows:
        return ""

    header = table_rows[0]
    body = table_rows[1:]
    lines: List[str] = []
    lines.append("TABLE")
    if table_title:
        lines.append(f"title: {table_title}")
    lines.append(f"columns: {', '.join(header)}")

    for row_idx, row in enumerate(body, start=1):
        cells: List[str] = []
        for col_name, cell in zip(header, row):
            if not cell:
                continue
            cells.append(f"{col_name}={cell}")
        row_text = " | ".join(cells) if cells else "(empty row)"
        lines.append(f"row{row_idx}: {row_text}")

    return "\n".join(lines).strip()


def _extract_page_table_texts(page) -> List[str]:
    table_texts: List[str] = []
    try:
        finder = page.find_tables()
        tables = getattr(finder, "tables", [])
        for table in tables:
            raw_rows = table.extract() or []
            rows: List[List[str]] = []
            for raw_row in raw_rows:
                rows.append([_normalize_cell(cell) for cell in raw_row])
            if not rows:
                continue
            table_title = _find_table_title(page, table)
            table_text = _table_to_text(rows, table_title=table_title)
            if table_text:
                table_texts.append(table_text)
    except Exception:
        # If table extraction fails for a page, continue with non-table chunks.
        pass
    return table_texts


def _find_table_title(page, table) -> str:
    # Heuristic: pick the closest non-empty text block above the table.
    try:
        blocks = page.get_text("blocks") or []
        table_bbox = getattr(table, "bbox", None)
        if not table_bbox:
            return ""

        table_top = float(table_bbox[1])
        candidates: List[tuple] = []
        for block in blocks:
            # PyMuPDF block format: (x0, y0, x1, y1, text, block_no, block_type, ...)
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            content = " ".join(str(text).split())
            if not content:
                continue
            # Keep only text above (or touching) table top with small tolerance.
            if float(y1) <= table_top + 2.0:
                # Prefer compact heading-like lines over long paragraphs.
                if len(content) <= 120:
                    distance = table_top - float(y1)
                    candidates.append((distance, content))

        if not candidates:
            return ""

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    except Exception:
        return ""


def _load_pdf_sections(pdf_path: Path, *, extract_table_chunks: bool = True) -> List[Chunk]:
    # Prefer PyMuPDF for robust layout/table extraction.
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        chunks: List[Chunk] = []
        for page_index in range(doc.page_count):
            page_no = page_index + 1
            page = doc.load_page(page_index)

            page_text = page.get_text("text") or ""
            page_text = " ".join(page_text.split())
            if page_text:
                for j, piece in enumerate(_chunk_text(page_text), start=1):
                    if piece:
                        chunks.append(
                            Chunk(
                                chunk_id=f"p{page_no:04d}_t{j:03d}",
                                page=page_no,
                                text=piece,
                                source_type="text",
                            )
                        )

            if extract_table_chunks:
                table_texts = _extract_page_table_texts(page)
                for k, table_text in enumerate(table_texts, start=1):
                    chunks.append(
                        Chunk(
                            chunk_id=f"p{page_no:04d}_tb{k:03d}",
                            page=page_no,
                            text=table_text,
                            source_type="table",
                        )
                    )
        return chunks
    except Exception as exc:
        raise RuntimeError(
            "Failed to extract PDF sections with PyMuPDF. "
            "Install/verify pymupdf and check whether the PDF is readable."
        ) from exc


def _sanitize_doc_stem(stem: str) -> str:
    s = " ".join((stem or "").split())
    s = re.sub(r"[^\w\-가-힣.]+", "_", s, flags=re.UNICODE)
    s = s.strip("_") or "doc"
    return s[:100]


def _assign_doc_slugs(pdf_paths: Sequence[Path]) -> Dict[Path, str]:
    stem_count: Dict[str, int] = defaultdict(int)
    out: Dict[Path, str] = {}
    for p in pdf_paths:
        base = _sanitize_doc_stem(p.stem)
        stem_count[base] += 1
        n = stem_count[base]
        out[p] = base if n == 1 else f"{base}_{n}"
    return out


def _chunks_for_pdf(
    pdf_path: Path,
    *,
    doc_slug: str,
    extract_table_chunks: bool = True,
) -> List[Chunk]:
    raw = _load_pdf_sections(pdf_path, extract_table_chunks=extract_table_chunks)
    prefixed: List[Chunk] = []
    for c in raw:
        prefixed.append(
            Chunk(
                chunk_id=f"{doc_slug}::{c.chunk_id}",
                page=c.page,
                text=c.text,
                source_type=c.source_type,
            )
        )
    return prefixed


def _chunk_text(text: str, *, chunk_size: int = 1000, chunk_overlap: int = 150) -> Iterable[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        yield text[start:end].strip()
        if end == n:
            break
        start = max(0, end - chunk_overlap)


def _get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "").strip()
    api_key = os.environ.get("QDRANT_API_KEY", "").strip()
    if not url:
        raise RuntimeError("Missing QDRANT_URL (set in .env)")
    if not api_key or api_key == "PASTE_YOUR_QDRANT_API_KEY_HERE":
        raise RuntimeError("Missing QDRANT_API_KEY (set in .env)")
    return QdrantClient(url=url, api_key=api_key)


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set in .env)")
    return OpenAI(api_key=api_key)


def _load_embed_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("BAAI/bge-m3")


def _load_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder("BAAI/bge-reranker-v2-m3")


def _load_models():
    # Lazy import to keep startup errors clearer.
    # Using sentence-transformers avoids optional deps that require MSVC build tools on Windows.
    return _load_embed_model(), _load_reranker()


def _embed_texts(embed_model, texts: List[str]) -> List[List[float]]:
    vecs = embed_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=len(texts) >= 32,
        normalize_embeddings=True,
    )
    return vecs.tolist() if hasattr(vecs, "tolist") else vecs


def is_fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


def _load_sparse_embedder(model_name: str = DEFAULT_SPARSE_MODEL):
    try:
        from fastembed import SparseTextEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "하이브리드(BM25)에 fastembed 패키지가 필요합니다. "
            "`pip install fastembed` 후 다시 시도하세요. "
            "(Windows에서는 Python 3.10~3.12 환경이 설치에 유리할 수 있습니다.)"
        ) from exc

    return SparseTextEmbedding(model_name=model_name)


def _numpyish_to_list(x) -> List:
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _sparse_embedding_to_qdrant(sparse_emb) -> qm.SparseVector:
    return qm.SparseVector(
        indices=_numpyish_to_list(getattr(sparse_emb, "indices", [])),
        values=_numpyish_to_list(getattr(sparse_emb, "values", [])),
    )


def collection_is_hybrid(qdrant: QdrantClient, collection: str) -> bool:
    info = qdrant.get_collection(collection_name=collection)
    params = info.config.params
    sparse = getattr(params, "sparse_vectors", None) or {}
    if not sparse:
        return False
    vecs = params.vectors
    if isinstance(vecs, dict):
        return HYBRID_DENSE in vecs
    return False


def _sparse_vector_params() -> qm.SparseVectorParams:
    modifier = getattr(qm, "Modifier", None)
    if modifier is not None and hasattr(modifier, "IDF"):
        try:
            return qm.SparseVectorParams(modifier=modifier.IDF)
        except Exception:
            pass
    return qm.SparseVectorParams()


def ingest_pdfs_to_qdrant(
    *,
    pdf_paths: Sequence[Union[Path, str]],
    collection: str = DEFAULT_COLLECTION,
    embed_model=None,
    extract_table_chunks: bool = True,
    enable_hybrid: bool = True,
    sparse_embedder=None,
) -> int:
    paths = [Path(p).expanduser().resolve() for p in pdf_paths]
    if not paths:
        raise ValueError("No PDF paths provided")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("PDF not found:\n" + "\n".join(missing))

    slugs = _assign_doc_slugs(paths)
    chunks: List[Chunk] = []
    chunk_sources: List[Path] = []
    for p in paths:
        doc_chunks = _chunks_for_pdf(p, doc_slug=slugs[p], extract_table_chunks=extract_table_chunks)
        if not doc_chunks:
            raise RuntimeError(f"No text extracted from PDF: {p.name} (스캔 이미지 PDF일 수 있음)")
        chunks.extend(doc_chunks)
        chunk_sources.extend([p] * len(doc_chunks))

    if embed_model is None:
        embed_model = _load_embed_model()
    texts = [c.text for c in chunks]
    vectors = _embed_texts(embed_model, texts)
    vector_size = len(vectors[0])

    do_hybrid = bool(enable_hybrid) and is_fastembed_available()
    if enable_hybrid and not do_hybrid:
        print(
            "주의: 하이브리드 인덱싱을 요청했지만 fastembed를 불러올 수 없어 밀집 벡터만 저장합니다. "
            "`pip install fastembed` 또는 Python 3.10~3.12 가상환경을 사용하세요."
        )

    sparse_vectors: List[qm.SparseVector] = []
    if do_hybrid:
        if sparse_embedder is None:
            sparse_embedder = _load_sparse_embedder()
        sparse_embs = list(sparse_embedder.embed(texts, batch_size=32))
        if len(sparse_embs) != len(chunks):
            raise RuntimeError("Sparse embedding count does not match chunk count")
        sparse_vectors = [_sparse_embedding_to_qdrant(s) for s in sparse_embs]

    qdrant = _get_qdrant_client()
    if qdrant.collection_exists(collection_name=collection):
        qdrant.delete_collection(collection_name=collection)
    if do_hybrid:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config={HYBRID_DENSE: VectorParams(size=vector_size, distance=Distance.COSINE)},
            sparse_vectors_config={HYBRID_SPARSE: _sparse_vector_params()},
        )
    else:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    points: List[PointStruct] = []
    for idx, (chunk, vector, src) in enumerate(zip(chunks, vectors, chunk_sources)):
        payload = {
            "doc": src.name,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
            "source_type": chunk.source_type,
            "text": chunk.text,
        }
        if do_hybrid:
            vec_payload = {HYBRID_DENSE: vector, HYBRID_SPARSE: sparse_vectors[idx]}
        else:
            vec_payload = vector
        points.append(PointStruct(id=idx, vector=vec_payload, payload=payload))

    qdrant.upsert(collection_name=collection, points=points)
    names = ", ".join(p.name for p in paths)
    mode = "hybrid dense+BM25 sparse" if do_hybrid else "dense only"
    print(f"Upserted {len(points)} chunks ({mode}) from {len(paths)} PDF(s) into '{collection}': {names}")
    return len(points)


def ingest_pdf_to_qdrant(
    *,
    pdf_path: Path,
    collection: str = DEFAULT_COLLECTION,
    embed_model=None,
    extract_table_chunks: bool = True,
    enable_hybrid: bool = True,
) -> int:
    return ingest_pdfs_to_qdrant(
        pdf_paths=[pdf_path],
        collection=collection,
        embed_model=embed_model,
        extract_table_chunks=extract_table_chunks,
        enable_hybrid=enable_hybrid,
    )


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
    if embed_model is None:
        embed_model = _load_embed_model()
    if use_reranker and reranker is None:
        reranker = _load_reranker()
    if qdrant is None:
        qdrant = _get_qdrant_client()

    hybrid_warning = ""
    qvec = _embed_texts(embed_model, [query])[0]
    is_hybrid_col = collection_is_hybrid(qdrant, collection)

    if use_hybrid and is_hybrid_col and is_fastembed_available():
        if sparse_embedder is None:
            sparse_embedder = _load_sparse_embedder()
        sq = list(sparse_embedder.query_embed(query))[0]
        svec = _sparse_embedding_to_qdrant(sq)
        resp = qdrant.query_points(
            collection_name=collection,
            prefetch=[
                qm.Prefetch(query=qvec, limit=qdrant_top_k, using=HYBRID_DENSE),
                qm.Prefetch(query=svec, limit=qdrant_top_k, using=HYBRID_SPARSE),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=qdrant_top_k,
            with_payload=True,
        )
    elif use_hybrid and is_hybrid_col:
        hybrid_warning = (
            "하이브리드 collection이지만 fastembed가 없어 BM25 없이 벡터 검색만 했습니다. "
            "`pip install fastembed` 하거나 Python 3.10~3.12 가상환경을 쓰면 RRF 하이브리드를 쓸 수 있습니다."
        )
        resp = qdrant.query_points(
            collection_name=collection,
            query=qvec,
            using=HYBRID_DENSE,
            limit=qdrant_top_k,
            with_payload=True,
        )
    elif use_hybrid and not is_hybrid_col:
        hybrid_warning = (
            "하이브리드 검색을 켰지만 이 collection은 BM25 sparse 없이 인덱싱되었습니다. "
            "벡터 검색만 사용했습니다. 하이브리드 인덱싱을 켠 뒤 다시 인덱싱하세요."
        )
        resp = qdrant.query_points(
            collection_name=collection,
            query=qvec,
            limit=qdrant_top_k,
            with_payload=True,
        )
    elif is_hybrid_col:
        resp = qdrant.query_points(
            collection_name=collection,
            query=qvec,
            using=HYBRID_DENSE,
            limit=qdrant_top_k,
            with_payload=True,
        )
    else:
        resp = qdrant.query_points(
            collection_name=collection,
            query=qvec,
            limit=qdrant_top_k,
            with_payload=True,
        )

    hits = resp.points
    if not hits:
        return [], hybrid_warning

    docs = []
    for h in hits:
        payload = h.payload or {}
        docs.append(
            {
                "score": float(h.score),
                "doc": payload.get("doc", ""),
                "page": payload.get("page"),
                "chunk_id": payload.get("chunk_id"),
                "source_type": payload.get("source_type", "text"),
                "text": payload.get("text", ""),
            }
        )

    if use_reranker and reranker is not None:
        pairs = [[query, d["text"]] for d in docs]
        rerank_scores = reranker.predict(pairs)
        for d, s in zip(docs, rerank_scores):
            d["rerank_score"] = float(s)
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    else:
        for d in docs:
            d["rerank_score"] = float(d["score"])
        docs.sort(key=lambda x: x["score"], reverse=True)

    return docs[:rerank_top_k], hybrid_warning


def _build_context_from_docs(docs: List[dict], *, use_reranker: bool = True) -> str:
    lines: List[str] = []
    rank_key = "rerank" if use_reranker else "vector"
    for i, doc in enumerate(docs, start=1):
        doc_name = doc.get("doc") or ""
        doc_bit = f" doc={doc_name}" if doc_name else ""
        lines.append(
            f"[{i}]{doc_bit} page={doc.get('page')} chunk_id={doc.get('chunk_id')} "
            f"type={doc.get('source_type')} {rank_key}={doc.get('rerank_score', 0.0):.4f}"
        )
        lines.append(doc.get("text", ""))
        lines.append("")
    return "\n".join(lines).strip()


def answer_with_openai(
    *,
    query: str,
    docs: List[dict],
    openai_client: OpenAI,
    model: str = DEFAULT_OPENAI_MODEL,
    use_reranker: bool = True,
) -> str:
    context = _build_context_from_docs(docs, use_reranker=use_reranker)
    if not context:
        return "검색된 컨텍스트가 없어 답변을 생성할 수 없습니다."

    system_prompt = (
        "당신은 문서 QA 어시스턴트입니다. 반드시 제공된 컨텍스트 범위 안에서만 답하세요. "
        "근거가 부족하면 모른다고 말하고, 추측하지 마세요. "
        "가능하면 답변 끝에 근거 청크 번호([1], [2]...)를 표시하세요."
    )
    user_prompt = (
        f"질문:\n{query}\n\n"
        f"컨텍스트:\n{context}\n\n"
        "요청: 질문에 한국어로 간결하고 정확하게 답하세요."
    )

    response = openai_client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return (response.output_text or "").strip()


def main():
    # Help Windows terminals display Korean properly when possible.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_dotenv()

    collection = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    openai_model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    ingest_on_start = os.environ.get("INGEST_ON_START", "1").strip().lower() not in {"0", "false", "no"}
    extract_table_chunks = os.environ.get("EXTRACT_TABLE_CHUNKS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }

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
            "Place files in this folder or set PDF_PATH (단일) / PDF_PATHS (여러 개, 쉼표·세미콜론·줄바꿈 구분) in .env"
        )

    use_reranker = os.environ.get("USE_RERANKER", "1").strip().lower() not in {"0", "false", "no"}
    want_hybrid = os.environ.get("USE_HYBRID", "1").strip().lower() not in {"0", "false", "no"}
    use_hybrid = want_hybrid and is_fastembed_available()
    if want_hybrid and not use_hybrid:
        print(
            "(알림) USE_HYBRID는 켜져 있지만 fastembed를 불러올 수 없습니다. "
            "밀집 벡터만 사용합니다. `pip install fastembed` 또는 Python 3.10~3.12 venv를 권장합니다."
        )

    # Load models once and reuse for interactive Q&A.
    embed_model = _load_embed_model()
    reranker = _load_reranker() if use_reranker else None
    sparse_embedder = _load_sparse_embedder() if use_hybrid else None
    qdrant = _get_qdrant_client()
    openai_client = _get_openai_client()

    if ingest_on_start:
        mode = "bge-m3 + BM25 sparse (hybrid)" if want_hybrid else "bge-m3 (dense only)"
        print(f"Ingesting {len(pdf_paths)} PDF(s) into Qdrant ({mode})...")
        ingest_pdfs_to_qdrant(
            pdf_paths=pdf_paths,
            collection=collection,
            embed_model=embed_model,
            extract_table_chunks=extract_table_chunks,
            enable_hybrid=want_hybrid,
        )
    else:
        print("Skipping ingest (INGEST_ON_START=0). Using existing Qdrant collection.")

    print("\n질문을 입력하세요. 종료하려면 'exit' 또는 'quit' 입력.")
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
            sparse_embedder=sparse_embedder,
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
            openai_client=openai_client,
            model=openai_model,
            use_reranker=use_reranker,
        )
        print("\nA> (OpenAI 답변)")
        print(llm_answer or "답변 생성에 실패했습니다.")


if __name__ == "__main__":
    main()
