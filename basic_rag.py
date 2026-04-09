import os
import re
import sys
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# LangChain core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChain integrations
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever


DEFAULT_PDF = "[POSCO홀딩스]임원ㆍ주요주주특정증권등소유상황보고서(2026.03.10).pdf"
DEFAULT_COLLECTION = "posco_holdings_report_2026_03_10"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
HYBRID_DENSE = "dense"
HYBRID_SPARSE = "sparse"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    text: str
    source_type: str


# ── PDF extraction helpers (PyMuPDF 커스텀 – 변경 없음) ──────────────────────

def _normalize_cell(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _has_colspan_pattern(row: List[str]) -> bool:
    """
    colspan 패턴 감지. PyMuPDF의 두 가지 동작을 모두 처리:
    - 값 반복: ["매출", "매출", "매출"]
    - 빈 문자열: ["매출", "", ""]
    """
    for i in range(len(row) - 1):
        if row[i] and row[i] == row[i + 1]:   # 값 반복
            return True
        if row[i] and not row[i + 1]:          # 값 뒤에 빈 셀
            return True
    return False


def _is_mostly_non_numeric(row: List[str]) -> bool:
    """행의 비어있지 않은 셀 중 절반 이상이 숫자가 아니면 True (헤더 행 판별용)."""
    non_empty = [c for c in row if c]
    if not non_empty:
        return False
    numeric = sum(1 for c in non_empty if re.match(r'^[\d,.\s\-+%()]+$', c))
    return numeric / len(non_empty) < 0.5


def _is_likely_subheader(row0: List[str], row1: List[str]) -> bool:
    """row0에 colspan 패턴이 있고, row1도 헤더처럼 보일 때 2단 헤더로 판단."""
    if len(row0) != len(row1):
        return False
    return _has_colspan_pattern(row0) and _is_mostly_non_numeric(row1)


def _combine_header_rows(row0: List[str], row1: List[str]) -> List[str]:
    """
    2단 헤더를 1단으로 합침.
    - 부모=자식 (rowspan)         → 값 하나만 사용
    - 부모만 있음                  → 부모 사용
    - 자식만 있음                  → 자식 사용
    - 둘 다 다름 (colspan)         → "부모_자식" (예: "매출_국내")
    - 부모가 빈 문자열 (colspan 연속) → 앞 부모값을 승계해 "부모_자식" 생성
    """
    combined = []
    last_parent = ""
    for i, (parent, child) in enumerate(zip(row0, row1)):
        parent, child = parent.strip(), child.strip()
        # 부모가 비어있으면 앞 colspan 부모를 승계
        effective_parent = parent if parent else last_parent
        if parent:
            last_parent = parent

        if effective_parent == child:
            combined.append(effective_parent or f"col{i + 1}")
        elif not child:
            combined.append(effective_parent or f"col{i + 1}")
        elif not effective_parent:
            combined.append(child)
        else:
            combined.append(f"{effective_parent}_{child}")
    return combined


def _fill_rowspan_cells(body: List[List[str]]) -> List[List[str]]:
    """
    PyMuPDF는 rowspan 병합 셀 값을 스팬의 마지막 행에 배치하고
    앞 행들은 빈 문자열로 둡니다. 빈 셀 구간을 뒤에 오는 값으로 소급 채웁니다.

    예) ["", "A-1", ...] / ["A계열", "A-2", ...] → ["A계열", "A-1", ...] / ["A계열", "A-2", ...]
    """
    if not body:
        return body
    n_cols = max(len(r) for r in body)
    result = [list(r) + [""] * (n_cols - len(r)) for r in body]
    n_rows = len(result)

    for col in range(n_cols):
        i = 0
        while i < n_rows:
            if not result[i][col]:
                # 빈 구간 시작 — 다음 비어있지 않은 셀을 찾아 소급 채움
                j = i + 1
                while j < n_rows and not result[j][col]:
                    j += 1
                if j < n_rows:
                    # [i, j-1] 범위를 result[j][col] 값으로 채움
                    for k in range(i, j):
                        result[k][col] = result[j][col]
                i = j + 1
            else:
                i += 1
    return result


def _table_to_text(
    table_rows: List[List[str]],
    table_title: str = "",
    *,
    start_page: int = 0,
    end_page: int = 0,
) -> str:
    if not table_rows:
        return ""

    # N단 헤더(colspan) 자동 감지: 연속된 헤더 행을 모두 병합
    header_depth = 1
    while (
        header_depth < len(table_rows) - 1
        and _is_likely_subheader(table_rows[header_depth - 1], table_rows[header_depth])
    ):
        header_depth += 1

    if header_depth > 1:
        header = table_rows[0]
        for i in range(1, header_depth):
            header = _combine_header_rows(header, table_rows[i])
        body = table_rows[header_depth:]
    else:
        header = table_rows[0]
        body = table_rows[1:]

    # rowspan 빈 셀 소급 채우기
    body = _fill_rowspan_cells(body)

    lines: List[str] = []
    lines.append("TABLE")
    if table_title:
        lines.append(f"title: {table_title}")
    if start_page and end_page and end_page != start_page:
        lines.append(f"pages: {start_page}-{end_page}")
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


def _row_cells_equal(a: List[str], b: List[str]) -> bool:
    if len(a) != len(b):
        return False
    return all(_normalize_cell(x) == _normalize_cell(y) for x, y in zip(a, b))


def _table_same_column_count(rows_a: List[List[str]], rows_b: List[List[str]]) -> bool:
    if not rows_a or not rows_b:
        return False
    return len(rows_a[0]) == len(rows_b[0])


def _continuation_body_rows(header_row: List[str], next_page_rows: List[List[str]]) -> List[List[str]]:
    if not next_page_rows:
        return []
    if _row_cells_equal(header_row, next_page_rows[0]):
        return next_page_rows[1:]
    return next_page_rows


def _is_last_table_on_sorted_page(tables: List[dict], idx: int) -> bool:
    if idx >= len(tables) - 1:
        return True
    return tables[idx + 1]["page"] != tables[idx]["page"]


def _is_first_table_on_sorted_page(tables: List[dict], idx: int) -> bool:
    if idx == 0:
        return True
    return tables[idx]["page"] != tables[idx - 1]["page"]


def _merge_geometry_suggests_split(prev: dict, nxt: dict) -> bool:
    """이전 조각이 페이지 하단에, 다음 조각이 다음 페이지 상단에 있으면 같은 표의 연장으로 본다."""
    ph_p = float(prev.get("tail_page_height") or prev.get("page_height") or 0)
    ph_n = float(nxt.get("page_height") or 0)
    if ph_p <= 0 or ph_n <= 0:
        return True
    _, _, _, py1 = prev["tail_bbox"]
    _, ny0, _, _ = nxt["bbox"]
    return (py1 / ph_p >= 0.30) and (ny0 / ph_n <= 0.72)


def _merge_cross_page_raw_tables(raw: List[dict]) -> List[dict]:
    if not raw:
        return []
    tables = sorted(raw, key=lambda t: (t["page"], t["bbox"][1]))
    merged: List[dict] = []
    i = 0
    while i < len(tables):
        cur = {
            "page": tables[i]["page"],
            "end_page": tables[i]["page"],
            "rows": list(tables[i]["rows"]),
            "title": tables[i]["title"],
            "bbox": tables[i]["bbox"],
            "page_height": tables[i]["page_height"],
            "tail_bbox": tables[i]["bbox"],
            "tail_page_height": tables[i]["page_height"],
        }
        i += 1
        while i < len(tables):
            prev_idx = i - 1
            nxt = tables[i]
            if nxt["page"] != cur["end_page"] + 1:
                break
            if not _is_last_table_on_sorted_page(tables, prev_idx):
                break
            if not _is_first_table_on_sorted_page(tables, i):
                break
            if not _table_same_column_count(cur["rows"], nxt["rows"]):
                break
            if not _merge_geometry_suggests_split(cur, nxt):
                break
            hdr = cur["rows"][0]
            extra = _continuation_body_rows(hdr, nxt["rows"])
            cur["rows"].extend(extra)
            cur["end_page"] = nxt["page"]
            cur["bbox"] = (cur["bbox"][0], cur["bbox"][1], nxt["bbox"][2], nxt["bbox"][3])
            cur["tail_bbox"] = nxt["bbox"]
            cur["tail_page_height"] = nxt["page_height"]
            i += 1
        merged.append(cur)
    return merged


def _extract_raw_tables_from_doc(doc) -> List[dict]:
    raw: List[dict] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_no = page_index + 1
        page_h = float(page.rect.height)
        try:
            finder = page.find_tables()
            for table in getattr(finder, "tables", []):
                raw_rows = table.extract() or []
                rows = [[_normalize_cell(c) for c in r] for r in raw_rows]
                if not rows:
                    continue
                bbox = tuple(float(x) for x in table.bbox)
                title = _find_table_title(page, table)
                raw.append(
                    {
                        "page": page_no,
                        "rows": rows,
                        "title": title,
                        "bbox": bbox,
                        "page_height": page_h,
                    }
                )
        except Exception:
            pass
    return raw


def _build_merged_table_dicts(doc, *, merge_cross_page_tables: bool) -> List[dict]:
    raw = _extract_raw_tables_from_doc(doc)
    if not raw:
        return []
    if merge_cross_page_tables:
        return _merge_cross_page_raw_tables(raw)
    ordered = sorted(raw, key=lambda t: (t["page"], t["bbox"][1]))
    return [
        {"page": t["page"], "end_page": t["page"], "rows": list(t["rows"]), "title": t["title"]}
        for t in ordered
    ]


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


def _load_pdf_sections(
    pdf_path: Path,
    *,
    extract_table_chunks: bool = True,
    merge_cross_page_tables: bool = True,
) -> List[Chunk]:
    # Prefer PyMuPDF for robust layout/table extraction.
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        chunks: List[Chunk] = []
        merged_tables: List[dict] = []
        if extract_table_chunks:
            merged_tables = _build_merged_table_dicts(
                doc, merge_cross_page_tables=merge_cross_page_tables
            )

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
            for k, mt in enumerate(merged_tables, start=1):
                start_p = mt["page"]
                text = _table_to_text(
                    mt["rows"],
                    table_title=mt["title"],
                    start_page=start_p,
                    end_page=mt["end_page"],
                )
                if text:
                    chunks.append(
                        Chunk(
                            chunk_id=f"p{start_p:04d}_tb{k:03d}",
                            page=start_p,
                            text=text,
                            source_type="table",
                        )
                    )

        doc.close()
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
    merge_cross_page_tables: bool = True,
) -> List[Chunk]:
    raw = _load_pdf_sections(
        pdf_path,
        extract_table_chunks=extract_table_chunks,
        merge_cross_page_tables=merge_cross_page_tables,
    )
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


# ── LangChain 클라이언트 / 모델 팩토리 ───────────────────────────────────────

_qdrant_client_singleton: QdrantClient | None = None
_sparse_embedder_singleton = None  # FastEmbedSparse (optional dep)
_embed_model_singleton: HuggingFaceEmbeddings | None = None
_reranker_singleton: HuggingFaceCrossEncoder | None = None


def _get_qdrant_client() -> QdrantClient:
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


def _load_sparse_embedder():
    """BM25 sparse 임베딩 모델 (프로세스 내 싱글턴으로 재사용)."""
    global _sparse_embedder_singleton
    if _sparse_embedder_singleton is None:
        from langchain_qdrant import FastEmbedSparse
        _sparse_embedder_singleton = FastEmbedSparse(model_name=DEFAULT_SPARSE_MODEL)
    return _sparse_embedder_singleton


def _load_embed_model() -> HuggingFaceEmbeddings:
    """BGE-M3 밀집 임베딩 모델 (프로세스 내 싱글턴으로 재사용)."""
    global _embed_model_singleton
    if _embed_model_singleton is None:
        _embed_model_singleton = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embed_model_singleton


def _load_reranker() -> HuggingFaceCrossEncoder:
    """BGE 리랭커 모델 (프로세스 내 싱글턴으로 재사용)."""
    global _reranker_singleton
    if _reranker_singleton is None:
        _reranker_singleton = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    return _reranker_singleton


def _get_llm(model: str = DEFAULT_OPENAI_MODEL) -> ChatOpenAI:
    """LangChain ChatOpenAI 인스턴스 반환."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set in .env)")
    return ChatOpenAI(model=model, temperature=0.1, api_key=api_key)


def is_fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


def collection_is_hybrid(qdrant: QdrantClient, collection: str) -> bool:
    info = qdrant.get_collection(collection_name=collection)
    params = info.config.params
    sparse = getattr(params, "sparse_vectors", None) or {}
    # langchain_qdrant은 sparse 벡터를 "langchain-sparse" 키로 저장하므로
    # sparse_vectors가 비어 있지 않으면 hybrid collection으로 판단한다.
    return bool(sparse)


# ── 인덱싱 ──────────────────────────────────────────────────────────────────

def ingest_pdfs_to_qdrant(
    *,
    pdf_paths: Sequence[Union[Path, str]],
    collection: str = DEFAULT_COLLECTION,
    embed_model=None,
    extract_table_chunks: bool = True,
    enable_hybrid: bool = True,
    merge_cross_page_tables: bool = True,
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
        doc_chunks = _chunks_for_pdf(
            p,
            doc_slug=slugs[p],
            extract_table_chunks=extract_table_chunks,
            merge_cross_page_tables=merge_cross_page_tables,
        )
        if not doc_chunks:
            raise RuntimeError(f"No text extracted from PDF: {p.name} (스캔 이미지 PDF일 수 있음)")
        chunks.extend(doc_chunks)
        chunk_sources.extend([p] * len(doc_chunks))

    if embed_model is None:
        embed_model = _load_embed_model()

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
                "doc": chunk_sources[i].name,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "source_type": c.source_type,
            },
        )
        for i, c in enumerate(chunks)
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
        vs_kwargs["sparse_embedding"] = _load_sparse_embedder()
        vs_kwargs["retrieval_mode"] = RetrievalMode.HYBRID
    else:
        vs_kwargs["retrieval_mode"] = RetrievalMode.DENSE

    QdrantVectorStore.from_documents(**vs_kwargs)

    names = ", ".join(p.name for p in paths)
    mode = "hybrid dense+BM25 sparse" if do_hybrid else "dense only"
    print(f"Upserted {len(lc_docs)} chunks ({mode}) from {len(paths)} PDF(s) into '{collection}': {names}")
    return len(lc_docs)


def ingest_pdf_to_qdrant(
    *,
    pdf_path: Path,
    collection: str = DEFAULT_COLLECTION,
    embed_model=None,
    extract_table_chunks: bool = True,
    enable_hybrid: bool = True,
    merge_cross_page_tables: bool = True,
) -> int:
    return ingest_pdfs_to_qdrant(
        pdf_paths=[pdf_path],
        collection=collection,
        embed_model=embed_model,
        extract_table_chunks=extract_table_chunks,
        enable_hybrid=enable_hybrid,
        merge_cross_page_tables=merge_cross_page_tables,
    )


# ── 검색 + 리랭킹 ────────────────────────────────────────────────────────────

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
    sparse_embedder=None,  # API 호환성 유지 (내부적으로 FastEmbedSparse 사용)
    qdrant=None,
) -> Tuple[List[dict], str]:
    if embed_model is None:
        embed_model = _load_embed_model()
    if qdrant is None:
        qdrant = _get_qdrant_client()

    import time as _time
    hybrid_warning = ""
    is_hybrid_col = collection_is_hybrid(qdrant, collection)

    if use_hybrid and is_hybrid_col and is_fastembed_available():
        retrieval_mode = RetrievalMode.HYBRID
        sparse = _load_sparse_embedder()
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
            reranker = _load_reranker()
        compressor = CrossEncoderReranker(model=reranker, top_n=rerank_top_k)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    else:
        retriever = base_retriever

    import logging as _logging
    _pl = _logging.getLogger("table_rag.perf")
    _pl.info(f"[search] 시작 (top_k={qdrant_top_k}, reranker={use_reranker}, hybrid={retrieval_mode})")
    t0 = _time.time()
    results = retriever.invoke(query)
    _pl.info(f"[search] 완료 ({_time.time()-t0:.1f}s) — 결과 {len(results)}개")

    docs = []
    for doc in results:
        meta = doc.metadata or {}
        # reranker 적용 후: relevance_score / 미적용: _relevance_score (Qdrant 기본)
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

    # 리랭커 미사용 시 rerank_top_k로 직접 슬라이싱 (reranker는 top_n으로 이미 제한됨)
    if not use_reranker:
        docs = docs[:rerank_top_k]

    return docs, hybrid_warning


# ── 컨텍스트 빌더 ─────────────────────────────────────────────────────────────

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


# ── LLM 답변 생성 (LCEL 체인) ─────────────────────────────────────────────────

def answer_with_openai(
    *,
    query: str,
    docs: List[dict],
    llm: ChatOpenAI,
    model: str = DEFAULT_OPENAI_MODEL,
    use_reranker: bool = True,
) -> str:
    context = _build_context_from_docs(docs, use_reranker=use_reranker)
    if not context:
        return "검색된 컨텍스트가 없어 답변을 생성할 수 없습니다."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서 QA 어시스턴트입니다. 반드시 제공된 컨텍스트 범위 안에서만 답하세요. "
                "근거가 부족하면 모른다고 말하고, 추측하지 마세요. "
                "가능하면 답변 끝에 근거 청크 번호([1], [2]...)를 표시하세요.",
            ),
            (
                "human",
                "질문:\n{query}\n\n컨텍스트:\n{context}\n\n"
                "요청: 질문에 한국어로 간결하고 정확하게 답하세요.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "context": context})


# ── CLI 진입점 ───────────────────────────────────────────────────────────────

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
    merge_cross_page_tables = os.environ.get("MERGE_CROSS_PAGE_TABLES", "1").strip().lower() not in {
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

    embed_model = _load_embed_model()
    reranker = _load_reranker() if use_reranker else None
    qdrant = _get_qdrant_client()
    llm = _get_llm(model=openai_model)

    if ingest_on_start:
        mode = "bge-m3 + BM25 sparse (hybrid)" if want_hybrid else "bge-m3 (dense only)"
        print(f"Ingesting {len(pdf_paths)} PDF(s) into Qdrant ({mode})...")
        ingest_pdfs_to_qdrant(
            pdf_paths=pdf_paths,
            collection=collection,
            embed_model=embed_model,
            extract_table_chunks=extract_table_chunks,
            enable_hybrid=want_hybrid,
            merge_cross_page_tables=merge_cross_page_tables,
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
