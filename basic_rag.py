import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


DEFAULT_PDF = "[POSCO홀딩스]임원ㆍ주요주주특정증권등소유상황보고서(2026.03.10).pdf"
DEFAULT_COLLECTION = "posco_holdings_report_2026_03_10"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    text: str


def _load_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    # Prefer PyMuPDF for better Korean text extraction on many PDFs.
    # Fallback to pypdf if PyMuPDF isn't available.
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        pages: List[Tuple[int, str]] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            text = " ".join(text.split())
            pages.append((i + 1, text))
        return pages
    except Exception:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages: List[Tuple[int, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            pages.append((i, text))
        return pages


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


def _build_chunks(pages: List[Tuple[int, str]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_num, page_text in pages:
        if not page_text:
            continue
        for j, piece in enumerate(_chunk_text(page_text), start=1):
            if not piece:
                continue
            chunk_id = f"p{page_num:04d}_c{j:03d}"
            chunks.append(Chunk(chunk_id=chunk_id, page=page_num, text=piece))
    return chunks


def _get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "").strip()
    api_key = os.environ.get("QDRANT_API_KEY", "").strip()
    if not url:
        raise RuntimeError("Missing QDRANT_URL in .env")
    if not api_key or api_key == "PASTE_YOUR_QDRANT_API_KEY_HERE":
        raise RuntimeError("Missing QDRANT_API_KEY in .env")
    return QdrantClient(url=url, api_key=api_key)


def _load_models():
    # Lazy import to keep startup errors clearer.
    # Using sentence-transformers avoids optional deps that require MSVC build tools on Windows.
    from sentence_transformers import CrossEncoder, SentenceTransformer

    embed = SentenceTransformer("BAAI/bge-m3")
    rerank = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return embed, rerank


def _embed_texts(embed_model, texts: List[str]) -> List[List[float]]:
    vecs = embed_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=len(texts) >= 32,
        normalize_embeddings=True,
    )
    return vecs.tolist() if hasattr(vecs, "tolist") else vecs


def ingest_pdf_to_qdrant(*, pdf_path: Path, collection: str = DEFAULT_COLLECTION) -> None:
    pages = _load_pdf_pages(pdf_path)
    chunks = _build_chunks(pages)
    if not chunks:
        raise RuntimeError("No text extracted from PDF (is it scanned 이미지 PDF?)")

    embed_model, _ = _load_models()
    vectors = _embed_texts(embed_model, [c.text for c in chunks])
    vector_size = len(vectors[0])

    qdrant = _get_qdrant_client()
    if qdrant.collection_exists(collection_name=collection):
        qdrant.delete_collection(collection_name=collection)
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points: List[PointStruct] = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            "doc": pdf_path.name,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
        }
        points.append(PointStruct(id=idx, vector=vector, payload=payload))

    qdrant.upsert(collection_name=collection, points=points)
    print(f"Upserted {len(points)} chunks into Qdrant collection '{collection}'.")


def search_and_rerank(
    *,
    query: str,
    collection: str = DEFAULT_COLLECTION,
    qdrant_top_k: int = 20,
    rerank_top_k: int = 5,
):
    embed_model, reranker = _load_models()
    qdrant = _get_qdrant_client()

    qvec = _embed_texts(embed_model, [query])[0]
    resp = qdrant.query_points(
        collection_name=collection,
        query=qvec,
        limit=qdrant_top_k,
        with_payload=True,
    )
    hits = resp.points
    if not hits:
        return []

    docs = []
    for h in hits:
        payload = h.payload or {}
        docs.append(
            {
                "score": float(h.score),
                "page": payload.get("page"),
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text", ""),
            }
        )

    pairs = [[query, d["text"]] for d in docs]
    rerank_scores = reranker.predict(pairs)
    for d, s in zip(docs, rerank_scores):
        d["rerank_score"] = float(s)

    docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return docs[:rerank_top_k]


def main():
    # Help Windows terminals display Korean properly when possible.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_dotenv()

    pdf_path = Path(os.environ.get("PDF_PATH", DEFAULT_PDF))
    collection = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    ingest_on_start = os.environ.get("INGEST_ON_START", "1").strip().lower() not in {"0", "false", "no"}

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Place the PDF in this folder or set PDF_PATH in .env"
        )

    # Load models once and reuse for interactive Q&A.
    embed_model, reranker = _load_models()
    qdrant = _get_qdrant_client()

    if ingest_on_start:
        print("Ingesting PDF into Qdrant (bge-m3)...")
        pages = _load_pdf_pages(pdf_path)
        chunks = _build_chunks(pages)
        if not chunks:
            raise RuntimeError("No text extracted from PDF (is it scanned 이미지 PDF?)")

        vectors = _embed_texts(embed_model, [c.text for c in chunks])
        vector_size = len(vectors[0])

        if qdrant.collection_exists(collection_name=collection):
            qdrant.delete_collection(collection_name=collection)
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        points: List[PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {
                "doc": pdf_path.name,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            }
            points.append(PointStruct(id=idx, vector=vector, payload=payload))

        qdrant.upsert(collection_name=collection, points=points)
        print(f"Upserted {len(points)} chunks into Qdrant collection '{collection}'.")
    else:
        print("Skipping ingest (INGEST_ON_START=0). Using existing Qdrant collection.")

    print("\n질문을 입력하세요. 종료하려면 'exit' 또는 'quit' 입력.")
    while True:
        user_q = input("\nQ> ").strip()
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            break

        qvec = _embed_texts(embed_model, [user_q])[0]
        resp = qdrant.query_points(
            collection_name=collection,
            query=qvec,
            limit=20,
            with_payload=True,
        )
        hits = resp.points
        if not hits:
            print("A> 검색 결과가 없습니다.")
            continue

        docs = []
        for h in hits:
            payload = h.payload or {}
            docs.append(
                {
                    "score": float(h.score),
                    "page": payload.get("page"),
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text", ""),
                }
            )

        pairs = [[user_q, d["text"]] for d in docs]
        rerank_scores = reranker.predict(pairs)
        for d, s in zip(docs, rerank_scores):
            d["rerank_score"] = float(s)

        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        top = docs[:5]

        print("\nA> (rerank 상위 5개 청크)")
        for i, r in enumerate(top, start=1):
            print(f"\n[{i}] page={r['page']} chunk_id={r['chunk_id']} rerank={r['rerank_score']:.4f}")
            print(r["text"])


if __name__ == "__main__":
    main()
