"""PDF document loading and text/table chunking.

Requires: pip install pymupdf
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from ..config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from ..models import Chunk
from ..table.extractor import extract_raw_tables_from_doc
from ..table.merger import merge_cross_page_raw_tables
from ..table.serializer import table_to_text


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterable[str]:
    """Split *text* into overlapping windows of at most *chunk_size* characters.

    Args:
        text:          Input text to split.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Yields:
        Non-empty stripped substrings.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        yield text[start:end].strip()
        if end == n:
            break
        start = max(0, end - chunk_overlap)


def sanitize_doc_stem(stem: str) -> str:
    """Convert a filename stem into a safe slug for use in chunk IDs.

    Collapses whitespace, replaces non-word characters with underscores,
    and truncates to 100 characters.
    """
    s = " ".join((stem or "").split())
    s = re.sub(r"[^\w\-가-힣.]+", "_", s, flags=re.UNICODE)
    s = s.strip("_") or "doc"
    return s[:100]


def assign_doc_slugs(paths: Sequence[Path]) -> Dict[Path, str]:
    """Assign a unique slug string to each document path.

    Duplicate stems are disambiguated with a numeric suffix (``_2``, ``_3``, …).

    Args:
        paths: Sequence of document ``Path`` objects.

    Returns:
        Mapping from each path to its slug string.
    """
    stem_count: Dict[str, int] = defaultdict(int)
    out: Dict[Path, str] = {}
    for p in paths:
        base = sanitize_doc_stem(p.stem)
        stem_count[base] += 1
        n = stem_count[base]
        out[p] = base if n == 1 else f"{base}_{n}"
    return out


def _build_merged_table_dicts(doc, *, merge_cross_page_tables: bool) -> List[dict]:
    raw = extract_raw_tables_from_doc(doc)
    if not raw:
        return []
    if merge_cross_page_tables:
        return merge_cross_page_raw_tables(raw)
    ordered = sorted(raw, key=lambda t: (t["page"], t["bbox"][1]))
    return [
        {"page": t["page"], "end_page": t["page"], "rows": list(t["rows"]), "title": t["title"]}
        for t in ordered
    ]


def load_pdf_chunks(
    pdf_path: Path,
    *,
    extract_table_chunks: bool = True,
    merge_cross_page_tables: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Extract text and table chunks from a PDF file using PyMuPDF.

    Args:
        pdf_path:              Path to the ``.pdf`` file.
        extract_table_chunks:  When ``True``, detected tables are serialized as
                               separate ``"table"`` chunks in addition to the
                               plain-text ``"text"`` chunks.
        merge_cross_page_tables: When ``True``, tables split across consecutive
                               pages are merged into a single chunk.
        chunk_size:            Maximum character length of each text chunk.
        chunk_overlap:         Character overlap between consecutive text chunks.

    Returns:
        List of :class:`~table_rag.models.Chunk` objects.  Each chunk has
        ``chunk_id`` in the form ``p<page>_t<n>`` (text) or ``p<page>_tb<n>``
        (table), without a doc-slug prefix — add one via
        :func:`table_rag.document.load_document_chunks`.

    Raises:
        RuntimeError: If PyMuPDF is not installed or the PDF cannot be read.
    """
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is not installed.  Run `pip install pymupdf`."
        ) from exc

    try:
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
            page_text = " ".join((page.get_text("text") or "").split())
            if page_text:
                for j, piece in enumerate(
                    chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                    start=1,
                ):
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
                text = table_to_text(
                    mt["rows"],
                    mt["title"],
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
            f"Failed to extract sections from '{pdf_path}'.  "
            "Check that pymupdf is installed and the file is a readable PDF."
        ) from exc
