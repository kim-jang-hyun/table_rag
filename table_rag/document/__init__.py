"""Document loading utilities for PDF and PPTX files.

The main entry point for loading chunks from any supported format is
:func:`load_document_chunks`, which dispatches to the correct loader based on
the file extension and prefixes every ``chunk_id`` with the document slug.
"""

from pathlib import Path
from typing import List

from ..models import Chunk
from .pdf_loader import (
    assign_doc_slugs,
    chunk_text,
    load_pdf_chunks,
    sanitize_doc_stem,
)
from .pptx_loader import load_pptx_chunks


def load_document_chunks(
    doc_path: Path,
    *,
    doc_slug: str,
    extract_table_chunks: bool = True,
    merge_cross_page_tables: bool = True,
) -> List[Chunk]:
    """Load chunks from a PDF or PPTX file, prefixing IDs with *doc_slug*.

    Dispatches to :func:`~.pdf_loader.load_pdf_chunks` for ``.pdf`` files and
    :func:`~.pptx_loader.load_pptx_chunks` for ``.pptx`` / ``.ppt`` files.

    Args:
        doc_path:              Path to the document.
        doc_slug:              Unique string prefix added to every ``chunk_id``,
                               e.g. ``"annual_report_2026"``.  Use
                               :func:`~.pdf_loader.assign_doc_slugs` to generate
                               collision-free slugs for multiple documents.
        extract_table_chunks:  Whether to extract table-specific chunks.
        merge_cross_page_tables: Whether to merge tables spanning pages (PDF only).

    Returns:
        List of :class:`~table_rag.models.Chunk` objects whose ``chunk_id``
        has the form ``"<doc_slug>::<original_id>"``.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = doc_path.suffix.lower()
    if ext == ".pdf":
        raw = load_pdf_chunks(
            doc_path,
            extract_table_chunks=extract_table_chunks,
            merge_cross_page_tables=merge_cross_page_tables,
        )
    elif ext in (".pptx", ".ppt"):
        raw = load_pptx_chunks(
            doc_path,
            extract_table_chunks=extract_table_chunks,
        )
    else:
        raise ValueError(
            f"Unsupported file extension: '{ext}'  (supported: .pdf, .pptx, .ppt)"
        )

    return [
        Chunk(
            chunk_id=f"{doc_slug}::{c.chunk_id}",
            page=c.page,
            text=c.text,
            source_type=c.source_type,
        )
        for c in raw
    ]


__all__ = [
    "load_document_chunks",
    "load_pdf_chunks",
    "load_pptx_chunks",
    "sanitize_doc_stem",
    "assign_doc_slugs",
    "chunk_text",
]
