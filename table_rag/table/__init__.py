"""table_rag.table — reusable PDF/PPTX table utilities.

This sub-package contains pure table-processing logic with **no RAG or vector DB
dependencies**.  It can be used independently of the rest of ``table_rag``.

Quick start::

    import fitz
    from table_rag.table import (
        extract_raw_tables_from_doc,
        merge_cross_page_raw_tables,
        table_to_text,
    )

    doc = fitz.open("report.pdf")
    raw    = extract_raw_tables_from_doc(doc)
    merged = merge_cross_page_raw_tables(raw)
    for t in merged:
        print(table_to_text(t["rows"], table_title=t["title"]))

Modules
-------
normalizer
    Cell normalization, colspan/rowspan detection, multi-row header merging.
extractor
    PyMuPDF-based table extraction from PDF documents.
merger
    Cross-page table merging with geometric and structural heuristics.
serializer
    Serialize tables to LLM-ready text or normalize to (header, body) tuples.
"""

from .normalizer import (
    normalize_cell,
    has_colspan_pattern,
    is_mostly_non_numeric,
    is_likely_subheader,
    combine_header_rows,
    fill_rowspan_cells,
)
from .extractor import extract_raw_tables_from_doc, find_table_title
from .merger import merge_cross_page_raw_tables
from .serializer import table_to_text, normalize_table

__all__ = [
    # normalizer
    "normalize_cell",
    "has_colspan_pattern",
    "is_mostly_non_numeric",
    "is_likely_subheader",
    "combine_header_rows",
    "fill_rowspan_cells",
    # extractor
    "extract_raw_tables_from_doc",
    "find_table_title",
    # merger
    "merge_cross_page_raw_tables",
    # serializer
    "table_to_text",
    "normalize_table",
]
