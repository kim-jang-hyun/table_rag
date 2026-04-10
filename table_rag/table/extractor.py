"""PDF table extraction using PyMuPDF (fitz).

Requires: pip install pymupdf
"""

from typing import List

from .normalizer import normalize_cell


def find_table_title(page, table) -> str:
    """Find the title of a table by scanning for the closest text block above it.

    Args:
        page:  An open PyMuPDF ``Page`` object.
        table: A PyMuPDF table object with a ``.bbox`` attribute.

    Returns:
        The nearest non-empty text block above the table (at most 120 chars),
        or an empty string if none is found.
    """
    try:
        blocks = page.get_text("blocks") or []
        table_bbox = getattr(table, "bbox", None)
        if not table_bbox:
            return ""

        table_top = float(table_bbox[1])
        candidates: List[tuple] = []
        for block in blocks:
            if len(block) < 5:
                continue
            _x0, _y0, _x1, y1, text = block[0], block[1], block[2], block[3], block[4]
            content = " ".join(str(text).split())
            if not content:
                continue
            if float(y1) <= table_top + 2.0 and len(content) <= 120:
                distance = table_top - float(y1)
                candidates.append((distance, content))

        if not candidates:
            return ""
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    except Exception:
        return ""


def extract_raw_tables_from_doc(doc) -> List[dict]:
    """Extract raw table data from every page of a PyMuPDF document.

    Args:
        doc: An open ``fitz.Document`` object.

    Returns:
        List of table dicts, each containing:

        - ``page``        (int)   — 1-based page number
        - ``rows``        (list)  — 2-D list of normalized cell strings
        - ``title``       (str)   — text block just above the table (may be "")
        - ``bbox``        (tuple) — ``(x0, y0, x1, y1)`` in page coordinates
        - ``page_height`` (float) — page height in points

    Example::

        import fitz
        from table_rag.table import extract_raw_tables_from_doc

        doc = fitz.open("report.pdf")
        tables = extract_raw_tables_from_doc(doc)
    """
    raw: List[dict] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_no = page_index + 1
        page_h = float(page.rect.height)
        try:
            finder = page.find_tables()
            for table in getattr(finder, "tables", []):
                raw_rows = table.extract() or []
                rows = [[normalize_cell(c) for c in r] for r in raw_rows]
                if not rows:
                    continue
                bbox = tuple(float(x) for x in table.bbox)
                title = find_table_title(page, table)
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
