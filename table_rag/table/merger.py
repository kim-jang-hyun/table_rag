"""Cross-page table merging logic.

Tables that span multiple pages are detected and merged into a single dict
using five geometric and structural conditions.
"""

from typing import List

from .normalizer import normalize_cell


# ── Internal helpers ──────────────────────────────────────────────────────────

def _row_cells_equal(a: List[str], b: List[str]) -> bool:
    if len(a) != len(b):
        return False
    return all(normalize_cell(x) == normalize_cell(y) for x, y in zip(a, b))


def _table_same_column_count(rows_a: List[List[str]], rows_b: List[List[str]]) -> bool:
    if not rows_a or not rows_b:
        return False
    return len(rows_a[0]) == len(rows_b[0])


def _continuation_body_rows(
    header_row: List[str], next_page_rows: List[List[str]]
) -> List[List[str]]:
    """Strip a repeated header row from the continuation page, if present."""
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
    """Return True when geometry indicates *prev* and *nxt* are one split table.

    Conditions:
    - Previous fragment ends in the lower 70 % of its page (y1/h ≥ 0.30).
    - Next fragment starts in the upper 72 % of its page  (y0/h ≤ 0.72).
    """
    ph_p = float(prev.get("tail_page_height") or prev.get("page_height") or 0)
    ph_n = float(nxt.get("page_height") or 0)
    if ph_p <= 0 or ph_n <= 0:
        return True
    _, _, _, py1 = prev["tail_bbox"]
    _, ny0, _, _ = nxt["bbox"]
    return (py1 / ph_p >= 0.30) and (ny0 / ph_n <= 0.72)


# ── Public API ────────────────────────────────────────────────────────────────

def merge_cross_page_raw_tables(raw: List[dict]) -> List[dict]:
    """Merge table fragments that span page boundaries into single table dicts.

    A fragment on page N is merged with the next fragment on page N+1 when
    **all five** conditions hold:

    1. The two fragments are on consecutive pages.
    2. The previous fragment is the **last** table on its page.
    3. The next fragment is the **first** table on its page.
    4. Both fragments have the same column count.
    5. Page geometry suggests a split (prev ends ≥30 % down the page and next
       starts ≤72 % down the next page).

    Repeated header rows on continuation pages are automatically stripped.

    Args:
        raw: List of raw table dicts as returned by
             :func:`table_rag.table.extractor.extract_raw_tables_from_doc`.

    Returns:
        List of merged table dicts.  Each dict has these keys:

        - ``page``             (int)   — start page
        - ``end_page``         (int)   — end page (equals ``page`` for single-page tables)
        - ``rows``             (list)  — merged 2-D cell list
        - ``title``            (str)   — title from the first fragment
        - ``bbox``             (tuple) — bounding box of the first fragment
        - ``page_height``      (float) — page height of the first fragment
        - ``tail_bbox``        (tuple) — bounding box of the last fragment
        - ``tail_page_height`` (float) — page height of the last fragment

    Example::

        from table_rag.table import extract_raw_tables_from_doc, merge_cross_page_raw_tables
        import fitz

        doc = fitz.open("report.pdf")
        raw    = extract_raw_tables_from_doc(doc)
        merged = merge_cross_page_raw_tables(raw)
    """
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
