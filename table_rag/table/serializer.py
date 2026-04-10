"""Table serialization for LLM/embedding consumption.

Provides two public functions:

- :func:`table_to_text`   — serialize a table for RAG (chunk text format)
- :func:`normalize_table` — normalize raw rows into (header, body) for PPTX / display
"""

from typing import List, Tuple

from .normalizer import (
    is_likely_subheader,
    combine_header_rows,
    fill_rowspan_cells,
    normalize_cell,
)


def table_to_text(
    table_rows: List[List[str]],
    table_title: str = "",
    *,
    start_page: int = 0,
    end_page: int = 0,
) -> str:
    """Serialize a table into a structured text format for LLM / embedding.

    Multi-row headers (colspan) are automatically detected and merged into a
    single header row.  rowspan gaps are back-filled before serialization.

    Output format::

        TABLE
        title: <title>
        pages: <start>-<end>        ← only when spanning multiple pages
        columns: col1, col2, ...
        row1: col1=val | col2=val | ...
        row2: ...

    Args:
        table_rows:  2-D list of strings; the first row is treated as the header.
        table_title: Optional title included in the ``title:`` line.
        start_page:  Starting page number (0 = omit the ``pages:`` line).
        end_page:    Ending page number   (0 = omit the ``pages:`` line).

    Returns:
        Formatted string, or ``""`` when *table_rows* is empty.

    Example::

        from table_rag.table import table_to_text

        rows = [["항목", "금액"], ["매출", "100"], ["영업이익", "20"]]
        print(table_to_text(rows, table_title="손익계산서"))
    """
    if not table_rows:
        return ""

    header_depth = 1
    while (
        header_depth < len(table_rows) - 1
        and is_likely_subheader(table_rows[header_depth - 1], table_rows[header_depth])
    ):
        header_depth += 1

    if header_depth > 1:
        header = table_rows[0]
        for i in range(1, header_depth):
            header = combine_header_rows(header, table_rows[i])
        body = table_rows[header_depth:]
    else:
        header = table_rows[0]
        body = table_rows[1:]

    body = fill_rowspan_cells(body)

    lines: List[str] = ["TABLE"]
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


def normalize_table(rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """Normalize raw table rows into a clean ``(header, body)`` tuple.

    Applies in order:

    1. Cell whitespace normalization via :func:`~.normalizer.normalize_cell`.
    2. Multi-row header merging (colspan detection).
    3. rowspan empty-cell back-fill.
    4. Column count alignment (body rows padded / truncated to header length).

    Args:
        rows: 2-D list of raw cell values (``None`` is allowed).

    Returns:
        ``(header, body)`` where *header* is ``List[str]`` and *body* is
        ``List[List[str]]`` with every row padded to ``len(header)`` columns.

    Example::

        from table_rag.table import normalize_table

        raw = [[None, "매출", "비용"], ["국내", "100", "80"], ["해외", "200", "150"]]
        header, body = normalize_table(raw)
    """
    if not rows:
        return [], []

    rows = [[normalize_cell(c) for c in r] for r in rows]

    header_depth = 1
    while (
        header_depth < len(rows) - 1
        and is_likely_subheader(rows[header_depth - 1], rows[header_depth])
    ):
        header_depth += 1

    if header_depth > 1:
        header = rows[0]
        for i in range(1, header_depth):
            header = combine_header_rows(header, rows[i])
        body = rows[header_depth:]
    else:
        header = rows[0]
        body = rows[1:]

    body = fill_rowspan_cells(body)

    n_cols = len(header)
    body = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in body]

    return header, body
