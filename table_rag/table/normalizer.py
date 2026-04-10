"""Cell and header normalization utilities for PDF/PPTX table extraction.

Handles:
- Cell value whitespace normalization
- colspan pattern detection
- Multi-row (N-level) header merging
- rowspan empty-cell back-filling
"""

import re
from typing import List


def normalize_cell(value) -> str:
    """Normalize a cell value to a clean, whitespace-collapsed string."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def has_colspan_pattern(row: List[str]) -> bool:
    """Detect colspan patterns in a header row.

    PyMuPDF represents colspans in two ways:
    - Repeated value:  ["매출", "매출", "매출"]
    - Empty string:    ["매출", "", ""]

    Returns True when either pattern is found.
    """
    for i in range(len(row) - 1):
        if row[i] and row[i] == row[i + 1]:
            return True
        if row[i] and not row[i + 1]:
            return True
    return False


def is_mostly_non_numeric(row: List[str]) -> bool:
    """Return True if the majority of non-empty cells are non-numeric.

    Used as a heuristic to identify header rows.
    """
    non_empty = [c for c in row if c]
    if not non_empty:
        return False
    numeric = sum(1 for c in non_empty if re.match(r"^[\d,.\s\-+%()]+$", c))
    return numeric / len(non_empty) < 0.5


def is_likely_subheader(row0: List[str], row1: List[str]) -> bool:
    """Return True when row0 has a colspan pattern and row1 also looks like a header.

    Used to detect 2-level (and N-level) header blocks.
    """
    if len(row0) != len(row1):
        return False
    return has_colspan_pattern(row0) and is_mostly_non_numeric(row1)


def combine_header_rows(row0: List[str], row1: List[str]) -> List[str]:
    """Merge two header rows into one combined header row.

    Merging rules (applied per column):
    - parent == child (rowspan)          → keep one value
    - parent only                        → use parent
    - child only                         → use child
    - both differ (colspan)              → "parent_child" (e.g. "매출_국내")
    - parent is empty (colspan carryover)→ inherit the previous parent value
    """
    combined = []
    last_parent = ""
    for i, (parent, child) in enumerate(zip(row0, row1)):
        parent, child = parent.strip(), child.strip()
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


def fill_rowspan_cells(body: List[List[str]]) -> List[List[str]]:
    """Back-fill empty cells caused by PyMuPDF's rowspan representation.

    PyMuPDF places a rowspan cell's value in the *last* row of the span and
    leaves earlier rows as empty strings.  This function propagates that value
    backward to fill the empty slots.

    Example::

        Input:  [["",      "A-1"], ["A계열", "A-2"]]
        Output: [["A계열", "A-1"], ["A계열", "A-2"]]
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
                j = i + 1
                while j < n_rows and not result[j][col]:
                    j += 1
                if j < n_rows:
                    for k in range(i, j):
                        result[k][col] = result[j][col]
                i = j + 1
            else:
                i += 1
    return result
