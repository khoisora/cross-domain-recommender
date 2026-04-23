"""Replace every 1×1 "→ Next up…" callout table with a plain body paragraph.

Each Next-up block is currently rendered as a single-cell table that Word
displays as a yellow callout. We extract the paragraphs from the cell,
insert them directly in front of the table, and drop the table.

Run styling (bold "Next up.", font sizes) is preserved because we move
the underlying w:p XML element verbatim.
"""

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _cell_text(cell):
    return " ".join(p.text for p in cell.paragraphs).strip()


def main():
    doc = Document(str(DOC))

    # Tables whose single cell starts with "→ Next up" or "Next up.".
    targets = []
    for t in doc.tables:
        if len(t.rows) != 1 or len(t.rows[0].cells) != 1:
            continue
        txt = _cell_text(t.rows[0].cells[0])
        if "Next up" in txt[:12] or txt.startswith("→ Next up"):
            targets.append(t)

    print(f"Found {len(targets)} Next-up callout tables")

    unboxed = 0
    for t in targets:
        tbl = t._tbl
        parent = tbl.getparent()
        cell = t.rows[0].cells[0]

        # Clone each cell paragraph as a body paragraph positioned
        # immediately before the table, preserving order.
        for p in cell.paragraphs:
            new_p = deepcopy(p._p)
            # Strip any cell-level pPr that references tblCellBorders etc.;
            # a plain body paragraph just needs run content. We keep pPr as-is
            # (style name "normal") — that matches surrounding body paragraphs.
            parent.insert(list(parent).index(tbl), new_p)

        parent.remove(tbl)
        unboxed += 1
        print(f"  ✓ unboxed: {_cell_text(cell)[:80]}…")

    doc.save(str(DOC))
    print(f"\nUnboxed {unboxed} Next-up blocks")


if __name__ == "__main__":
    main()
