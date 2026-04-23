"""Replace `Arial Unicode MS` with `Arial` on every run.

Arial Unicode MS has much larger line metrics than Arial, so paragraphs
that mix the two fonts appear to have inconsistent line spacing even when
the paragraph-format line-spacing multiplier is identical. Modern Arial
handles the special glyphs (→, ×, ≥, λ, …) used in this report, so the
swap is safe.
"""

from pathlib import Path
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

OLD_FONT = "Arial Unicode MS"
NEW_FONT = "Arial"


def _patch(run):
    # Set all three name slots that Word checks (ascii, hAnsi, eastAsia).
    rFonts = run._element.find(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr/"
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts"
    )
    if rFonts is None:
        # Fall back to python-docx setter which creates rFonts if missing.
        run.font.name = NEW_FONT
        return True
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    changed = False
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        k = f"{ns}{attr}"
        if rFonts.get(k) == OLD_FONT:
            rFonts.set(k, NEW_FONT)
            changed = True
    return changed


def main():
    doc = Document(str(DOC))
    count = 0
    for p in doc.paragraphs:
        for r in p.runs:
            if r.font.name == OLD_FONT:
                if _patch(r):
                    count += 1
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        if r.font.name == OLD_FONT:
                            if _patch(r):
                                count += 1
    doc.save(str(DOC))
    print(f"Replaced {OLD_FONT} → {NEW_FONT} on {count} runs")


if __name__ == "__main__":
    main()
