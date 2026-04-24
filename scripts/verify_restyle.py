"""Verify restyled §5 structure."""
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

def main():
    doc = Document(str(DOC))
    in_s5 = False
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        s = p.style.name
        if s == "Heading 1":
            if t.startswith("5. Experiments"):
                in_s5 = True
                print(f"\n{'='*60}")
                print(f"p#{i} [{s}] {t}")
                continue
            elif in_s5:
                break
        if not in_s5:
            continue

        # Get first run info
        first_r = p._p.find(qn("w:r"))
        run_info = ""
        if first_r is not None:
            rPr = first_r.find(qn("w:rPr"))
            if rPr is not None:
                sz_el = rPr.find(qn("w:sz"))
                sz = sz_el.get(qn("w:val"), "") if sz_el is not None else ""
                b_el = rPr.find(qn("w:b"))
                bold = "B" if b_el is not None else ""
                i_el = rPr.find(qn("w:i"))
                ital = "I" if i_el is not None else ""
                color_el = rPr.find(qn("w:color"))
                color = color_el.get(qn("w:val"), "") if color_el is not None else ""
                run_info = f" [{sz}{bold}{ital} #{color}]"

        if (s.startswith("Heading") or t.startswith("HYPOTHESIS")
                or t.startswith("KEY FINDING") or t.startswith("NEXT UP")
                or t.startswith("Setup") or t.startswith("In this lesson")
                or t.startswith("We ") or t.startswith("Will ")
                or t.startswith("Can ") or t.startswith("Does ")
                or "?" in t[-2:]):
            print(f"p#{i} [{s}]{run_info} {t[:110]}")

if __name__ == "__main__":
    main()
