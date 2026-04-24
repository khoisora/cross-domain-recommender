"""Verify §4 restyling."""
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

def main():
    doc = Document(str(DOC))
    in_s4 = False
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        s = p.style.name
        if s == "Heading 1":
            if "Methodology" in t:
                in_s4 = True
                print(f"p#{i} [{s}] {t}")
                continue
            elif in_s4:
                break
        if not in_s4:
            continue
        if not s.startswith("Heading") and not t.startswith("Figure") and not t.startswith("Equation"):
            continue
        first_r = p._p.find(qn("w:r"))
        ri = ""
        if first_r is not None:
            rPr = first_r.find(qn("w:rPr"))
            if rPr is not None:
                sz = rPr.find(qn("w:sz"))
                szv = sz.get(qn("w:val")) if sz is not None else ""
                col = rPr.find(qn("w:color"))
                cv = col.get(qn("w:val")) if col is not None else ""
                b = "B" if rPr.find(qn("w:b")) is not None else ""
                ii = "I" if rPr.find(qn("w:i")) is not None else ""
                ri = f" [{szv}{b}{ii} #{cv}]"
        print(f"p#{i} [{s}]{ri} {t[:100]}")

if __name__ == "__main__":
    main()
