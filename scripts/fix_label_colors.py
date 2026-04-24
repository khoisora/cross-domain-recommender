"""Fix label paragraph colors — HYPOTHESIS should be blue, not body color."""
from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

BLUE = "2E5AAC"
GOLD = "8A6200"

LABEL_COLORS = {
    "HYPOTHESIS": BLUE,
}


def main():
    doc = Document(str(DOC))
    fixed = 0
    for p in doc.paragraphs:
        t = p.text.strip()
        if t in LABEL_COLORS:
            target_color = LABEL_COLORS[t]
            for r in p._p.findall(qn("w:r")):
                rPr = r.find(qn("w:rPr"))
                if rPr is None:
                    continue
                for old in rPr.findall(qn("w:color")):
                    rPr.remove(old)
                c = OxmlElement("w:color")
                c.set(qn("w:val"), target_color)
                rPr.append(c)
            fixed += 1

    doc.save(str(DOC))
    print(f"Fixed {fixed} label colors")


if __name__ == "__main__":
    main()
