"""Dump §5 paragraphs around each lesson heading to understand structure."""

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
                print(f"\np#{i} [{s}] {t}")
                continue
            elif in_s5:
                break
        if not in_s5:
            continue
        if (s.startswith("Heading")
                or t.startswith("In this lesson")
                or t.startswith("Experiment Setup")
                or t.startswith("What changed")
                or "Variable Changed" in t
                or "Held Constant" in t
                or "Expectation" in t
                or t.startswith("•")
                or t.startswith("Question")
                or t.startswith("Key Finding")
                or t.startswith("Lesson")
                or len(t) < 3):
            print(f"p#{i} [{s}] {t[:130]}")

    # Also check existing table header shading
    print("\n--- Table header shading samples ---")
    body_el = doc.element.body
    in_s5 = False
    tcount = 0
    for child in body_el.iterchildren():
        tag = child.tag.split("}", 1)[-1]
        if tag == "p":
            text = "".join(tt.text or "" for tt in child.iter(qn("w:t"))).strip()
            pPr = child.find(qn("w:pPr"))
            pStyle = pPr.find(qn("w:pStyle")) if pPr is not None else None
            sv = pStyle.get(qn("w:val")) if pStyle is not None else None
            if sv in ("Heading1", "heading1"):
                if text.startswith("5. Experiments"):
                    in_s5 = True
                elif in_s5:
                    break
        elif tag == "tbl" and in_s5:
            tcount += 1
            if tcount <= 3:
                first_row = child.find(qn("w:tr"))
                if first_row is not None:
                    for tc in first_row.findall(qn("w:tc"))[:3]:
                        tcPr = tc.find(qn("w:tcPr"))
                        shd = tcPr.find(qn("w:shd")) if tcPr is not None else None
                        fill = shd.get(qn("w:fill")) if shd is not None else "none"
                        cell_text = "".join(tt.text or "" for tt in tc.iter(qn("w:t")))[:30]
                        # Check run color
                        for r in tc.iter(qn("w:r")):
                            rPr = r.find(qn("w:rPr"))
                            color_el = rPr.find(qn("w:color")) if rPr is not None else None
                            color_val = color_el.get(qn("w:val")) if color_el is not None else "inherit"
                            print(f"  tbl#{tcount} cell='{cell_text}' fill={fill} textColor={color_val}")
                            break


if __name__ == "__main__":
    main()
