"""Quick verification of §5 lesson structure after v2 polish."""
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
                print(f"\n{'='*60}\np#{i} [{s}] {t}")
                continue
            elif in_s5:
                break
        if not in_s5:
            continue
        if (s.startswith("Heading")
                or t.startswith("In this lesson")
                or t.startswith("We ")
                or t.startswith("Experiment Setup")
                or t.startswith("What changed")
                or t.startswith("Key Finding")
                or t.startswith("•")):
            print(f"p#{i} [{s}] {t[:120]}")

    # Also sample first table header
    print("\n--- Table header samples ---")
    body = doc.element.body
    in_s5 = False
    tcount = 0
    for ch in body.iterchildren():
        tag = ch.tag.split("}", 1)[-1]
        if tag == "p":
            pPr = ch.find(qn("w:pPr"))
            pStyle = pPr.find(qn("w:pStyle")) if pPr is not None else None
            sv = pStyle.get(qn("w:val")) if pStyle is not None else None
            txt = "".join(t.text or "" for t in ch.iter(qn("w:t"))).strip()
            if sv in ("Heading1",):
                if txt.startswith("5."):
                    in_s5 = True
                elif in_s5:
                    break
        elif tag == "tbl" and in_s5:
            tcount += 1
            if tcount <= 3:
                first_row = ch.find(qn("w:tr"))
                if first_row is not None:
                    for tc in first_row.findall(qn("w:tc"))[:2]:
                        tcPr = tc.find(qn("w:tcPr"))
                        shd = tcPr.find(qn("w:shd")) if tcPr is not None else None
                        fill = shd.get(qn("w:fill")) if shd is not None else "none"
                        for r in tc.iter(qn("w:r")):
                            rPr = r.find(qn("w:rPr"))
                            col = rPr.find(qn("w:color")) if rPr is not None else None
                            cv = col.get(qn("w:val")) if col is not None else "inherit"
                            ct = "".join(t.text or "" for t in r.iter(qn("w:t")))[:20]
                            print(f"  tbl#{tcount} '{ct}' fill={fill} text={cv}")
                            break

if __name__ == "__main__":
    main()
