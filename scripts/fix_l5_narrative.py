"""Fix L5 narrative paragraph which got L4's text instead."""
from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

WRONG = "We tighten the source-domain filter"
RIGHT = (
    "We prune the movie catalog by removing items with fewer than 10 "
    "interactions, cutting 74% of titles while retaining 75% of "
    "interactions. "
    "Does a sharper, denser catalog improve cross-domain transfer, or "
    "does every model lose diversity?"
)


def main():
    doc = Document(str(DOC))
    in_l5 = False
    for i, p in enumerate(doc.paragraphs):
        s = p.style.name
        t = p.text.strip()
        if s == "Heading 2" and t.startswith("5.5"):
            in_l5 = True
            continue
        if s == "Heading 2" and t.startswith("5.6"):
            break
        if not in_l5:
            continue
        if t.startswith(WRONG):
            # Replace runs
            for r in list(p._p.findall(qn("w:r"))):
                p._p.remove(r)
            r = OxmlElement("w:r")
            rPr = OxmlElement("w:rPr")
            rFonts = OxmlElement("w:rFonts")
            for attr in ("ascii", "hAnsi", "cs"):
                rFonts.set(qn(f"w:{attr}"), "Arial")
            rPr.append(rFonts)
            for tag in ("w:sz", "w:szCs"):
                e = OxmlElement(tag)
                e.set(qn("w:val"), "20")
                rPr.append(e)
            r.append(rPr)
            te = OxmlElement("w:t")
            te.text = RIGHT
            te.set(qn("xml:space"), "preserve")
            r.append(te)
            p._p.append(r)
            print(f"Fixed p#{i}")
            break
    else:
        print("No fix needed or L5 narrative not found")

    doc.save(str(DOC))
    print("✓ L5 narrative fixed")


if __name__ == "__main__":
    main()
