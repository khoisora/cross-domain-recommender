"""Inspect the redesigned doc's §5 structure."""

from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2_redesigned.docx"


def main():
    doc = Document(str(DOC))
    in_s5 = False
    count = 0
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        s = p.style.name
        sid = p.style.style_id

        # Detect §5 entry
        if "Heading" in s and ("5." in t[:4] or "Experiments" in t or "Lesson 1" in t or "Lesson 2" in t):
            in_s5 = True
        if in_s5 and "Heading" in s and t.startswith("6."):
            break

        if not in_s5:
            continue

        # Get first run formatting
        run_info = ""
        first_r = p._p.find(qn("w:r"))
        if first_r is not None:
            rPr = first_r.find(qn("w:rPr"))
            if rPr is not None:
                font_el = rPr.find(qn("w:rFonts"))
                font = font_el.get(qn("w:ascii"), "") if font_el is not None else ""
                sz_el = rPr.find(qn("w:sz"))
                sz = sz_el.get(qn("w:val"), "") if sz_el is not None else ""
                b_el = rPr.find(qn("w:b"))
                bold = " B" if b_el is not None else ""
                i_el = rPr.find(qn("w:i"))
                ital = " I" if i_el is not None else ""
                color_el = rPr.find(qn("w:color"))
                color = f" #{color_el.get(qn('w:val'))}" if color_el is not None else ""
                run_info = f" [{font} {sz}{bold}{ital}{color}]"

        # Paragraph props
        pPr = p._p.find(qn("w:pPr"))
        spacing_info = ""
        if pPr is not None:
            sp = pPr.find(qn("w:spacing"))
            if sp is not None:
                before = sp.get(qn("w:before"), "")
                after = sp.get(qn("w:after"), "")
                line = sp.get(qn("w:line"), "")
                spacing_info = f" sp(b={before},a={after},l={line})"

        numPr_info = ""
        if pPr is not None:
            np = pPr.find(qn("w:numPr"))
            if np is not None:
                ilvl = np.find(qn("w:ilvl"))
                numId = np.find(qn("w:numId"))
                numPr_info = f" numPr(lvl={ilvl.get(qn('w:val')) if ilvl is not None else '?'},id={numId.get(qn('w:val')) if numId is not None else '?'})"

        print(f"p#{i} [{s}/{sid}]{spacing_info}{numPr_info}{run_info}")
        print(f"     {t[:130]}")
        count += 1
        if count > 80:
            break


if __name__ == "__main__":
    main()
