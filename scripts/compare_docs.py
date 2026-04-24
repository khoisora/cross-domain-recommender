"""Compare §5.1-5.2 structure and styling between redesigned and current doc."""

from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC_NEW = ROOT / "project_report_v2_redesigned.docx"
DOC_OLD = ROOT / "project_report_v2.docx"


def _dump_section(doc, label, start_prefix, stop_prefix):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    in_section = False
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        s = p.style.name
        sid = p.style.style_id

        if s == "Heading 2" and t.startswith(start_prefix):
            in_section = True
        elif s == "Heading 2" and in_section and not t.startswith(start_prefix[:3]):
            if t.startswith(stop_prefix):
                break

        if not in_section:
            continue

        # Get paragraph-level formatting
        pPr = p._p.find(qn("w:pPr"))
        spacing_info = ""
        ind_info = ""
        jc_info = ""
        if pPr is not None:
            sp = pPr.find(qn("w:spacing"))
            if sp is not None:
                before = sp.get(qn("w:before"), "")
                after = sp.get(qn("w:after"), "")
                line = sp.get(qn("w:line"), "")
                spacing_info = f" sp(b={before},a={after},l={line})"
            ind = pPr.find(qn("w:ind"))
            if ind is not None:
                left = ind.get(qn("w:left"), "")
                hang = ind.get(qn("w:hanging"), "")
                ind_info = f" ind(l={left},h={hang})"
            jc = pPr.find(qn("w:jc"))
            if jc is not None:
                jc_info = f" jc={jc.get(qn('w:val'))}"

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
                run_info = f" [{font} {sz}pt{bold}{ital}{color}]"

        # Check for numbering (bullets/lists)
        numPr = ""
        if pPr is not None:
            np = pPr.find(qn("w:numPr"))
            if np is not None:
                ilvl = np.find(qn("w:ilvl"))
                numId = np.find(qn("w:numId"))
                ilvl_v = ilvl.get(qn("w:val")) if ilvl is not None else "?"
                numId_v = numId.get(qn("w:val")) if numId is not None else "?"
                numPr = f" numPr(lvl={ilvl_v},id={numId_v})"

        print(f"p#{i} [{s}/{sid}]{spacing_info}{ind_info}{jc_info}{numPr}{run_info}")
        print(f"     {t[:120]}")


def main():
    doc_new = Document(str(DOC_NEW))
    doc_old = Document(str(DOC_OLD))

    # Lesson 1
    _dump_section(doc_new, "REDESIGNED — Lesson 1", "5.1", "5.2")
    _dump_section(doc_old, "CURRENT — Lesson 1", "5.1", "5.2")

    # Lesson 2
    _dump_section(doc_new, "REDESIGNED — Lesson 2", "5.2", "5.3")
    _dump_section(doc_old, "CURRENT — Lesson 2", "5.2", "5.3")


if __name__ == "__main__":
    main()
