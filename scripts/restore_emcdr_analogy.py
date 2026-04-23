"""Restore the EMCDR analogy paragraph that was dropped during the EMCDR
rewrite.

Inserts a single "Analogy." paragraph at the top of the §4.3.2 EMCDR
Architecture subsection, before "Two independent embedding spaces…".
The analogy extends the translator metaphor already used in §4 overview
and mirrors the "Meet Maya" narrative style.
"""

from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

ANALOGY_LABEL = "Analogy. "
ANALOGY_BODY = (
    "Imagine Maya has two regulars — one who only rents movies and one who "
    "only plays games — and she wants to recommend games to the movie regular. "
    "CMF tried to describe both tastes with one profile card. EMCDR does "
    "something different: it keeps two separate profile cards per customer "
    "(one written in “movie language”, one written in “game language”) and "
    "hires a bilingual translator who has studied the small group of "
    "customers who rent both. When a movie-only customer walks in, Maya "
    "reads their movie card, hands it to the translator, and gets back the "
    "same taste rewritten in game language — which she can now match "
    "against her game shelf. The translator is the mapping network f_θ; "
    "the bilingual customers who taught it are the overlap users."
)


def _anchor_heading(doc):
    for p in doc.paragraphs:
        if (p.style.name == "Heading 4"
                and p.text.strip() == "EMCDR Architecture"):
            return p
    raise RuntimeError("EMCDR Architecture heading not found")


def _make_label_run(text):
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")

    rFonts = OxmlElement("w:rFonts")
    for attr in ("ascii", "hAnsi", "cs"):
        rFonts.set(qn(f"w:{attr}"), "Arial")
    rPr.append(rFonts)

    b = OxmlElement("w:b")
    rPr.append(b)

    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "20")
    rPr.append(sz)
    szCs = OxmlElement("w:szCs")
    szCs.set(qn("w:val"), "20")
    rPr.append(szCs)

    r.append(rPr)

    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    return r


def _make_body_run(text):
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")

    rFonts = OxmlElement("w:rFonts")
    for attr in ("ascii", "hAnsi", "cs"):
        rFonts.set(qn(f"w:{attr}"), "Arial")
    rPr.append(rFonts)

    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "20")
    rPr.append(sz)
    szCs = OxmlElement("w:szCs")
    szCs.set(qn("w:val"), "20")
    rPr.append(szCs)

    r.append(rPr)

    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    return r


def _make_analogy_paragraph():
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    pStyle = OxmlElement("w:pStyle")
    pStyle.set(qn("w:val"), "normal")
    pPr.append(pStyle)

    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "120")
    spacing.set(qn("w:after"), "120")
    spacing.set(qn("w:line"), "324")  # ≈1.35 line spacing (240 × 1.35)
    spacing.set(qn("w:lineRule"), "auto")
    pPr.append(spacing)

    p.append(pPr)

    p.append(_make_label_run(ANALOGY_LABEL))
    p.append(_make_body_run(ANALOGY_BODY))
    return p


def main():
    doc = Document(str(DOC))
    anchor = _anchor_heading(doc)

    # Skip if the analogy has already been inserted.
    nxt = anchor._p.getnext()
    while nxt is not None and nxt.tag.endswith("}p"):
        nxt_text = "".join(t.text or "" for t in nxt.iter(qn("w:t")))
        if nxt_text.startswith("Analogy."):
            print("Analogy already present — skipping.")
            return
        # Stop scanning past the first non-empty paragraph.
        if nxt_text.strip():
            break
        nxt = nxt.getnext()

    new_p = _make_analogy_paragraph()
    anchor._p.addnext(new_p)

    doc.save(str(DOC))
    print("✓ Inserted EMCDR analogy paragraph after 'EMCDR Architecture' heading")


if __name__ == "__main__":
    main()
