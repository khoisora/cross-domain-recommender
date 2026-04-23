"""Move the SBERT-CDR and Co-occurrence Reranking method descriptions out
of §4 Methodology and into the lessons that introduce them.

- §4.4.1 SBERT-CDR content → §5.7 Lesson 7 (Content-Aware CDR).
- §4.4.2 Co-occurrence Reranking content → §5.8 Lesson 8.
- The §4.4 Heading 2, §4.4.1 Heading 3, and §4.4.2 Heading 3 wrappers
  are deleted (the lesson headings already name the method).

Content inside each block (sub-subheadings, figures, captions, body paras)
is preserved verbatim — we only relocate the XML elements.
"""

from pathlib import Path
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

# Identifying strings for anchor paragraphs (match on stripped paragraph text).
H_44_TOP = "4.4 Content-Based and Post-Processing Models"
H_441 = "4.4.1 SBERT-CDR: Content-Based Semantic Matching"
H_442 = "4.4.2 Co-occurrence Reranking"

# Block boundaries are expressed as (start_text, end_text_inclusive_marker).
SBERT_START_TEXT = "A note on naming"
SBERT_END_TEXT_STARTSWITH = "The artifacts are simply the pre-computed item embedding matrix"

COOC_START_TEXT_STARTSWITH = "Matrix Construction"  # this is a heading
COOC_END_TEXT_STARTSWITH = "Key properties: Co-occurrence reranking is training-free"

# Lesson anchors — we insert method blocks immediately after the lesson's
# "Question." paragraph.
LESSON7_QUESTION_PREFIX = "Question.  Collaborative filters live and die"
LESSON8_QUESTION_PREFIX = "Question.  Every model so far is trained end-to-end"


def _strip(p):
    return p.text.strip()


def _find_para_index(doc, predicate):
    for i, p in enumerate(doc.paragraphs):
        if predicate(p):
            return i
    raise ValueError("anchor paragraph not found")


def _find_heading_index(doc, text, min_idx=0):
    """Find a Heading-styled paragraph whose stripped text equals `text`."""
    for i, p in enumerate(doc.paragraphs):
        if i < min_idx:
            continue
        if p.style.name.startswith("Heading") and _strip(p) == text:
            return i
    raise ValueError(f"heading not found: {text!r}")


def _collect_block(doc, start_idx, end_idx_inclusive):
    """Collect the XML elements for paragraphs in [start_idx, end_idx_inclusive]."""
    return [doc.paragraphs[i]._p for i in range(start_idx, end_idx_inclusive + 1)]


def _move_after(anchor_elem, elems_in_order):
    """Move `elems_in_order` so that, post-move, they sit directly after
    anchor_elem in their original order."""
    # Strategy: insert in reverse after the anchor so their final order is
    # preserved (each addnext puts the new element immediately after anchor).
    for el in reversed(elems_in_order):
        # Detach first (addnext on already-placed element will just re-parent).
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)
        anchor_elem.addnext(el)


def _remove(elem):
    parent = elem.getparent()
    if parent is not None:
        parent.remove(elem)


def main():
    doc = Document(str(DOC))

    # --- Locate everything while indices are still stable. Match on
    # Heading style to avoid hitting the TOC entries (which are `normal`
    # paragraphs that reproduce the same text). ---
    idx_h44 = _find_heading_index(doc, H_44_TOP)
    idx_h441 = _find_heading_index(doc, H_441)
    idx_h442 = _find_heading_index(doc, H_442)

    # SBERT block: from para after H_441 up to "The artifacts are simply…" paragraph.
    # We include the blank paragraph between heading and first sub-heading too.
    sbert_start = idx_h441 + 1
    sbert_end = _find_para_index(
        doc,
        lambda p: _strip(p).startswith(SBERT_END_TEXT_STARTSWITH),
    )

    # Cooc block: from the "Matrix Construction" heading up to and including the
    # "Key properties" paragraph.
    cooc_start = _find_para_index(
        doc,
        lambda p: _strip(p) == "Matrix Construction",
    )
    cooc_end = _find_para_index(
        doc,
        lambda p: _strip(p).startswith(COOC_END_TEXT_STARTSWITH),
    )

    # Lesson anchors.
    idx_l7_q = _find_para_index(
        doc, lambda p: _strip(p).startswith(LESSON7_QUESTION_PREFIX)
    )
    idx_l8_q = _find_para_index(
        doc, lambda p: _strip(p).startswith(LESSON8_QUESTION_PREFIX)
    )

    print(f"§4.4 heading         @ {idx_h44}")
    print(f"§4.4.1 SBERT-CDR     @ {idx_h441}, block {sbert_start}..{sbert_end}")
    print(f"§4.4.2 Cooc          @ {idx_h442}, block {cooc_start}..{cooc_end}")
    print(f"L7 Question          @ {idx_l7_q}")
    print(f"L8 Question          @ {idx_l8_q}")

    # --- Capture XML element references (stable across reorderings). ---
    sbert_elems = _collect_block(doc, sbert_start, sbert_end)
    cooc_elems = _collect_block(doc, cooc_start, cooc_end)
    h44_elem = doc.paragraphs[idx_h44]._p
    h441_elem = doc.paragraphs[idx_h441]._p
    h442_elem = doc.paragraphs[idx_h442]._p
    l7_anchor = doc.paragraphs[idx_l7_q]._p
    l8_anchor = doc.paragraphs[idx_l8_q]._p

    # --- Perform moves. Do cooc first so it doesn't disturb sbert indices. ---
    _move_after(l8_anchor, cooc_elems)
    _move_after(l7_anchor, sbert_elems)

    # --- Delete the now-empty wrapper headings. ---
    for el in (h441_elem, h442_elem, h44_elem):
        _remove(el)

    doc.save(str(DOC))
    print(f"Moved SBERT block ({len(sbert_elems)} paras) and Cooc block "
          f"({len(cooc_elems)} paras). Saved {DOC}")


if __name__ == "__main__":
    main()
