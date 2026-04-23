"""§5 polish pass 2:

1. After each lesson intro ("In this lesson..."), insert a short narrative
   paragraph describing the experiment setup, ending with a question.
2. Remove the old "Experiment Setup." header + three bullet paragraphs
   (now redundant — the narrative covers the same ground).
3. Rename every "What changed vs previous lesson" → "Experiment Setup".
4. Re-shade all §5 table headers to theme accent1 (4F81BD) with black text.
"""

from pathlib import Path
from copy import deepcopy

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

HEADER_FILL = "4F81BD"   # theme accent1
HEADER_TEXT = "000000"   # black

# --- Narrative paragraphs (keyed by lesson heading prefix) -----------------
# Each ends with a piquing question.
NARRATIVES = {
    "5.1": (
        "We train the same matrix-factorisation architecture twice on "
        "identical data — once minimising RMSE regression loss, once "
        "minimising BPR pairwise ranking loss — and compare Recall@10. "
        "Will the implicit ranking objective outperform the explicit "
        "point-predicting approach on ranking metrics?"
    ),
    "5.2": (
        "We pit three single-domain models (MF-BPR, NCF, LightGCN) against "
        "three CDR models (CMF, EMCDR, PTUPCDR) on the full 1M-user dataset "
        "where only 5.2% of users appear in both domains. "
        "Can CDR models leverage movie history to beat a graph-based "
        "recommender when bridge users are this scarce?"
    ),
    "5.3": (
        "We filter the dataset to users active in both domains, pushing "
        "overlap from 5.2% to 100%, while keeping everything else constant. "
        "Does removing the overlap bottleneck finally let CDR models close "
        "the gap with LightGCN?"
    ),
    "5.4": (
        "We tighten the source-domain filter from ≥5 to ≥10 movie ratings "
        "per user while keeping 100% overlap. "
        "Does richer source history give CDR mapping functions better "
        "embeddings to transfer from?"
    ),
    "5.5": (
        "We prune the movie catalog by removing items with fewer than 10 "
        "interactions, cutting 74% of titles while retaining 75% of "
        "interactions. "
        "Does a sharper, denser catalog improve cross-domain transfer, or "
        "does every model lose diversity?"
    ),
    "5.6": (
        "We split users 80/20, stripping the cold 20% of all game "
        "interactions at training time, and evaluate exclusively on these "
        "cold users. "
        "Can CDR models transfer movie knowledge to rank a user's very first "
        "game, or does a popularity baseline prove hard to beat?"
    ),
    "5.7": (
        "We add SBERT and SBERT-CDR to the model comparison, encoding item "
        "metadata into a shared semantic space. "
        "Can content-based embeddings rescue the niche, long-tail users that "
        "collaborative filters consistently miss?"
    ),
    "5.8": (
        "We layer a training-free co-occurrence rerank on top of every "
        "model's existing scores, using only how often a movie and a game "
        "are co-liked by the same users. "
        "Can this zero-cost post-processing lift all models at once without "
        "any retraining?"
    ),
}


def _run(text, *, bold=False, italic=False, font="Arial", sz="20"):
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    for attr in ("ascii", "hAnsi", "cs"):
        rFonts.set(qn(f"w:{attr}"), font)
    rPr.append(rFonts)
    if bold:
        rPr.append(OxmlElement("w:b"))
    if italic:
        rPr.append(OxmlElement("w:i"))
    for tag in ("w:sz", "w:szCs"):
        e = OxmlElement(tag)
        e.set(qn("w:val"), sz)
        rPr.append(e)
    r.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    return r


def _make_narrative_paragraph(text):
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "120")
    spacing.set(qn("w:after"), "120")
    spacing.set(qn("w:line"), "324")
    spacing.set(qn("w:lineRule"), "auto")
    pPr.append(spacing)
    p.append(pPr)
    p.append(_run(text))
    return p


def _para_text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def _retext(el, text, *, bold=False):
    for r in list(el.findall(qn("w:r"))):
        el.remove(r)
    el.append(_run(text, bold=bold))


def _is_heading(el, level_ids):
    pPr = el.find(qn("w:pPr"))
    if pPr is None:
        return False
    pStyle = pPr.find(qn("w:pStyle"))
    if pStyle is None:
        return False
    return pStyle.get(qn("w:val")) in level_ids


def _apply_header_style(tbl_el):
    """Theme accent1 fill + black text on first row."""
    first_row = tbl_el.find(qn("w:tr"))
    if first_row is None:
        return 0
    count = 0
    for tc in first_row.findall(qn("w:tc")):
        # Cell shading
        tcPr = tc.find(qn("w:tcPr"))
        if tcPr is None:
            tcPr = OxmlElement("w:tcPr")
            tc.insert(0, tcPr)
        for old in tcPr.findall(qn("w:shd")):
            tcPr.remove(old)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), HEADER_FILL)
        tcPr.append(shd)
        # Text color → black for every run in the cell
        for r in tc.iter(qn("w:r")):
            rPr = r.find(qn("w:rPr"))
            if rPr is None:
                rPr = OxmlElement("w:rPr")
                r.insert(0, rPr)
            for old in rPr.findall(qn("w:color")):
                rPr.remove(old)
            color = OxmlElement("w:color")
            color.set(qn("w:val"), HEADER_TEXT)
            rPr.append(color)
        count += 1
    return count


def main():
    doc = Document(str(DOC))
    body = doc.element.body

    H1 = ("Heading1", "heading1")
    H2 = ("Heading2", "heading2")

    # Collect §5 child elements (paragraphs + tables) in order.
    children = list(body)
    s5_start = s5_end = None
    for idx, ch in enumerate(children):
        tag = ch.tag.split("}", 1)[-1]
        if tag == "p" and _is_heading(ch, H1):
            txt = _para_text(ch).strip()
            if txt.startswith("5. Experiments"):
                s5_start = idx
            elif s5_start is not None and s5_end is None:
                s5_end = idx
    if s5_start is None:
        raise RuntimeError("§5 not found")
    if s5_end is None:
        s5_end = len(children)

    # --- Pass 1: identify which lesson each child belongs to ----------------
    lesson_key = None  # e.g. "5.1", "5.2"
    for idx in range(s5_start, s5_end):
        ch = children[idx]
        tag = ch.tag.split("}", 1)[-1]
        if tag == "p" and _is_heading(ch, H2):
            txt = _para_text(ch).strip()
            # Extract "5.X" prefix
            parts = txt.split(" ", 1)
            if parts[0].startswith("5.") and parts[0][-1].isdigit():
                lesson_key = parts[0]

    # --- Pass 2: do all edits -----------------------------------------------
    # Re-iterate since we'll mutate.
    # Strategy: collect actions first, then apply in reverse index order.

    # Rebuild the fresh list.
    children = list(body)
    lesson_key = None
    narratives_inserted = 0
    setups_removed = 0
    renamed = 0
    tables_styled = 0

    # We'll work through indices manually.
    idx = s5_start
    while idx < len(children):
        ch = children[idx]
        tag = ch.tag.split("}", 1)[-1]

        if tag == "p":
            txt = _para_text(ch).strip()

            # Track current lesson
            if _is_heading(ch, H2):
                parts = txt.split(" ", 1)
                if parts[0].startswith("5.") and parts[0].rstrip(".").replace(".", "").replace("5", "").isdigit():
                    lesson_key = parts[0]

            # Detect end of §5
            if _is_heading(ch, H1) and not txt.startswith("5."):
                break

            # Insert narrative after "In this lesson..." paragraph
            if txt.startswith("In this lesson") and lesson_key in NARRATIVES:
                narr = _make_narrative_paragraph(NARRATIVES[lesson_key])
                ch.addnext(narr)
                # Refresh children list since tree mutated
                children = list(body)
                narratives_inserted += 1
                idx += 2  # skip past inserted element
                continue

            # Remove "Experiment Setup." header + 3 bullet paragraphs
            if txt == "Experiment Setup.":
                # Remove this paragraph + next 3 (the bullets)
                to_remove = [ch]
                ci = idx + 1
                removed_count = 0
                while ci < len(children) and removed_count < 3:
                    nxt = children[ci]
                    ntag = nxt.tag.split("}", 1)[-1]
                    if ntag != "p":
                        break
                    ntxt = _para_text(nxt).strip()
                    if ntxt.startswith("•"):
                        to_remove.append(nxt)
                        removed_count += 1
                        ci += 1
                    elif ntxt == "":
                        ci += 1  # skip blanks
                    else:
                        break
                for el in to_remove:
                    body.remove(el)
                children = list(body)
                setups_removed += 1
                continue  # don't increment idx — element was removed

            # Rename "What changed vs previous lesson" → "Experiment Setup"
            if txt == "What changed vs previous lesson":
                _retext(ch, "Experiment Setup", bold=True)
                renamed += 1

        elif tag == "tbl":
            # Check if still in §5 (we stop at next Heading 1)
            n = _apply_header_style(ch)
            if n > 0:
                tables_styled += 1

        idx += 1

    doc.save(str(DOC))
    print(f"Narratives inserted: {narratives_inserted}")
    print(f"Old 'Experiment Setup.' + bullets removed: {setups_removed}")
    print(f"'What changed' → 'Experiment Setup' renamed: {renamed}")
    print(f"Tables re-styled (accent1 + black): {tables_styled}")
    print("✓ §5 polish v2 done")


if __name__ == "__main__":
    main()
