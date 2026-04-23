"""Three edits to §5 Experiments and Results:

1. Rewrite each "Question." paragraph as a "In this lesson, we will see...
   explore..." lesson intro.
2. Convert each "Experiment Setup." paragraph into a three-bullet list
   (Variable Changed / Held Constant / Expectation), using the existing
   "List Bullet" paragraph style. Bold labels inline.
3. Shade the first-row cells of every table inside §5 with a light blue
   fill (hex D9E7FB).
"""

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

LIGHT_BLUE = "D9E7FB"

# --- New lesson intros --------------------------------------------------------
# Each entry: (old_prefix, new_text). Prefix is enough to uniquely identify the
# Question paragraph — we use the first ~60 chars of the original body.
QUESTION_REWRITES = [
    (
        "Honestly, this is a basic distinction I should have internalised",
        "In this lesson, we will see why predicting a 1–5 rating is "
        "fundamentally different from ranking items a user picked over ones "
        "they skipped, and explore how choosing the wrong loss function on "
        "implicit-feedback data can silently poison every downstream comparison.",
    ),
    (
        "Real platforms rarely see every user rate items in every domain",
        "In this lesson, we will explore whether, at a realistic 5% user "
        "overlap, there is enough cross-domain signal for a CDR model to "
        "beat a single-domain graph recommender, or whether the mapping "
        "function starves on too few bridge users.",
    ),
    (
        "If low overlap is what suffocated CDR in Lesson 2",
        "In this lesson, we will see how much of the gap closes once every "
        "user is forced to be a bridge, and explore whether LightGCN remains "
        "unbeatable even when CDR models finally have enough overlap to breathe.",
    ),
    (
        "Overlap alone isn't the same as source richness",
        "In this lesson, we will explore whether keeping only users with "
        "deep movie histories — not just overlap — is what mapping-based CDR "
        "needs to finally catch up to LightGCN.",
    ),
    (
        "A catalog is only as sharp as its tail allows",
        "In this lesson, we will see what happens when the bottom 74% of "
        "rarely-rated movies are trimmed away, and explore whether the "
        "remaining signal is cleaner for cross-domain transfer or whether "
        "every model simply loses diversity.",
    ),
    (
        "The true test of CDR is the user the single-domain model has never seen",
        "In this lesson, we will explore the true test of CDR — users the "
        "single-domain model has never seen — and see whether cross-domain "
        "transfer can still rank their first game, or whether a simple "
        "popularity prior is hard to beat.",
    ),
    (
        "Collaborative filters live and die by interaction counts",
        "In this lesson, we will explore whether a semantic content bridge "
        "(SBERT) can rescue the long-tail niche users that every "
        "collaborative model misses.",
    ),
    (
        "Every model so far is trained end-to-end",
        "In this lesson, we will see whether a simple, training-free "
        "rerank — built only from how often a movie and a game are liked by "
        "the same users — can lift every model at once, without retraining "
        "or extra data.",
    ),
]


# --- Helpers ------------------------------------------------------------------
def _run_with_text(text, *, bold=False, italic=False, font="Arial", sz="20"):
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


def _retext(p_element, runs):
    """Replace all <w:r> children with the supplied run elements."""
    for r in list(p_element.findall(qn("w:r"))):
        p_element.remove(r)
    for r in runs:
        p_element.append(r)


def _make_bullet_paragraph(label, body):
    """Build a <w:p> styled as List Bullet with a bold label + body text."""
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    pStyle = OxmlElement("w:pStyle")
    pStyle.set(qn("w:val"), "ListBullet")
    pPr.append(pStyle)
    p.append(pPr)

    p.append(_run_with_text(f"{label} ", bold=True))
    p.append(_run_with_text(body))
    return p


def _split_setup_body(full_text):
    """Split an Experiment Setup paragraph's body into (var, held, expect).

    Input text looks like:
      "Experiment Setup. Variable Changed: X. Held Constant: Y.
       Expectation: Z."
    Newlines may or may not separate the three clauses.
    """
    # Drop the "Experiment Setup." prefix.
    if full_text.startswith("Experiment Setup."):
        body = full_text[len("Experiment Setup."):].strip()
    else:
        body = full_text.strip()

    # Normalise internal whitespace and soft breaks to single spaces so that
    # our keyword splitters work regardless of how the original was wrapped.
    body = body.replace("\n", " ").replace("\r", " ")
    while "  " in body:
        body = body.replace("  ", " ")

    # Anchor the three labels.
    labels = ("Variable Changed:", "Held Constant:", "Expectation:")
    positions = [body.find(lab) for lab in labels]
    if -1 in positions:
        return None  # not a well-formed setup
    # Slice.
    var_s, held_s, exp_s = positions
    var_text = body[var_s + len(labels[0]):held_s].strip().rstrip(".").strip()
    held_text = body[held_s + len(labels[1]):exp_s].strip().rstrip(".").strip()
    exp_text = body[exp_s + len(labels[2]):].strip()
    return var_text, held_text, exp_text


def _apply_header_shading(tbl):
    """Add <w:shd> w:fill=LIGHT_BLUE to every first-row cell."""
    first_row = tbl.find(qn("w:tr"))
    if first_row is None:
        return 0
    count = 0
    for tc in first_row.findall(qn("w:tc")):
        tcPr = tc.find(qn("w:tcPr"))
        if tcPr is None:
            tcPr = OxmlElement("w:tcPr")
            tc.insert(0, tcPr)
        # Remove any existing shd, then add our own.
        for existing in tcPr.findall(qn("w:shd")):
            tcPr.remove(existing)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), LIGHT_BLUE)
        tcPr.append(shd)
        count += 1
    return count


def _iter_s5_paragraphs_and_tables(doc):
    """Yield (kind, element) pairs ('p' or 'tbl') that fall within §5 in
    document order — stops at the §6 Heading 1."""
    body = doc.element.body
    in_s5 = False
    for child in body.iterchildren():
        tag = child.tag.split("}", 1)[-1]
        if tag == "p":
            pPr = child.find(qn("w:pPr"))
            pStyle = pPr.find(qn("w:pStyle")) if pPr is not None else None
            style_val = pStyle.get(qn("w:val")) if pStyle is not None else None
            # python-docx names "Heading 1" → XML style id "Heading1".
            is_h1 = style_val in ("Heading1", "heading1", "Heading 1")
            text = "".join(t.text or "" for t in child.iter(qn("w:t"))).strip()
            if is_h1:
                if text.startswith("5. Experiments"):
                    in_s5 = True
                    continue
                elif in_s5:
                    in_s5 = False
                    break
            if in_s5:
                yield "p", child
        elif tag == "tbl":
            if in_s5:
                yield "tbl", child


def main():
    doc = Document(str(DOC))

    items = list(_iter_s5_paragraphs_and_tables(doc))
    print(f"§5 contains {sum(1 for k, _ in items if k == 'p')} paragraphs "
          f"and {sum(1 for k, _ in items if k == 'tbl')} tables")

    # --- 1. Question rewrites ------------------------------------------------
    q_map = {prefix: new for prefix, new in QUESTION_REWRITES}
    question_hits = 0
    for kind, el in items:
        if kind != "p":
            continue
        text = "".join(t.text or "" for t in el.iter(qn("w:t")))
        if not text.startswith("Question."):
            continue
        body = text[len("Question."):].lstrip()
        matched = None
        for prefix, new in q_map.items():
            if body.startswith(prefix):
                matched = new
                break
        if matched is None:
            print(f"  !! no rewrite match: {body[:80]}...")
            continue
        # Replace runs: keep original "Question." bold label? The user wants
        # the opening to read as an intro sentence — drop the label, replace
        # with just the new body.
        _retext(el, [_run_with_text(matched)])
        question_hits += 1
    print(f"Rewrote {question_hits} Question. paragraphs")

    # --- 2. Experiment Setup → bullet list -----------------------------------
    setup_hits = 0
    for kind, el in items:
        if kind != "p":
            continue
        text = "".join(t.text or "" for t in el.iter(qn("w:t")))
        # Also capture line breaks as spaces for keyword search.
        text_for_split = text
        # Include w:br breaks explicitly (w:br between two w:t parts — .text
        # loses them). Walk child elements of each <w:r>.
        rebuilt = []
        for r in el.findall(qn("w:r")):
            for c in r.iter():
                lt = c.tag.split("}", 1)[-1]
                if lt == "t":
                    rebuilt.append(c.text or "")
                elif lt == "br":
                    rebuilt.append("\n")
        text_for_split = "".join(rebuilt)

        if not text_for_split.startswith("Experiment Setup."):
            continue
        parts = _split_setup_body(text_for_split)
        if parts is None:
            print(f"  !! malformed setup: {text_for_split[:80]}...")
            continue
        var_text, held_text, exp_text = parts

        parent = el.getparent()
        idx = list(parent).index(el)

        # Replace the whole "Experiment Setup." paragraph with a single bold
        # label header paragraph, then three bullet paragraphs below.
        header_p = OxmlElement("w:p")
        header_p.append(_run_with_text("Experiment Setup.", bold=True))

        bullets = [
            _make_bullet_paragraph("Variable Changed:", var_text),
            _make_bullet_paragraph("Held Constant:", held_text),
            _make_bullet_paragraph("Expectation:", exp_text),
        ]

        # Remove the original paragraph.
        parent.remove(el)
        # Insert header + 3 bullets at the same index.
        for offset, new_p in enumerate([header_p, *bullets]):
            parent.insert(idx + offset, new_p)

        setup_hits += 1
    print(f"Converted {setup_hits} Experiment Setup paragraphs to bullets")

    # --- 3. Shade §5 table headers ------------------------------------------
    # Re-iterate fresh (mutated tree).
    items2 = list(_iter_s5_paragraphs_and_tables(doc))
    shaded = 0
    for kind, el in items2:
        if kind != "tbl":
            continue
        n = _apply_header_shading(el)
        if n > 0:
            shaded += 1
    print(f"Shaded headers on {shaded} §5 tables")

    doc.save(str(DOC))
    print("✓ §5 polished")


if __name__ == "__main__":
    main()
