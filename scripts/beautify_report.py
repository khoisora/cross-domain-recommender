"""Cosmetic clean-up pass on project_report_v2.docx.

Changes (formatting only — no content additions or deletions):
  1. Fix heading hierarchy in section 4:
     - 4.X.Y subsections (MF-BPR, NCF, ..., Co-occurrence) → Heading 3.
     - Model sub-subheadings (Architecture/Training/Inference/…) → Heading 4.
  2. Delete the empty Heading 2 paragraph between §4.2.1 and §4.2.2.
  3. Collapse double-spaces in headings (e.g. "4.4  Content-…").
  4. Remove runs of ≥2 consecutive blank paragraphs in body sections
     (leaves the TOC region untouched).
"""

from pathlib import Path
import re
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

# Paragraph index (inclusive) from which non-TOC body starts — before this
# index we leave blank-line structure alone.
BODY_START = 92  # "1. Executive Summary"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# Sub-subheading titles that should become Heading 4 (bold-normal paragraphs
# and stray Heading 2 entries alike).
SUBSUB_TITLES = {
    _norm(t) for t in [
        "MF-BPR: Training process",
        "MF-BPR: Inference",
        "NCF (NeuMF): Architecture",
        "NCF (NeuMF): Training",
        "NCF (NeuMF): Inference",
        "LightGCN: Graph Architecture",
        "LightGCN: Training",
        "LightGCN: Inference",
        "CMF Architecture",
        "CMF Training",
        "CMF Inference",
        "CMF Limitations",
        "EMCDR Training",
        "EMCDR Inference",
        "PTUPCDR Training",
        "PTUPCDR Inference",
        "SBERT Encoding",
        "Cross-Domain Mechanism",
        "SBERT-CDR Inference",
        "Matrix Construction",
        "Co-occurrence reranking Inference",
    ]
}

# Top 4.X.Y subsections that should become Heading 3.
H3_PREFIX_RE = re.compile(r"^4\.\d+\.\d+\b")


def _set_style(p, style_name):
    try:
        p.style = p.part.document.styles[style_name]
    except KeyError:
        # Fall back silently if the target style is missing from the template.
        pass


def _remove_paragraph(p):
    p._p.getparent().remove(p._p)


def _has_image(p):
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
    return any(r._element.findall(f".//{ns}inline") for r in p.runs) or \
        any(r._element.findall(f".//{ns}anchor") for r in p.runs)


def _is_blank(p):
    return not p.text.strip() and not _has_image(p)


def _collapse_double_space(p):
    """If a heading text has '  ' (double space), rewrite it as a single space."""
    txt = p.text
    if "  " not in txt:
        return False
    new = re.sub(r" {2,}", " ", txt)
    # Preserve first run's formatting; rewrite its text, remove other runs.
    runs = list(p.runs)
    if not runs:
        return False
    runs[0].text = new
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    return True


def fix_headings(doc):
    """Pass 1: fix heading levels and delete the empty H2."""
    to_delete = []
    for p in doc.paragraphs:
        style = p.style.name
        text = _norm(p.text)

        # Delete empty Heading 2 placeholders anywhere in the doc.
        if style.startswith("Heading") and text == "":
            to_delete.append(p)
            continue

        # Promote 4.X.Y subsections to Heading 3.
        if H3_PREFIX_RE.match(text) and style == "Heading 2":
            _set_style(p, "Heading 3")
            continue

        # Sub-sub (model-phase) headings → Heading 4.
        if text in SUBSUB_TITLES:
            _set_style(p, "Heading 4")
            # Ensure runs are bold (H4 style should handle it, but be safe).
            for r in p.runs:
                r.bold = True
            continue

    for p in to_delete:
        _remove_paragraph(p)

    # Collapse double-spaces in every heading.
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading"):
            _collapse_double_space(p)


def collapse_blank_runs(doc):
    """Pass 2: body-section only. Reduce runs of ≥2 blank paras to 1."""
    paragraphs = doc.paragraphs
    # Recompute after structural edits — blank detection is dynamic.
    streak_start = None
    removals = []
    for i, p in enumerate(paragraphs):
        if i < BODY_START:
            streak_start = None
            continue
        if _is_blank(p):
            if streak_start is None:
                streak_start = i
        else:
            if streak_start is not None and i - streak_start > 1:
                # Keep first blank; remove the rest in the streak.
                for j in range(streak_start + 1, i):
                    removals.append(paragraphs[j])
            streak_start = None
    # Trailing streak at end of document.
    if streak_start is not None and len(paragraphs) - streak_start > 1:
        for j in range(streak_start + 1, len(paragraphs)):
            removals.append(paragraphs[j])

    for p in removals:
        _remove_paragraph(p)

    return len(removals)


def main():
    doc = Document(str(DOC))
    before_paras = len(doc.paragraphs)
    fix_headings(doc)
    removed = collapse_blank_runs(doc)
    after_paras = len(doc.paragraphs)
    doc.save(str(DOC))
    print(f"Paragraphs: {before_paras} → {after_paras} (removed {before_paras - after_paras})")
    print(f"Blank-streak collapses: {removed}")
    print(f"Saved {DOC}")


if __name__ == "__main__":
    main()
