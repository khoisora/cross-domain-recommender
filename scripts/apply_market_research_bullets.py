"""Rewrite §2.1 business-case and market-context sections:
  - concise intros
  - bullet points instead of long paragraphs
  - insert the market-research illustration after the market bullets
"""

from pathlib import Path
from docx import Document
from docx.shared import Inches

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"
FIG = ROOT / "report_figures_v3" / "fig_market_research.png"


BC_INTRO = (
    "Users rarely stay in one content category. Single-domain recommenders "
    "structurally ignore cross-category signal, which shows up as three concrete "
    "gaps:"
)

BC_BULLETS = [
    "Overlapping interests — action-movie fans often play action games; fantasy-film viewers often play RPGs. Single-domain models cannot use this link.",
    "Missed signals — source-domain behaviour (movies) is dense for most users; ignoring it throws away the best available preference data for a target-domain (games) cold-start.",
    "Cross-sell gap — without a cross-domain ranker, the platform cannot surface adjacent categories a user would naturally adopt.",
]

MARKET_INTRO = (
    "Major platforms already operate across multiple domains, but most still route "
    "each domain through a separate recommender. A working CDR system converts that "
    "existing multi-domain log into three measurable business outcomes — retention, "
    "cross-sell, and happy-surprise discovery (Figure M1)."
)

PROBLEM_TEXT = (
    "We study this on movies → games as a concrete analogue of the Netflix-plus-games "
    "setting: given users who rate items in both domains, does cross-domain signal "
    "measurably improve recommendation quality over single-domain baselines, and "
    "under which conditions? The practical deliverable is a routing rule that picks "
    "the right model family per user based on target-domain history depth and "
    "target-item popularity, wrapped in a prototype that serves the rule end-to-end."
)


def set_paragraph_text(p, text):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.add_run(text)


def insert_paragraph_after(paragraph, text=None, style=None):
    new_p = paragraph._parent.add_paragraph(text or "", style=style)
    paragraph._p.addnext(new_p._p)
    return new_p


def remove_paragraph(paragraph):
    paragraph._p.getparent().remove(paragraph._p)


def main():
    doc = Document(DOC)
    ps = doc.paragraphs

    # Sanity checks at current indices.
    assert "Why this matters" in ps[69].text, ps[69].text
    assert "Users rarely enjoy" in ps[70].text, ps[70].text
    assert "Market context" in ps[71].text, ps[71].text
    assert "Major platforms are already multi-domain" in ps[72].text, ps[72].text
    assert "Problem statement" in ps[73].text, ps[73].text
    assert "Given a platform" in ps[74].text, ps[74].text

    # 1) Problem statement — replace body (p74) with a tighter version.
    set_paragraph_text(ps[74], PROBLEM_TEXT)

    # 2) Market context — replace body (p72) with short intro and insert figure + caption after it.
    set_paragraph_text(ps[72], MARKET_INTRO)
    # Insert caption then image after the intro (bottom-up so image lands above caption).
    cap_p = insert_paragraph_after(ps[72],
                                    "Figure M1: Platforms are already multi-domain; a working CDR "
                                    "converts existing logs into retention, cross-sell, and happy-"
                                    "surprise outcomes.",
                                    style="Caption")
    cap_p.alignment = 1
    img_p = insert_paragraph_after(ps[72])
    img_p.alignment = 1
    img_p.add_run().add_picture(str(FIG), width=Inches(6.5))

    # 3) Business case — replace body (p70) with short intro + bullets.
    set_paragraph_text(ps[70], BC_INTRO)
    # Insert bullets in reverse so they appear in order right after p70.
    for bullet in reversed(BC_BULLETS):
        insert_paragraph_after(ps[70], bullet, style="List Bullet")

    doc.save(DOC)
    print(f"Saved {DOC}")


if __name__ == "__main__":
    main()
