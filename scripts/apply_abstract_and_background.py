"""Rewrite Executive Summary p56 (drop 'lesson-by-lesson', keep 3-family framing)
and expand §2.1 Background & Problem Statement with market/business case drawn
from the proposal slides.
"""

from pathlib import Path
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"


ABSTRACT_P1 = (
    "Abstract. This project investigates cross-domain recommendation (CDR) for "
    "transferring user preferences from movies to video games on the Amazon Reviews "
    "2023 dataset. The central question is: can a user's movie history improve game "
    "recommendations, especially at cold-start? We run a systematic comparison "
    "benchmark across three model families — single-domain collaborative filtering "
    "(MF-BPR, NCF, LightGCN), cross-domain transfer (CMF, EMCDR, PTUPCDR), and "
    "content-based (SBERT, SBERT-CDR) — evaluated head-to-head on six purpose-built "
    "data cohorts."
)


# ─── §2.1 Background and Problem Statement ─────────────────────────────

BG_INTRO = (
    "Most production recommendation systems are single-domain: they recommend items "
    "based solely on a user's interaction history within that one domain. "
    "Cross-Domain Recommendation (CDR) relaxes this constraint — it transfers "
    "knowledge from a source domain (movies) to a target domain (games), which is "
    "especially valuable at cold-start: when a user has little or no target-domain "
    "history, CDR can still rank target items by leveraging their source-domain "
    "preferences."
)

BG_MOTIVATION_HEADING = ("Why this matters: the business case", "Heading 3")

BG_MOTIVATION = (
    "Users rarely enjoy just one type of content. Action-movie fans often enjoy "
    "action games; fantasy-film viewers frequently play RPGs. Ignoring this "
    "cross-domain behaviour leaves preference signal on the table — signal that "
    "a single-domain recommender structurally cannot use. The business "
    "consequences are three-fold: (i) lower engagement, because the top-K list "
    "misses items the user would have enjoyed; (ii) a harder cold-start, because "
    "single-domain models cannot recommend at all to users with zero target-domain "
    "interactions; and (iii) a missed cross-sell opportunity, because the system "
    "cannot surface adjacent categories a user would naturally adopt."
)

BG_MARKET_HEADING = ("Market context", "Heading 3")

BG_MARKET = (
    "Major platforms are already multi-domain. Netflix expanded from video "
    "streaming into mobile games in 2021, so its catalog now spans two distinct "
    "entertainment modalities under a single account. Amazon operates personalised "
    "cross-category product recommendation across its full e-commerce catalog. "
    "Spotify bridges music and podcast discovery in a shared feed. In each case "
    "the platform owns rich cross-domain interaction logs but, to first order, "
    "still routes those logs through domain-specific recommenders. A working CDR "
    "system converts those existing logs into three business outcomes: "
    "higher retention (more relevant recommendations keep users on-platform "
    "longer), cross-sell lift (users are introduced to adjacent categories they "
    "would naturally enjoy), and the \"happy surprise\" effect (the recommender "
    "surfaces items outside the user's active category that nonetheless land well)."
)

BG_PROBLEM_HEADING = ("Problem statement", "Heading 3")

BG_PROBLEM = (
    "Given a platform where users rate items in two entertainment domains — movies "
    "and video games, as a concrete analogue of the Netflix-plus-games setting — "
    "we ask whether cross-domain signal measurably improves recommendation quality "
    "over single-domain baselines, and under which conditions. The practical "
    "deliverable is a routing rule that tells the system which model family to "
    "apply to a given user based on their target-domain history depth and the "
    "popularity of target items, together with a prototype system that serves the "
    "rule end-to-end."
)


def set_paragraph_text(p, text):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.add_run(text)


def insert_paragraph_after(paragraph, text=None, style=None):
    new_p = paragraph._parent.add_paragraph(text or "", style=style)
    paragraph._p.addnext(new_p._p)
    return new_p


def main():
    doc = Document(DOC)
    ps = doc.paragraphs
    assert ps[56].text.startswith("Abstract."), ps[56].text[:80]
    assert "2.1 Background" in ps[67].text
    assert "Most recommendation systems are single-domain" in ps[68].text

    # 1. Replace Abstract (p56).
    set_paragraph_text(ps[56], ABSTRACT_P1)

    # 2. Replace §2.1 body (p68) with BG_INTRO, then append motivation/market/problem sections.
    set_paragraph_text(ps[68], BG_INTRO)

    # Insert new paragraphs after p68 in reverse so they land in order.
    blocks = [
        BG_MOTIVATION_HEADING,
        (BG_MOTIVATION, None),
        BG_MARKET_HEADING,
        (BG_MARKET, None),
        BG_PROBLEM_HEADING,
        (BG_PROBLEM, None),
    ]
    for text, style in reversed(blocks):
        insert_paragraph_after(ps[68], text, style=style)

    doc.save(DOC)
    print(f"Saved {DOC}")


if __name__ == "__main__":
    main()
