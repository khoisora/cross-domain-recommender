"""Insert explainer figures (Fig 42–46) after each §5.9 sweep caption.

For each subsection, add:
  - one intuition paragraph ("Intuition: …")
  - the explainer PNG
  - its caption
Inserted bottom-up so earlier indices stay valid.
"""

from pathlib import Path
from docx import Document
from docx.shared import Inches

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"
FIG_DIR = ROOT / "report_figures_v3"


def insert_paragraph_after(paragraph, text=None, style=None):
    new_p = paragraph._parent.add_paragraph(text or "", style=style)
    paragraph._p.addnext(new_p._p)
    return new_p


def insert_explainer_after(caption_p, intuition_text, image_path, caption_text,
                            width_inches=6.0):
    """After the sweep caption, insert: [intuition] [img] [caption]."""
    # We must insert in reverse (caption first, then img, then intuition)
    # because each add_next pushes right after the anchor.
    cap_p = insert_paragraph_after(caption_p, caption_text, style="Caption")
    cap_p.alignment = 1
    img_p = insert_paragraph_after(caption_p)
    img_p.alignment = 1
    img_p.add_run().add_picture(str(image_path), width=Inches(width_inches))
    intu_p = insert_paragraph_after(caption_p, intuition_text)
    # leave default style/alignment


INSERTS = [
    # (caption_idx, intuition, image, new_caption)
    (246,
     "Intuition. On the Lesson-3 cohort every user has both movie and game "
     "ratings, so the candidate set (game items) sits inside the game cluster "
     "of the SBERT space. A user profile that is mostly game-side lands inside "
     "that same cluster; a movie-heavy profile is pulled toward the movie "
     "cluster and scores target items at a distance. Figure 46 visualises this "
     "with a single user: the winning profile (sw = 0.1) sits among the games, "
     "while the losing profile (sw = 0.9) is stuck in the movie cluster far "
     "from any candidate.",
     FIG_DIR / "fig_hp_sbert_explainer.png",
     "Figure 46: SBERT-CDR profile placement — source_weight controls where "
     "the user vector lands between the movie and game clusters; only the "
     "low-sw profile sits near the candidate items."),
    (242,
     "Intuition. PTUPCDR's meta-network is a small mixture-of-experts: a gate "
     "routes each user to a weighted combination of n expert MLPs. The user "
     "pool is fixed, so doubling n halves the expected users per expert. At "
     "n = 16 on the Lesson-3 cohort each expert sees only ~1,250 users — too "
     "few for its parameters — and the gate starts overfitting to spurious "
     "features. At n = 2 each expert trains on ~10,000 users, which is "
     "enough to generalise. Figure 45 makes the budget trade-off visible.",
     FIG_DIR / "fig_hp_ptupcdr_explainer.png",
     "Figure 45: PTUPCDR expert budget — with a fixed 20k-user pool, "
     "users-per-expert drops from ~10k at n = 2 to ~1.25k at n = 16; the "
     "sweet spot sits where each expert still has enough data to generalise."),
    (238,
     "Intuition. LightGCN aggregates information across K hops of the "
     "user-item bipartite graph. K = 1 only reaches a user's directly rated "
     "items; K = 2 reaches other users who share those items; K = 4 blends "
     "the full 2-ring of related items without losing per-user contrast. Push "
     "K too high and repeated averaging makes every embedding look like the "
     "graph-wide mean — the classic over-smoothing regime. Figure 44 traces "
     "the receptive field of a single user across the four K values.",
     FIG_DIR / "fig_hp_lightgcn_explainer.png",
     "Figure 44: LightGCN receptive field — starting from user u₀, the "
     "reachable neighbourhood grows with K; beyond K = 4 the propagation "
     "saturates and embeddings begin to collapse."),
    (234,
     "Intuition. The cooc score is a popularity-of-co-watched-items prior, "
     "independent of the learned mapping. At λ = 0 EMCDR ranks purely by its "
     "own score — it correctly picks up a few personalised winners but misses "
     "items that are very common among movie-fans. A small λ = 0.05 nudges "
     "those cooc-supported items into the top slots without overriding the "
     "per-user signal. Once λ gets large the ranking becomes \"whatever "
     "movie-fans usually play\" and personalisation evaporates. Figure 43 "
     "sketches the same top-5 under the three regimes.",
     FIG_DIR / "fig_hp_emcdr_explainer.png",
     "Figure 43: EMCDR + cooc three regimes — a light touch (λ = 0.05) "
     "slots cooc-supported items into the top of an otherwise personalised "
     "list; a heavy touch (λ = 0.3) collapses the ranking onto crowd "
     "favourites."),
    (230,
     "Intuition. Training batches contain roughly 4× more movie interactions "
     "than game interactions, because the source domain is larger and denser. "
     "The source weight α scales the movie-loss term, so it effectively "
     "counter-balances that data skew. At α = 0.5 movies still dominate the "
     "gradient (80% of batch × 0.5 = 40% vs 20% × 0.5 = 10%), and game "
     "predictions get under-trained; at α = 0.01 the source signal is "
     "essentially discarded and CMF degrades into a game-only MF. α = 0.2 "
     "hits the 50/50 balance and trains both domains equally. Figure 42 "
     "illustrates this arithmetic.",
     FIG_DIR / "fig_hp_cmf_explainer.png",
     "Figure 42: CMF gradient balance — because movies are 4× more plentiful "
     "than games in each batch, only α ≈ 0.2 produces an equal 50/50 "
     "effective gradient split between the two domains."),
]


def main():
    doc = Document(DOC)
    ps = doc.paragraphs
    # Sanity check
    assert "Figure 35" in ps[230].text, ps[230].text
    assert "Figure 39" in ps[246].text, ps[246].text

    for caption_idx, intu, img, new_cap in INSERTS:
        insert_explainer_after(ps[caption_idx], intu, img, new_cap)

    doc.save(DOC)
    print(f"Saved {DOC}")


if __name__ == "__main__":
    main()
