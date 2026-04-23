"""Three-part docx update:

  1. Swap Figure 18 image (`cooc_matrix.png`) for the regenerated version
     that shows the correct formula `cooc[m][g] = log(1 + count)`, and
     remove the "(fix)" suffix from the caption.

  2. Insert a lessons-overview table at the top of §5 "Experiments and
     Results" that summarises all eight lessons.

  3. For each §5.N heading (N = 1..8), insert "What changed vs previous
     lesson", "Data characteristics", "Key findings", and a
     connecting-hook paragraph for the next lesson.
"""

from pathlib import Path
import shutil

from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"
FRESH_FIG18 = ROOT / "report_figures" / "cooc_matrix.png"
TARGET_FIG18 = ROOT / "report_figures_modern" / "cooc_matrix.png"


# ─── helpers ──────────────────────────────────────────────────────────

def set_paragraph_text(p, text):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.add_run(text)


def insert_paragraph_after(paragraph, text=None, style=None):
    new_p = paragraph._parent.add_paragraph(text or "", style=style)
    paragraph._p.addnext(new_p._p)
    return new_p


def insert_table_after(doc, anchor_para, data, style="Light Grid Accent 1"):
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    try:
        table.style = style
    except KeyError:
        pass
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            c = table.rows[i].cells[j]
            c.text = str(cell)
            if i == 0:
                for p in c.paragraphs:
                    for r in p.runs:
                        r.bold = True
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    anchor_para._p.addnext(table._tbl)
    return table


def find_heading(ps, needle, style_substr):
    for i, p in enumerate(ps):
        if needle in p.text and style_substr in p.style.name:
            return i
    raise LookupError(f"{needle!r} with style {style_substr!r} not found")


# ─── content ──────────────────────────────────────────────────────────

LESSONS_OVERVIEW = [
    ["#", "Lesson", "Isolated variable", "Cohort / split", "Key finding"],
    ["L1", "Explicit vs implicit",
     "Training loss (MSE vs BPR)",
     "1M users, 5.2% overlap, LLO",
     "BPR wins 5.2× on Recall@10 — ranking loss beats reconstruction loss."],
    ["L2", "Mixed-population baseline",
     "All six models on a realistic low-overlap cohort",
     "1M users, 5.2% overlap, LLO",
     "LightGCN leads; CDR underperforms at 5% overlap."],
    ["L3", "Overlap filtering",
     "User overlap % (5.2% → 100%)",
     "19,880 overlap users, LLO",
     "CDR recovers: PTUPCDR +276%, CMF +180% vs L2."],
    ["L4", "Source-rich / target-sparse",
     "Movie minimum (≥ 5 → ≥ 10)",
     "14,328 users, LLO",
     "PTUPCDR closes to −17% of LightGCN."],
    ["L5", "Catalog sharpening",
     "Movie popularity filter (≥ 10 ratings)",
     "14,328 users, 10,311 movies, LLO",
     "Balanced source catalog; CDR tightens toward L4."],
    ["L6", "Cold-start",
     "User-split 80/20, cold users have 0 games",
     "2,000 cold users, user-split",
     "CDR 4× over LightGCN on zero-game users; Popularity 0.0365 tops."],
    ["L7", "Content-aware CDR",
     "Semantic (SBERT) vs collaborative",
     "14,328 users, LLO",
     "SBERT wins 11× on niche items; SBERT-CDR matches LightGCN overall."],
    ["L8", "Co-occurrence reranking",
     "Training-free cross-domain rerank (λ = 0.05)",
     "L3 LLO + L6 cold-start",
     "Lifts every model; rescues LightGCN at cold-start 0.010 → 0.033."],
]

L5_INTRO = (
    "Each lesson isolates a single experimental variable to build understanding "
    "incrementally. We report Recall@10 (full-rank) as the primary metric; "
    "sampled HR@10 appears where relevant. Table 6 summarises the eight "
    "lessons; the per-lesson sections below detail what changed from the "
    "previous cohort, the dataset characteristics, the benchmark result, and a "
    "connecting hook to the next lesson."
)


# Per-lesson blocks. Order of insertion per lesson (right after heading):
#   1. What changed vs previous lesson (bold label + bullets)
#   2. Data characteristics (bullets)
#   3. Key findings (bullets)
#   4. Connecting hook (single paragraph at the end of the section)
#
# The existing body text + figure stays in place between the data characteristics
# and the key findings. The hook is appended after the existing last paragraph
# of that lesson (before the next Heading 2).

LESSONS = {
    1: {
        "what_changed_intro": "What changed vs Phase 0 (infrastructure-only):",
        "what_changed": [
            "Added the first two models — MF-Explicit (Surprise SVD) and MF-BPR — to produce the first end-to-end benchmark.",
            "Implicit conversion rule fixed: positive = rating ≥ 4 (POSITIVE_THRESHOLD); no user k-core filter; users randomly sampled to ≈1M.",
            "Evaluation: leave-last-out split on games, full-rank Recall@10 as primary metric (sampled HR@10 kept as a diagnostic).",
        ],
        "data_intro": "Data characteristics (L1 cohort):",
        "data": [
            "1,000,000 users · 45,933 movie items · 11,706 game items",
            "1,804,737 movie interactions · 337,803 game interactions (rating ≥ 4)",
            "52,281 overlap users (5.2%) · leave-last-out on games",
        ],
        "findings_intro": "Key findings:",
        "findings": [
            "MF-BPR 0.0130 Recall@10 vs MF-Explicit 0.0025 — BPR wins 5.2× on full-rank Recall@10 and 6.7× on NDCG@10.",
            "With all positives collapsed to rating ≥ 4, SVD has almost no variance to learn from; BPR's pairwise objective is the right fit for implicit signal.",
            "Absolute numbers are low (1.3% Recall@10 with no cross-domain signal) — motivates bringing in graph and CDR models in Lesson 2.",
        ],
        "hook": (
            "→ Lesson 2 expands the portfolio from two MF models to six (adding LightGCN, "
            "NCF, CMF, EMCDR, PTUPCDR) and runs them on the same naturally-sparse mixed "
            "population, asking whether CDR models can beat single-domain when only 5% "
            "of users overlap."
        ),
    },
    2: {
        "what_changed_intro": "What changed vs Lesson 1:",
        "what_changed": [
            "Model portfolio expanded from 2 → 6: + LightGCN, NCF, CMF, EMCDR, PTUPCDR (ported from MoviesGamesRecommender).",
            "Cohort and evaluation protocol kept identical to L1 — the only moving variable is model family.",
            "Introduces the natural-ratio baseline: a realistic platform sees low cross-domain overlap (5.2%), so CDR models must prove they help in this regime.",
        ],
        "data_intro": "Data characteristics (L2 cohort, same as L1):",
        "data": [
            "1,000,000 users · 45,933 movie / 11,706 game items",
            "5.2% overlap · LLO split on games",
        ],
        "findings_intro": "Key findings:",
        "findings": [
            "LightGCN 0.0290 leads; MF-BPR 0.0170; best CDR (EMCDR) 0.0160, PTUPCDR only 0.0085.",
            "All CDR models trail MF-BPR — at 5% overlap the cross-domain mapping has too few shared users to learn a useful transfer function.",
            "Sampled HR@10 ≈ 0.80–0.89 for NCF / EMCDR / PTUPCDR diverges sharply from full-rank Recall@10 — confirming sampled evaluation is too easy on sparse catalogues and should not be the primary metric.",
        ],
        "hook": (
            "→ Lesson 3 isolates user overlap as the variable: we restrict training and "
            "evaluation to the 19,880 users who have rated items in both domains, asking "
            "whether CDR recovers once every user is a bridge."
        ),
    },
    3: {
        "what_changed_intro": "What changed vs Lesson 2:",
        "what_changed": [
            "Added user k-core ≥ 10 total interactions + overlap filter (movies ≥ 5, games ≥ 1) → 100% overlap by construction.",
            "User count dropped from 1M (sampled, low overlap) to 19,880 (dense, full overlap) — same six models, same LLO protocol.",
            "Data-processing change: new `min_movie_ratings` / `min_game_ratings` params in process_data.py; benchmark_common.py now points at `processed_overlap/`.",
        ],
        "data_intro": "Data characteristics (L3 cohort):",
        "data": [
            "19,880 users · 40,288 movie / 10,034 game items",
            "430,736 movie interactions · 87,613 game interactions",
            "100% overlap · LLO split on games",
        ],
        "findings_intro": "Key findings:",
        "findings": [
            "Recall@10: LightGCN 0.0555 · PTUPCDR 0.0320 · EMCDR 0.0245 · NCF 0.0230 · CMF 0.0140 · MF-BPR 0.0050.",
            "Forcing 100% overlap lifts every CDR model dramatically: PTUPCDR +276%, CMF +180%, EMCDR +53% vs L2.",
            "LightGCN gap closes from −71% to −42% for PTUPCDR — CDR becomes competitive but does not overtake.",
            "MF-BPR collapses from 0.0170 to 0.0050 (−71%) on the smaller 87K-interaction cohort — pairwise sampling starves on too-small implicit data.",
        ],
        "hook": (
            "→ Lesson 4 stresses CDR toward its designed regime by tightening the movie "
            "minimum from ≥ 5 to ≥ 10 — keeping only source-rich users — and asks whether "
            "richer movie histories let the mapping-CDR family close the remaining gap."
        ),
    },
    4: {
        "what_changed_intro": "What changed vs Lesson 3:",
        "what_changed": [
            "Movie minimum raised from ≥ 5 → ≥ 10 (game minimum, item k-core, 100% overlap all kept).",
            "User count 19,880 → 14,328 (−5.5K movie-poor users); game interactions 87.6K → 58.8K (−33%).",
            "Benchmark pointer moved to `processed_sparse_loose/` — one default dataset, no cohort variants.",
        ],
        "data_intro": "Data characteristics (L4 cohort):",
        "data": [
            "14,328 users · 39,534 movie / 9,147 game items",
            "388,919 movie interactions · 58,782 game interactions",
            "100% overlap · LLO split on games",
        ],
        "findings_intro": "Key findings:",
        "findings": [
            "Recall@10: LightGCN 0.0350 · PTUPCDR 0.0290 · EMCDR 0.0230 · CMF 0.0160 · NCF 0.0140 · MF-BPR 0.0070.",
            "Gap to LightGCN: PTUPCDR −17% (was −42%), EMCDR −34% (was −56%), CMF −54% (was −75%) — CDR is closing steadily.",
            "LightGCN itself drops 37% (0.0555 → 0.0350): removing movie-poor / game-rich users strips away its strongest users; the cohort is now tilted toward the CDR regime.",
            "Trend across L2-L4 is monotonic for CDR: mixed → overlap → source-rich. Every tightening step helps mapping-CDR disproportionately more than single-domain.",
        ],
        "hook": (
            "→ Lesson 5 attacks source-side noise: most L4 movie items have only a "
            "handful of ratings from the overlap subset. Applying a movie-popularity "
            "filter (≥ 10 ratings within-cohort) tests whether a cleaner source catalog "
            "further improves CDR transfer."
        ),
    },
    5: {
        "what_changed_intro": "What changed vs Lesson 4:",
        "what_changed": [
            "Added a movie-popularity ≥ 10 filter (applied post-overlap) → 39,534 → 10,311 movie items (−74%).",
            "Source / target catalog rebalances from 4:1 (imbalanced, noisy) to ≈ 1:1 (balanced).",
            "Game side untouched (9,147 items, 58,782 interactions); user cohort unchanged at 14,328.",
        ],
        "data_intro": "Data characteristics (L5 cohort):",
        "data": [
            "14,328 users · 10,311 movie / 9,147 game items",
            "282,896 movie interactions · 58,782 game interactions",
            "99.9% overlap · LLO split on games",
        ],
        "findings_intro": "Key findings:",
        "findings": [
            "Recall@10: LightGCN 0.0315 · PTUPCDR 0.0255 · NCF 0.0215 · EMCDR 0.0210 · CMF 0.0155 · MF-BPR 0.0070.",
            "CDR models hold most of the L4 gain: CMF −3%, EMCDR −9%, PTUPCDR −12% vs L4 — cleaner source catalog outperforms the much larger noisy one.",
            "NCF jumps +54% (0.0140 → 0.0215) — a tighter item space helps neural interaction learning even on a game-only model.",
            "LightGCN dips −10% as the movie side of the bipartite graph loses neighborhood diversity; game-only metrics are unaffected because the game catalog is untouched.",
        ],
        "hook": (
            "→ Lesson 6 introduces the regime CDR was designed for: a user-split "
            "protocol in which 20% of users have zero game history at training time. "
            "We ask whether cross-domain transfer can rank games for users the "
            "collaborative models have never seen."
        ),
    },
    6: {
        "what_changed_intro": "What changed vs Lesson 5:",
        "what_changed": [
            "Dataset reverts to processed_overlap (L3 cohort — movies ≥ 5) because richer movie history is what feeds the transfer mapping.",
            "Split changed from LLO on games to user-split 80/20: 10,124 warm users keep their full game history; 2,530 cold users have every game interaction removed (2,000 evaluated).",
            "Added a Popularity baseline (computed on warm-user counts) as the natural first-game prediction for users with no game history.",
        ],
        "data_intro": "Data characteristics (L6 cohort):",
        "data": [
            "19,880 total users → 10,124 warm train / 2,000 cold eval",
            "Movie interactions (all users): 430,736 · Game train interactions (warm only): 64,057",
            "Protocol: user-split cold-start (cold users have 0 games at train time)",
        ],
        "findings_intro": "Key findings:",
        "findings": [
            "Recall@10 on cold users: Popularity 0.0365 · PTUPCDR 0.0305 · EMCDR 0.0300 · LightGCN 0.0080 · NCF 0.0020 · MF-BPR 0.0000.",
            "Mapping-based CDR delivers ~4× over LightGCN on zero-game users — the direct cold-start routing rule: 0 games → EMCDR / PTUPCDR.",
            "Popularity tops the table (0.0365): first-game choice is often a well-known title, so a strong prior on popular items beats personalised cross-domain signal at true cold-start.",
            "Single-domain MF-BPR and NCF collapse to ~0 — no game training data means no usable signal at all; CDR is the only family able to rank for these users.",
        ],
        "hook": (
            "→ Lesson 7 turns to a complementary cold-start regime — users who do have "
            "some game history but whose taste is niche. We bring in a content-based "
            "bridge (SBERT) and ask whether semantic similarity can rescue the long-tail "
            "items that every collaborative model is blind to."
        ),
    },
    7: {
        "what_changed_intro": "What changed vs Lesson 6:",
        "what_changed": [
            "Split reverts to LLO on games (we are no longer testing zero-history cold-start); dataset moves to processed_sparse_loose (L4 cohort, movies ≥ 10).",
            "Adds two new models: SBERT (content-only single-domain, movie→movie similarity in 384-dim SBERT space) and SBERT-CDR (user profile built from movie + game item embeddings, scored against game items).",
            "Analysis shifts from overall ranking to subgroup splits — niche vs popular test items, cold vs movie-heavy users.",
        ],
        "data_intro": "Data characteristics (L7 cohort):",
        "data": [
            "14,328 users · 39,534 movie / 9,147 game items (same as L4)",
            "SBERT encoder: all-MiniLM-L6-v2 (384-dim)",
            "100% overlap · LLO on games · subgroup split by target-item popularity",
        ],
        "findings_intro": "Key findings:",
        "findings": [
            "Recall@10 overall: LightGCN 0.0335 · SBERT-CDR 0.0330 · SBERT 0.0310 · PTUPCDR 0.0220 — SBERT-CDR matches LightGCN within 1.5% with zero training.",
            "On the `one_shot_unpopular_target` subgroup (n = 82): SBERT 0.1341 vs LightGCN 0.0122 — an 11× win for content on niche long-tail items.",
            "Semantic similarity captures a signal collaborative models cannot: niche users have genre-consistent preferences (horror-fan stays horror) that interaction counts miss but SBERT embeddings preserve.",
            "SBERT-CDR is the routing rule's answer for any user whose target item is likely long-tail, regardless of history depth.",
        ],
        "hook": (
            "→ Lesson 8 asks whether a single training-free layer can improve every "
            "model in every regime. A cross-domain co-occurrence rerank (λ = 0.05 blend "
            "with the base score) is evaluated on the L3 LLO cohort and the L6 cold-"
            "start split — the two regimes that matter most for the routing rule."
        ),
    },
    8: {
        "what_changed_intro": "What changed vs Lesson 7:",
        "what_changed": [
            "No new models — a post-processing layer applied on top of every model's scores.",
            "Co-occurrence matrix built once from training data: cooc[m][g] = log(1 + count of overlap users who rated both m and g).",
            "Inference blend: final_score(u, g) = base_score(u, g) + λ · Σ_{m ∈ user-movies} cooc[m][g], with λ = 0.05.",
        ],
        "data_intro": "Data characteristics (L8 evaluation):",
        "data": [
            "Evaluated on two cohorts: L3 LLO (100% overlap, 19,880 users) and L6 cold-start (2,000 cold users).",
            "No retraining — the rerank layer is computed from the same training data every base model already saw.",
        ],
        "findings_intro": "Key findings:",
        "findings": [
            "LLO (L3): every model improves — e.g. EMCDR +31% on Recall@10 at λ = 0.05.",
            "Cold-start (L6): rescues single-domain LightGCN 0.010 → 0.033 (+230%), and MF-BPR 0.001 → 0.020 — the layer is effectively doing the cold-start transfer that these models could not do themselves.",
            "CDR mapping models gain only marginally at cold-start (already near the popularity ceiling), but cost is zero — rerank is universally dominant.",
            "λ tuning is sharp: the layer helps monotonically up to λ ≈ 0.05–0.10, then personalisation collapses as the rerank starts overriding base scores.",
        ],
        "hook": (
            "→ §5.9 turns from comparing models to tuning them: the hyperparameter "
            "sweeps that produced the defaults used throughout §5.1–§5.8 are reported "
            "there, documenting the sensitivity of each model to its key knobs."
        ),
    },
}


# ─── main ─────────────────────────────────────────────────────────────

def fix_fig18(doc):
    """Swap the Figure 18 image for the regenerated cooc_matrix.png and clean the caption."""
    # Ensure the fresh image is mirrored at the canonical path the report uses.
    shutil.copyfile(FRESH_FIG18, TARGET_FIG18)

    ps = doc.paragraphs
    img_para = ps[340]
    caption_para = ps[341]
    assert "Figure 18" in caption_para.text, caption_para.text

    # Remove the existing picture run(s).
    for r in list(img_para.runs):
        r._element.getparent().remove(r._element)

    # Add a fresh picture run pointing at the regenerated image.
    new_run = img_para.add_run()
    new_run.add_picture(str(TARGET_FIG18), width=Inches(6.0))
    img_para.alignment = 1  # centered

    # Clean "(fix)" from the caption.
    new_caption = caption_para.text.replace("(fix)", "").strip()
    set_paragraph_text(caption_para, new_caption)


BULLET = "•  "


def expand_lesson(doc, heading_idx, lesson_num):
    """Insert per-lesson content right after the §5.N heading and append a
    connecting-hook paragraph just before the next Heading 2."""
    ps = doc.paragraphs
    block = LESSONS[lesson_num]
    heading = ps[heading_idx]

    # Items to insert immediately after the heading, in display order.
    # Tuple = (text, bold_label)
    after_heading_items = []
    after_heading_items.append((block["what_changed_intro"], True))
    for b in block["what_changed"]:
        after_heading_items.append((BULLET + b, False))
    after_heading_items.append((block["data_intro"], True))
    for b in block["data"]:
        after_heading_items.append((BULLET + b, False))

    # Insert in reverse so paragraphs land in order right after the heading.
    for text, bold_label in reversed(after_heading_items):
        new_p = insert_paragraph_after(heading, text)
        if bold_label:
            for r in new_p.runs:
                r.bold = True

    # "Key findings" + hook go at the end of the section, right before the next
    # Heading 2.
    ps2 = doc.paragraphs
    target_text = LESSONS_HEADINGS[lesson_num]
    section_end_idx = None
    for j, p in enumerate(ps2):
        if p.text.strip() == target_text and "Heading 2" in p.style.name:
            section_end_idx = j
            break
    if section_end_idx is None:
        raise LookupError(f"could not re-locate heading for lesson {lesson_num}")

    next_text = LESSONS_HEADINGS.get(lesson_num + 1, "5.9 Hyperparameter Analysis")
    next_idx = None
    for j in range(section_end_idx + 1, len(ps2)):
        if ps2[j].text.strip().startswith(next_text) and "Heading 2" in ps2[j].style.name:
            next_idx = j
            break
    if next_idx is None:
        raise LookupError(f"could not re-locate next heading after lesson {lesson_num}")

    anchor_before_next = ps2[next_idx - 1]

    hook_items = [(block["findings_intro"], True)]
    for b in block["findings"]:
        hook_items.append((BULLET + b, False))
    hook_items.append((block["hook"], False))

    for text, bold_label in reversed(hook_items):
        new_p = insert_paragraph_after(anchor_before_next, text)
        if bold_label:
            for r in new_p.runs:
                r.bold = True


LESSONS_HEADINGS = {
    1: "5.1 Lesson 1: Explicit vs Implicit Ranking",
    2: "5.2 Lesson 2: Low Overlap Kills CDR",
    3: "5.3 Lesson 3: Overlap Filtering Rescues CDR",
    4: "5.4 Lesson 4: Source-Rich / Target-Sparse",
    5: "5.5 Lesson 5: Catalog Sharpening",
    6: "5.6 Lesson 6: Cold-Start Protocol",
    7: "5.7 Lesson 7: Content-Aware CDR (SBERT)",
    8: "5.8 Lesson 8: Co-occurrence Reranking",
}


def insert_lessons_overview(doc):
    """Rewrite §5 intro paragraph and add the lessons-overview table."""
    ps = doc.paragraphs
    h5_idx = find_heading(ps, "5. Experiments and Results", "Heading 1")
    intro_para = ps[h5_idx + 1]  # the existing "Each lesson isolates..." paragraph

    # Replace intro with the table-aware version.
    set_paragraph_text(intro_para, L5_INTRO)

    # Insert caption + table right after the intro.
    cap = insert_paragraph_after(
        intro_para,
        "Table 6: Eight-lesson summary — each lesson isolates one variable from the previous.",
    )
    for r in cap.runs:
        r.italic = True
    insert_table_after(doc, cap, LESSONS_OVERVIEW)


def main():
    doc = Document(DOC)

    # 1. Figure 18 swap + caption cleanup.
    fix_fig18(doc)

    # 2-3. Expand each lesson bottom-up (L8 → L1) so earlier indices remain valid.
    #      We re-locate headings by text inside expand_lesson.
    for n in sorted(LESSONS.keys(), reverse=True):
        # Locate the §5.N heading fresh each pass.
        ps = doc.paragraphs
        heading_text = LESSONS_HEADINGS[n]
        heading_idx = None
        # Only match the §5 occurrence (Heading 2 style, not the TOC entries which are "normal").
        for i, p in enumerate(ps):
            if p.text.strip() == heading_text and "Heading 2" in p.style.name:
                heading_idx = i
                break
        if heading_idx is None:
            raise LookupError(f"lesson {n} heading not found: {heading_text}")
        expand_lesson(doc, heading_idx, n)

    # 4. Lessons-overview table at the top of §5.
    insert_lessons_overview(doc)

    doc.save(DOC)
    print(f"Saved {DOC}")


if __name__ == "__main__":
    main()
