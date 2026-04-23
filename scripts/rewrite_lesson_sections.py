"""Rebuild §5 + extend §6 of project_report_v2.docx with rich formatting.

Restores the pre-expansion backup, then re-applies:
  1. Figure 18 image swap (regenerated cooc_matrix).
  2. Lessons-overview table (§5 intro) with header-shading + zebra rows.
  3. For each §5.N (N = 1..8):
        - a plain opening-argument paragraph (bold "Question." + italic body),
        - "What changed vs previous" as a 3-col zebra table,
        - "Data characteristics" as a 3-col zebra table (Property / Previous / This lesson Δ),
        - the main lesson illustration (report_figures_v3/fig_lessonN_*.png),
        - existing body + bench graph (preserved),
        - "Key findings" as a 2-col zebra table,
        - a per-lesson key-findings comparison graph (fig_findings_lessonN.png),
        - spacer + a formatted "→ Next up." hook callout.
  4. A new §5.10 "Lessons Recap" with multi-paragraph synthesis.
  5. A new §6.3 "Recommendation Rows Design" with row-by-row rationale.

Run: python scripts/rewrite_lesson_sections.py
"""

from pathlib import Path
import shutil

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"
BACKUP = Path("/tmp/project_report_v2_backup_before_expansions.docx")
FRESH_FIG18 = ROOT / "report_figures" / "cooc_matrix.png"
TARGET_FIG18 = ROOT / "report_figures_modern" / "cooc_matrix.png"
ILLUS_DIR = ROOT / "report_figures_v3"

HEADER_FILL = "2E75B6"
HEADER_TEXT = RGBColor(0xFF, 0xFF, 0xFF)
ZEBRA_FILL = "F2F2F2"
HOOK_FILL = "FFF4D6"
HOOK_BORDER = "F0A202"
HOOK_TEXT = RGBColor(0x7A, 0x54, 0x00)
DELTA_POS_COLOR = RGBColor(0x10, 0x98, 0x1F)   # green
DELTA_NEG_COLOR = RGBColor(0xC0, 0x39, 0x2B)   # red


# ─── XML / cell helpers ─────────────────────────────────────────────────

def _set_cell_shading(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    for shd in tcPr.findall(qn("w:shd")):
        tcPr.remove(shd)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tcPr.append(shd)


def _set_cell_border(cell, left=None, top=None, right=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    for existing in tcPr.findall(qn("w:tcBorders")):
        tcPr.remove(existing)
    tcBorders = OxmlElement("w:tcBorders")
    for side, spec in (("left", left), ("top", top), ("right", right), ("bottom", bottom)):
        if spec is None:
            continue
        size, color = spec
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:color"), color)
        tcBorders.append(el)
    tcPr.append(tcBorders)


def _set_runs(cell, bold=False, size=Pt(9), color=None):
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = bold
            r.font.size = size
            if color is not None:
                r.font.color.rgb = color


def _write_cell(cell, text, bold=False, size=Pt(9), color=None):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = bold
    run.font.size = size
    if color is not None:
        run.font.color.rgb = color


def _write_cell_colored_parts(cell, parts, size=Pt(9)):
    """parts = [(text, bold, color_or_None), ...]"""
    cell.text = ""
    p = cell.paragraphs[0]
    for text, bold, color in parts:
        run = p.add_run(text)
        run.bold = bold
        run.font.size = size
        if color is not None:
            run.font.color.rgb = color


def _insert_paragraph_after(paragraph, text="", style=None):
    new_p = paragraph._parent.add_paragraph(text, style=style)
    paragraph._p.addnext(new_p._p)
    return new_p


def _set_paragraph_text(p, text):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.add_run(text)


def _add_table_after(doc, anchor_para, rows, cols):
    table = doc.add_table(rows=rows, cols=cols)
    anchor_para._p.addnext(table._tbl)
    return table


def _format_zebra(table, header=True):
    for i, row in enumerate(table.rows):
        if header and i == 0:
            for cell in row.cells:
                _set_cell_shading(cell, HEADER_FILL)
                _set_runs(cell, bold=True, size=Pt(9), color=HEADER_TEXT)
        else:
            body_i = i - (1 if header else 0)
            if body_i % 2 == 1:
                for cell in row.cells:
                    _set_cell_shading(cell, ZEBRA_FILL)


def _fill_table(table, data, bold_first_col=False):
    for i, row_data in enumerate(data):
        for j, val in enumerate(row_data):
            cell = table.rows[i].cells[j]
            bold = (i == 0) or (bold_first_col and j == 0 and i > 0)
            _write_cell(cell, str(val), bold=bold)


def _fill_data_table(table, rows):
    """Data table: header + body rows. Body rows are (prop, prev, now_text, delta_tuple).
    delta_tuple = ("+12%", positive?) or None.
    Previous-lesson column is the 2nd column; "This lesson (Δ)" packs value + coloured delta.
    """
    header = rows[0]
    for j, h in enumerate(header):
        _write_cell(table.rows[0].cells[j], h, bold=True)
    for i, row in enumerate(rows[1:], start=1):
        prop, prev, now, delta = row
        _write_cell(table.rows[i].cells[0], prop, bold=True)
        _write_cell(table.rows[i].cells[1], prev, bold=False)
        # This-lesson cell: value + optional coloured delta.
        parts = [(now, False, None)]
        if delta is not None:
            tag, positive = delta
            colour = DELTA_POS_COLOR if positive else DELTA_NEG_COLOR
            parts.append((f"  {tag}", True, colour))
        _write_cell_colored_parts(table.rows[i].cells[2], parts)


def _insert_hook_callout(doc, anchor_para, arrow, body):
    table = _add_table_after(doc, anchor_para, rows=1, cols=1)
    cell = table.rows[0].cells[0]
    _set_cell_shading(cell, HOOK_FILL)
    _set_cell_border(
        cell,
        left=(24, HOOK_BORDER),
        top=(4, HOOK_BORDER),
        right=(4, HOOK_BORDER),
        bottom=(4, HOOK_BORDER),
    )
    cell.text = ""
    p = cell.paragraphs[0]
    arrow_run = p.add_run(arrow + "  ")
    arrow_run.bold = True
    arrow_run.font.size = Pt(11)
    arrow_run.font.color.rgb = HOOK_TEXT
    body_run = p.add_run(body)
    body_run.font.size = Pt(10)
    body_run.font.color.rgb = HOOK_TEXT
    return table


# ─── §5 overview table ──────────────────────────────────────────────────

LESSONS_OVERVIEW = [
    ["#", "Lesson", "Isolated variable", "Cohort / split", "Key finding"],
    ["L1", "Explicit vs implicit", "Training loss (MSE vs BPR)",
     "1M users, 5.2% overlap, LLO", "BPR wins 5.2× on Recall@10."],
    ["L2", "Mixed-population baseline", "Six models on realistic low-overlap cohort",
     "1M users, 5.2% overlap, LLO", "LightGCN leads; CDR underperforms."],
    ["L3", "Overlap filtering", "User overlap % (5.2% → 100%)",
     "19,880 overlap users, LLO", "CDR recovers: PTUPCDR +276%."],
    ["L4", "Source-rich / target-sparse", "Movie minimum (≥ 5 → ≥ 10)",
     "14,328 users, LLO", "PTUPCDR closes to −17% of LightGCN."],
    ["L5", "Catalog sharpening", "Movie popularity filter (≥ 10 ratings)",
     "14,328 users, 10,311 movies, LLO", "Balanced catalog; CDR holds gains."],
    ["L6", "Cold-start", "User-split 80/20, cold users have 0 games",
     "2,000 cold users, user-split", "CDR 4× LightGCN; Popularity tops."],
    ["L7", "Content-aware CDR", "Semantic (SBERT) vs collaborative",
     "14,328 users, LLO", "SBERT wins 11× on niche items."],
    ["L8", "Co-occurrence rerank", "Training-free cross-domain layer (λ = 0.05)",
     "L3 LLO + L6 cold-start", "Lifts every model; rescues cold-start."],
]

L5_INTRO = (
    "Each lesson isolates a single experimental variable to build understanding "
    "incrementally. We report Recall@10 (full-rank) as the primary metric; "
    "sampled HR@10 appears where relevant. Table 6 summarises the eight lessons; "
    "every per-lesson section below follows a uniform layout — opening question, "
    "data changes with deltas, illustration, benchmark graph, findings graph, "
    "and a connecting hook to the next lesson."
)


# ─── per-lesson content ─────────────────────────────────────────────────
# Each block has:
#   argument        — plain opening paragraph text (body after "Question.")
#   what_changed    — 4-row table [header + 3 rows]
#   data            — [header, (prop, prev, now, delta|None), ...]  (delta="±X%", positive)
#   findings        — [header, (finding, evidence), ...]
#   illustration    — filename in report_figures_v3/
#   illustration_caption
#   findings_fig    — filename for the findings comparison graph
#   findings_caption
#   hook            — connecting-hook text for the next lesson

LESSONS = {
    1: {
        "argument": (
            "Honestly, this is a basic distinction I should have internalised before "
            "starting — predicting a 1–5 rating is fundamentally different from "
            "ranking items a user picked over ones they skipped. I'm genuinely glad "
            "it surfaced on Lesson 1 rather than silently poisoning later ones. "
            "Getting the loss function wrong on implicit data wastes every downstream "
            "comparison, so the early rehearsal was well worth the humility."
        ),
        "what_changed": [
            ["Aspect", "Before (Phase 0)", "L1"],
            ["Models", "None — infrastructure only", "MF-Explicit (SVD) + MF-BPR"],
            ["Training loss", "N/A", "Pointwise MSE vs pairwise BPR"],
            ["Eval", "N/A", "LLO on games, full-rank Recall@10"],
        ],
        "data": [
            ["Property", "Previous lesson", "This lesson (Δ)"],
            ("Users", "—", "1,000,000", ("new", True)),
            ("Overlap", "—", "5.2%", ("new", True)),
            ("Movie · game interactions", "—", "1.80M · 337K", ("new", True)),
            ("Split", "—", "Leave-last-out on games", None),
        ],
        "findings": [
            ["Finding", "Evidence"],
            ["BPR crushes MF-Explicit", "Recall@10 0.0130 vs 0.0025 — 5.2× win; NDCG 6.7×"],
            ["Ranking loss fits implicit data", "All positives collapse to rating ≥ 4; SVD loses variance"],
            ["Low ceiling without CDR", "Best single-domain is only 1.3% Recall@10"],
        ],
        "illustration": "fig_lesson1_pointwise_pairwise.png",
        "illustration_caption": (
            "Figure L1: Pointwise vs pairwise. With positives collapsed to rating ≥ 4, "
            "BPR extracts 5.2× more signal than MF-Explicit."
        ),
        "findings_fig": "fig_findings_lesson1.png",
        "findings_caption": (
            "Figure L1b: MF-BPR vs MF-Explicit on Recall@10 and NDCG@10 — ranking loss "
            "dominates on both metrics."
        ),
        "hook": (
            "Lesson 2 expands the portfolio from two MF models to six (adding "
            "LightGCN, NCF, CMF, EMCDR, PTUPCDR) and runs them on the same "
            "naturally-sparse mixed population — asking whether CDR models can "
            "beat single-domain when only 5% of users overlap."
        ),
    },
    2: {
        "argument": (
            "Real platforms rarely see every user rate items in every domain. At a "
            "realistic 5% overlap, is there enough cross-domain signal for a CDR "
            "model to beat a single-domain graph recommender, or does the mapping "
            "function starve on too few bridge users?"
        ),
        "what_changed": [
            ["Aspect", "L1", "L2"],
            ["Models", "2 (MF-Explicit, MF-BPR)", "6 (+ LightGCN, NCF, CMF, EMCDR, PTUPCDR)"],
            ["Cohort", "1M users, 5.2% overlap", "Identical — only model family varies"],
            ["Question", "MSE vs BPR", "CDR vs single-domain at 5% overlap"],
        ],
        "data": [
            ["Property", "L1", "L2 (Δ)"],
            ("Users", "1,000,000", "1,000,000", ("same", True)),
            ("Overlap", "5.2%", "5.2%", ("same", True)),
            ("Portfolio", "2 models", "6 models", ("+4", True)),
            ("Split", "LLO", "LLO", None),
        ],
        "findings": [
            ["Finding", "Evidence"],
            ["LightGCN dominates", "Recall@10 0.0290 beats every CDR model"],
            ["Best CDR trails MF-BPR", "EMCDR 0.0160 < MF-BPR 0.0170; PTUPCDR only 0.0085"],
            ["Sampled HR@10 is misleading", "NCF / EMCDR score 0.80–0.89 sampled; collapse full-rank"],
        ],
        "illustration": "fig_lesson2_overlap_funnel.png",
        "illustration_caption": (
            "Figure L2: At 5.2% overlap the bridge population is tiny. LightGCN leads; "
            "dashed red arrows show how far each CDR model trails it."
        ),
        "findings_fig": "fig_findings_lesson2.png",
        "findings_caption": (
            "Figure L2b: Recall@10 on the mixed cohort — single-domain (blue) vs "
            "cross-domain (teal); LightGCN reference line marks the ceiling."
        ),
        "hook": (
            "Lesson 3 isolates user overlap as the variable: we restrict training "
            "and evaluation to the 19,880 users who have rated items in both "
            "domains, asking whether CDR finally recovers once every user is a "
            "bridge."
        ),
    },
    3: {
        "argument": (
            "If low overlap is what suffocated CDR in Lesson 2, forcing every user "
            "to be a bridge should let CDR breathe. How much of the gap closes when "
            "we drop everyone who doesn't cross domains, and does LightGCN remain "
            "unbeatable even then?"
        ),
        "what_changed": [
            ["Aspect", "L2", "L3"],
            ["Overlap", "5.2% natural", "100% forced"],
            ["User filter", "Random 1M sample", "k-core ≥ 10 + overlap"],
            ["Dataset path", "processed/movie_game", "processed_overlap/movie_game"],
        ],
        "data": [
            ["Property", "L2", "L3 (Δ)"],
            ("Users", "1,000,000", "19,880", ("−98%", False)),
            ("Overlap", "5.2%", "100%", ("+1823%", True)),
            ("Movie items", "45,933", "40,288", ("−12%", False)),
            ("Game items", "11,706", "10,034", ("−14%", False)),
        ],
        "findings": [
            ["Finding", "Evidence"],
            ["Every CDR model jumps", "PTUPCDR +276%; CMF +180%; EMCDR +53%"],
            ["LightGCN still leads", "0.0555 Recall@10 — gap narrows but holds"],
            ["MF-BPR collapses −71%", "0.0170 → 0.0050; pairwise sampling starves on 87K rows"],
            ["CDR competitive, not winning", "PTUPCDR closes from −71% to −42%"],
        ],
        "illustration": "fig_lesson3_overlap_lift.png",
        "illustration_caption": (
            "Figure L3: Moving from 5.2% → 100% overlap; green per-model arrows visualise "
            "the lift, from PTUPCDR's +276% down to LightGCN's +91%."
        ),
        "findings_fig": "fig_findings_lesson3.png",
        "findings_caption": (
            "Figure L3b: L2 vs L3 Recall@10 per model — the overlap filter helps CDR "
            "disproportionately more than single-domain."
        ),
        "hook": (
            "Lesson 4 stresses CDR toward its designed regime by tightening the "
            "movie minimum from ≥ 5 to ≥ 10 — keeping only source-rich users — and "
            "asks whether richer movie histories finally let the mapping-CDR "
            "family close the remaining gap."
        ),
    },
    4: {
        "argument": (
            "Overlap alone isn't the same as source richness — a user can cross "
            "domains while leaving almost no trace in either. If we keep only users "
            "with deep movie histories, does mapping-based CDR finally catch up to "
            "LightGCN?"
        ),
        "what_changed": [
            ["Aspect", "L3", "L4"],
            ["Movie minimum", "≥ 5 ratings per user", "≥ 10 ratings per user"],
            ["Cohort size", "19,880 users", "14,328 users (movie-rich)"],
            ["Dataset path", "processed_overlap", "processed_sparse_loose"],
        ],
        "data": [
            ["Property", "L3", "L4 (Δ)"],
            ("Users", "19,880", "14,328", ("−28%", False)),
            ("Movie interactions", "430,736", "388,919", ("−10%", False)),
            ("Game interactions", "87,613", "58,782", ("−33%", False)),
            ("Overlap", "100%", "100%", ("same", True)),
        ],
        "findings": [
            ["Finding", "Evidence"],
            ["PTUPCDR closes to −17%", "Gap to LightGCN: −42% (L3) → −17% (L4)"],
            ["EMCDR −34% (was −56%)", "Every mapping-CDR tightens toward LightGCN"],
            ["LightGCN drops 37%", "0.0555 → 0.0350 — loses its best users"],
            ["Monotonic trend L2 → L4", "Each tightening helps CDR more than single-domain"],
        ],
        "illustration": "fig_lesson4_gap_collapse.png",
        "illustration_caption": (
            "Figure L4: % behind LightGCN, by model. Green delta arrows show every "
            "CDR family's deficit shrinking between L3 and L4."
        ),
        "findings_fig": "fig_findings_lesson4.png",
        "findings_caption": (
            "Figure L4b: L3 vs L4 gap (lower = better). Source richness is the "
            "lever — CDR narrows, NCF drifts slightly wider."
        ),
        "hook": (
            "Lesson 5 attacks source-side noise: most L4 movie items have only a "
            "handful of in-cohort ratings. Applying a movie-popularity filter "
            "(≥ 10 ratings within-cohort) tests whether a cleaner source catalog "
            "further improves CDR transfer."
        ),
    },
    5: {
        "argument": (
            "A catalog is only as sharp as its tail allows. When the bottom 74% of "
            "rarely-rated movies are trimmed away, is the remaining signal cleaner "
            "for cross-domain transfer, or do we simply lose diversity and hurt "
            "every model?"
        ),
        "what_changed": [
            ["Aspect", "L4", "L5"],
            ["Movie popularity filter", "None", "≥ 10 in-cohort ratings"],
            ["Catalog ratio (movie : game)", "≈ 4 : 1 noisy", "≈ 1 : 1 balanced"],
            ["Game side", "9,147 items", "Unchanged"],
        ],
        "data": [
            ["Property", "L4", "L5 (Δ)"],
            ("Movie items", "39,534", "10,311", ("−74%", False)),
            ("Movie interactions", "388,919", "282,896", ("−27%", False)),
            ("Game items / interactions", "9,147 · 58,782", "9,147 · 58,782", ("same", True)),
            ("Users", "14,328", "14,328", ("same", True)),
        ],
        "findings": [
            ["Finding", "Evidence"],
            ["CDR holds most L4 gain", "CMF −3%, EMCDR −9%, PTUPCDR −12% vs L4"],
            ["NCF jumps +54%", "0.0140 → 0.0215 — cleaner item space helps fitting"],
            ["LightGCN dips −10%", "Movie-side graph loses neighbour diversity"],
            ["Cleaner > bigger", "Balanced catalogs outperform noisy large ones"],
        ],
        "illustration": "fig_lesson5_catalog_sharpening.png",
        "illustration_caption": (
            "Figure L5: Popularity-based trim — 39,534 → 10,311 movie items. Green "
            "−74% arrow marks the trim; the residual catalog stays informative."
        ),
        "findings_fig": "fig_findings_lesson5.png",
        "findings_caption": (
            "Figure L5b: L4 vs L5 Recall@10. CDR models hold their ground; NCF "
            "benefits; LightGCN takes a small hit."
        ),
        "hook": (
            "Lesson 6 introduces the regime CDR was designed for: a user-split "
            "protocol where 20% of users have zero game history. We ask whether "
            "cross-domain transfer can rank games for users the collaborative "
            "models have never seen."
        ),
    },
    6: {
        "argument": (
            "The true test of CDR is the user the single-domain model has never "
            "seen. With 2,000 users stripped of every game interaction, can "
            "cross-domain transfer still rank their first game — and can anything "
            "beat a simple popularity prior?"
        ),
        "what_changed": [
            ["Aspect", "L5", "L6"],
            ["Dataset", "processed_sparse_loose", "processed_overlap (richer source)"],
            ["Split", "LLO on games", "User-split 80/20 (cold users have 0 games)"],
            ["New baseline", "—", "Popularity (warm-user counts)"],
        ],
        "data": [
            ["Property", "L5", "L6 (Δ)"],
            ("Users evaluated", "14,328", "2,000 cold", ("−86%", False)),
            ("Cold game history", "—", "0 by construction", ("new", True)),
            ("Protocol", "LLO", "User-split cold-start", None),
            ("Warm training users", "—", "10,124", ("new", True)),
        ],
        "findings": [
            ["Finding", "Evidence"],
            ["Popularity wins cold-start", "0.0365 Recall@10 — first-game is often a hit"],
            ["CDR 4× over LightGCN", "PTUPCDR 0.0305 vs LightGCN 0.0080"],
            ["Single-domain collapses", "MF-BPR 0.000, NCF 0.002 — no game signal at all"],
            ["Routing rule born", "0 games → EMCDR/PTUPCDR + Popularity"],
        ],
        "illustration": "fig_lesson6_cold_start.png",
        "illustration_caption": (
            "Figure L6: User-split cold-start. Dashed red arrows mark the collapse "
            "of single-domain models; CDR and Popularity carry the 2,000 cold users."
        ),
        "findings_fig": "fig_findings_lesson6.png",
        "findings_caption": (
            "Figure L6b: Cold-start Recall@10 across six models — Popularity (amber) "
            "tops; CDR (teal) 4× over single-domain (blue)."
        ),
        "hook": (
            "Lesson 7 turns to a complementary cold-start regime — users who do "
            "have some game history but whose taste is niche. We bring in a "
            "content-based bridge (SBERT) and ask whether semantic similarity can "
            "rescue long-tail items every collaborative model is blind to."
        ),
    },
    7: {
        "argument": (
            "Collaborative filters live and die by interaction counts, so a niche "
            "fantasy-horror fan looks identical to a user with no preferences at "
            "all. Can a semantic content bridge (SBERT) rescue the long tail that "
            "every collaborative model misses?"
        ),
        "what_changed": [
            ["Aspect", "L6", "L7"],
            ["Split", "User-split cold-start", "LLO on games (L4 cohort)"],
            ["New models", "Popularity baseline", "SBERT + SBERT-CDR (MiniLM 384-dim)"],
            ["Analysis", "Overall cold-user Recall@10", "Subgroups — niche vs popular targets"],
        ],
        "data": [
            ["Property", "L6", "L7 (Δ)"],
            ("Users", "2,000 cold", "14,328", ("+617%", True)),
            ("Protocol", "User-split cold", "LLO on games", None),
            ("Encoder", "—", "all-MiniLM-L6-v2 (384-d)", ("new", True)),
            ("Models", "+1 Popularity", "+2 SBERT, SBERT-CDR", ("+2", True)),
        ],
        "findings": [
            ["Finding", "Evidence"],
            ["SBERT-CDR matches LightGCN", "0.0330 vs 0.0335 — within 1.5%, zero training"],
            ["SBERT 11× on niche", "one_shot_unpopular (n=82): SBERT 0.1341 vs LightGCN 0.0122"],
            ["Semantics bridges long tail", "Genre consistency invisible to CF, visible in SBERT"],
            ["Routing rule: long-tail → SBERT-CDR", "Content is the only reliable family there"],
        ],
        "illustration": "fig_lesson7_niche_win.png",
        "illustration_caption": (
            "Figure L7: Overall parity panel vs niche subgroup panel. Green ratio "
            "arrow from LightGCN to SBERT marks the 11× win on niche items."
        ),
        "findings_fig": "fig_findings_lesson7.png",
        "findings_caption": (
            "Figure L7b: Overall vs niche Recall@10. SBERT's niche advantage is an "
            "order-of-magnitude effect collaborative filtering cannot produce."
        ),
        "hook": (
            "Lesson 8 asks whether a single training-free layer can improve every "
            "model in every regime. A cross-domain co-occurrence rerank "
            "(λ = 0.05) is evaluated on both the L3 LLO cohort and the L6 "
            "cold-start split."
        ),
    },
    8: {
        "argument": (
            "Every model so far is trained end-to-end. Could a simple, training-free "
            "rerank — built only from how often a movie and a game are liked by the "
            "same users — lift all of them at once, without retraining or extra "
            "data?"
        ),
        "what_changed": [
            ["Aspect", "L7", "L8"],
            ["New model", "SBERT, SBERT-CDR", "None — post-processing layer only"],
            ["Mechanism", "End-to-end trained score", "final = base + λ · Σ cooc[m,g]"],
            ["Cost", "Full training run per model", "Zero — cooc matrix computed once"],
        ],
        "data": [
            ["Property", "L7", "L8 (Δ)"],
            ("Evaluation cohorts", "L4 LLO", "L3 LLO + L6 cold-start", ("+1", True)),
            ("Training cost", "Full runs", "Zero", ("removed", True)),
            ("Blend weight λ", "—", "0.05 (tuned 0.01 → 0.20)", ("new", True)),
            ("Matrix formula", "—", "log(1 + co-liked users)", ("new", True)),
        ],
        "findings": [
            ["Finding", "Evidence"],
            ["Lifts every model on LLO", "EMCDR +31% Recall@10 at λ = 0.05"],
            ["Rescues cold-start", "LightGCN 0.010 → 0.033 (+230%); MF-BPR 0.001 → 0.020"],
            ["Mapping CDR: marginal gain", "Already near popularity ceiling — but cost is zero"],
            ["λ tuning is sharp", "Monotonic up to ~0.05–0.10; collapses past 0.20"],
        ],
        "illustration": "fig_lesson8_cooc_layer.png",
        "illustration_caption": (
            "Figure L8: Co-occurrence rerank pipeline + per-model lift panel. Each "
            "before → after pair carries an explicit green delta arrow."
        ),
        "findings_fig": "fig_findings_lesson8.png",
        "findings_caption": (
            "Figure L8b: Base vs rerank Recall@10 across four deployments — every "
            "model either lifts or holds; single-domain cold-start recovers."
        ),
        "hook": (
            "§5.9 turns from comparing models to tuning them: the hyperparameter "
            "sweeps that produced the defaults used throughout §5.1–§5.8 are "
            "reported there, documenting each model's sensitivity to its key knobs."
        ),
    },
}


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

NEXT_HEADING_PREFIX = {
    1: "5.2 Lesson 2",
    2: "5.3 Lesson 3",
    3: "5.4 Lesson 4",
    4: "5.5 Lesson 5",
    5: "5.6 Lesson 6",
    6: "5.7 Lesson 7",
    7: "5.8 Lesson 8",
    8: "5.9 Hyperparameter Analysis",
}


# ─── Figure 18 swap ─────────────────────────────────────────────────────

def fix_fig18(doc):
    shutil.copyfile(FRESH_FIG18, TARGET_FIG18)
    ps = doc.paragraphs
    img_para = ps[340]
    caption_para = ps[341]
    assert "Figure 18" in caption_para.text, caption_para.text
    for r in list(img_para.runs):
        r._element.getparent().remove(r._element)
    run = img_para.add_run()
    run.add_picture(str(TARGET_FIG18), width=Inches(6.0))
    img_para.alignment = 1
    cleaned = caption_para.text.replace("(fix)", "").strip()
    _set_paragraph_text(caption_para, cleaned)


def _find_heading(ps, heading_text, level="Heading 2"):
    for i, p in enumerate(ps):
        if p.text.strip() == heading_text and level in p.style.name:
            return i
    raise LookupError(f"heading not found: {heading_text}")


def _find_next_heading_2(ps, start_idx, prefix):
    for j in range(start_idx + 1, len(ps)):
        if ps[j].text.strip().startswith(prefix) and "Heading 2" in ps[j].style.name:
            return j
    raise LookupError(f"next heading not found after idx {start_idx}: {prefix}")


# ─── lesson expansion ───────────────────────────────────────────────────

def expand_lesson(doc, lesson_num):
    ps = doc.paragraphs
    heading_idx = _find_heading(ps, LESSONS_HEADINGS[lesson_num])
    block = LESSONS[lesson_num]
    heading = ps[heading_idx]

    # After heading, top-to-bottom:
    #   1. Opening "Question." paragraph (plain, no callout).
    #   2. Bold "What changed vs previous lesson" label.
    #   3. What-changed 3-col table.
    #   4. Bold "Data characteristics" label.
    #   5. Data 3-col table (prop / prev / this-lesson with Δ).
    #   6. Illustration image + italic caption (centered).
    #
    # Insert in reverse so addnext chains push prior inserts down.

    # 6b. Caption
    cap_p = _insert_paragraph_after(heading, block["illustration_caption"])
    cap_p.alignment = 1
    for r in cap_p.runs:
        r.italic = True
        r.font.size = Pt(9)

    # 6a. Illustration
    img_p = _insert_paragraph_after(heading)
    img_p.alignment = 1
    img_p.add_run().add_picture(str(ILLUS_DIR / block["illustration"]), width=Inches(5.8))

    # 5. Data table (3-col)
    data_label_anchor = _insert_paragraph_after(heading, "")
    data_tbl = _add_table_after(doc, data_label_anchor, rows=len(block["data"]), cols=3)
    _fill_data_table(data_tbl, block["data"])
    _format_zebra(data_tbl)

    # 4. Data label
    _set_paragraph_text(data_label_anchor, "Data characteristics")
    for r in data_label_anchor.runs:
        r.bold = True
        r.font.size = Pt(10.5)

    # 3. What-changed table
    wc_label_anchor = _insert_paragraph_after(heading, "")
    wc_tbl = _add_table_after(doc, wc_label_anchor, rows=len(block["what_changed"]), cols=3)
    _fill_table(wc_tbl, block["what_changed"], bold_first_col=True)
    _format_zebra(wc_tbl)

    # 2. What-changed label
    _set_paragraph_text(wc_label_anchor, "What changed vs previous lesson")
    for r in wc_label_anchor.runs:
        r.bold = True
        r.font.size = Pt(10.5)

    # 1. Opening "Question." paragraph (plain, italic body).
    q_p = _insert_paragraph_after(heading, "")
    q_label = q_p.add_run("Question.  ")
    q_label.bold = True
    q_label.font.size = Pt(10.5)
    q_body = q_p.add_run(block["argument"])
    q_body.italic = True
    q_body.font.size = Pt(10.5)

    # ── End of section (before next heading 2): findings label + table +
    # findings graph + spacer + hook.
    ps2 = doc.paragraphs
    next_idx2 = _find_next_heading_2(
        ps2, _find_heading(ps2, LESSONS_HEADINGS[lesson_num]),
        NEXT_HEADING_PREFIX[lesson_num],
    )
    end_anchor = ps2[next_idx2 - 1]

    # Spacer + hook (deepest first so hook is at very bottom).
    hook_anchor = _insert_paragraph_after(end_anchor, "")
    _insert_hook_callout(doc, hook_anchor, "→ Next up.", block["hook"])
    hook_anchor._p.getparent().remove(hook_anchor._p)
    # Explicit spacer paragraph before the hook.
    spacer = _insert_paragraph_after(end_anchor, "")
    spacer.paragraph_format.space_before = Pt(18)
    spacer.paragraph_format.space_after = Pt(6)

    # Findings graph caption, then image, then table + label.
    fg_cap = _insert_paragraph_after(end_anchor, block["findings_caption"])
    fg_cap.alignment = 1
    for r in fg_cap.runs:
        r.italic = True
        r.font.size = Pt(9)
    fg_img = _insert_paragraph_after(end_anchor)
    fg_img.alignment = 1
    fg_img.add_run().add_picture(str(ILLUS_DIR / block["findings_fig"]), width=Inches(5.6))

    findings_label_anchor = _insert_paragraph_after(end_anchor, "")
    findings_tbl = _add_table_after(
        doc, findings_label_anchor,
        rows=len(block["findings"]), cols=2,
    )
    _fill_table(findings_tbl, block["findings"], bold_first_col=True)
    _format_zebra(findings_tbl)
    _set_paragraph_text(findings_label_anchor, "Key findings")
    for r in findings_label_anchor.runs:
        r.bold = True
        r.font.size = Pt(10.5)


# ─── §5 overview table ──────────────────────────────────────────────────

def insert_lessons_overview(doc):
    ps = doc.paragraphs
    for i, p in enumerate(ps):
        if p.text.strip().startswith("5. Experiments and Results") and "Heading 1" in p.style.name:
            h5_idx = i
            break
    else:
        raise LookupError("§5 heading not found")
    intro = ps[h5_idx + 1]
    _set_paragraph_text(intro, L5_INTRO)

    cap = _insert_paragraph_after(
        intro,
        "Table 6: Eight-lesson summary — each lesson isolates one variable from the previous.",
    )
    for r in cap.runs:
        r.italic = True
        r.font.size = Pt(9)

    tbl = _add_table_after(doc, cap, rows=len(LESSONS_OVERVIEW), cols=len(LESSONS_OVERVIEW[0]))
    _fill_table(tbl, LESSONS_OVERVIEW, bold_first_col=True)
    _format_zebra(tbl)


# ─── §5.10 Lessons Recap ────────────────────────────────────────────────

RECAP_PARAS = [
    (
        "Taken together, the eight lessons form a deliberate staircase: each step "
        "removes one variable the previous step could not explain, and the "
        "result is a much clearer picture of when cross-domain recommendation "
        "actually helps. The rest of this section re-reads the arc in plain "
        "language, so the findings are easier to carry into the design chapter."
    ),
    (
        "Lessons 1–2 set the floor. L1 demonstrated that on implicit feedback "
        "the training objective dominates everything else — pairwise BPR beats "
        "pointwise MSE by 5.2× on Recall@10 with identical data. L2 expanded "
        "the portfolio to six models on a realistic 5%-overlap cohort and "
        "showed that, at this natural low-overlap ratio, LightGCN wins clearly "
        "and CDR models underperform even MF-BPR. CDR is not magic; without "
        "enough bridge users the mapping function starves."
    ),
    (
        "Lessons 3–5 peeled back the conditions CDR actually needs. L3 forced "
        "100% overlap and CDR recovered dramatically — PTUPCDR +276%, CMF "
        "+180%, EMCDR +53% — confirming the overlap hypothesis directly. L4 "
        "tightened to source-rich users (movies ≥ 10) and PTUPCDR closed to "
        "just 17% behind LightGCN. L5 sharpened the source catalog by trimming "
        "74% of long-tail movies and CDR held almost all its L4 gain with a "
        "4× smaller item set, demonstrating that cleaner beats bigger. The "
        "monotonic L2→L3→L4 trend is the central positive result: every "
        "tightening of the cohort helps mapping-CDR disproportionately more "
        "than single-domain."
    ),
    (
        "Lessons 6–7 took the work into the regimes CDR was designed for. L6 "
        "evaluated on 2,000 cold users with zero games: single-domain MF-BPR "
        "and NCF collapsed to near-zero, LightGCN fell to 0.008, and "
        "mapping-CDR (EMCDR, PTUPCDR) landed at 0.030 — roughly 4× over "
        "LightGCN. A simple Popularity prior topped the table at 0.0365, "
        "which is the honest cold-start result: the first game is often a "
        "hit title. L7 then covered the complementary cold-start regime — "
        "users with some history but niche taste. SBERT-CDR matched LightGCN "
        "overall without any training; on the niche subgroup (n=82) SBERT "
        "beat LightGCN 11× (0.134 vs 0.012). Content is the only family that "
        "reliably ranks long-tail items."
    ),
    (
        "Lesson 8 is the pragmatic close: a training-free co-occurrence rerank "
        "layer that lifts every model in every regime. It rescues single-"
        "domain LightGCN from 0.010 to 0.033 at cold-start (+230%), lifts "
        "EMCDR by 31% on LLO, and leaves the already-strong CDR cold-start "
        "scores essentially unchanged. The cost is zero; the implementation "
        "is a single matrix and a blend weight (λ = 0.05)."
    ),
    (
        "Three synthesis points worth carrying forward. First, there is no "
        "single winning model family — the right answer depends on the user's "
        "history depth and the target item's popularity, which is exactly "
        "why §6 articulates a routing rule rather than picking a champion. "
        "Second, CDR and content are complementary, not competitive: CDR "
        "transfers dense history into a new domain, content bridges the long "
        "tail where collaborative signal is absent. Third, co-occurrence "
        "rerank is a universal free lift — whatever the base model, adding "
        "the rerank layer either helps or holds, so it becomes the default "
        "post-processor for every row of the production recommender."
    ),
]


def insert_recap(doc):
    """Insert §5.10 Lessons Recap heading + paragraphs right before §6."""
    ps = doc.paragraphs
    for i, p in enumerate(ps):
        if p.text.strip().startswith("6. Discussion") and "Heading 1" in p.style.name:
            six_idx = i
            break
    else:
        raise LookupError("§6 heading not found")
    anchor = ps[six_idx - 1]  # last para of §5

    # Insert bottom-up so they land in order.
    # Paragraphs (reverse), then heading at top.
    for text in reversed(RECAP_PARAS):
        p = _insert_paragraph_after(anchor, text)
    head_p = _insert_paragraph_after(anchor, "5.10 Lessons Recap", style="Heading 2")


# ─── §6.3 Recommendation Rows Design ────────────────────────────────────

ROW_ROWS = [
    ["Row", "Label", "Model / algorithm", "Routed by", "Lesson rationale"],
    ["1", "Primary headline pick",
     "EMCDR + cooc (cold) · PTUPCDR + cooc (one-shot) · LightGCN + cooc (warm)",
     "Game-history depth",
     "L6 routing rule: cold users → mapping-CDR; L3 → LightGCN wins warm; L8 lift everywhere"],
    ["2", "Secondary collaborative",
     "LightGCN + cooc (one-shot) · EMCDR + cooc (warm) · (skipped when cold)",
     "Complement to Row 1",
     "Visible contrast between two strong families; avoids CDR/CDR collision at cold-start"],
    ["3", "Cross-domain explainable",
     "Co-occurrence standalone",
     "Constant row",
     "L8: training-free, explainable ('because you liked X'), instant refresh"],
    ["4", "Hidden Gems (games)",
     "SBERT-CDR on bottom-50% popularity games + cooc",
     "Constant row, niche-filtered",
     "L7: SBERT wins 11× on niche; filter enforces the validated operating range"],
    ["5", "Similar movies (content)",
     "SBERT on movie profile",
     "Constant row",
     "In-domain content similarity — same semantic bridge as Row 4, opposite direction"],
    ["6", "Similar movies (collab)",
     "LightGCN movies + reverse cooc",
     "Constant row",
     "L2+L8: LightGCN is the single-domain champion; reverse cooc adds game-to-movie bridge"],
    ["7", "Cross-domain reverse",
     "Reverse co-occurrence standalone",
     "Constant row",
     "Mirror of Row 3 for games→movies; explainable and training-free"],
    ["8", "Trending games",
     "Popularity (global counts)",
     "Constant row",
     "L6: Popularity tops true cold-start; safe anchor row at the bottom of the page"],
    ["9", "Trending movies",
     "Popularity (global counts)",
     "Constant row",
     "Symmetric anchor for the movie side"],
]


ROW_DESIGN_INTRO = (
    "The front-end surfaces nine rows. The design deliberately matches each row "
    "to one lesson (or a small combination of lessons) so the final product "
    "mirrors the staircase of findings rather than hiding them behind a single "
    "generic model. Two rules shape the whole layout: (i) co-occurrence rerank "
    "(Lesson 8) is a universal post-processor, so it is applied to every "
    "ranking row except the pure popularity rows — where Lesson 8 showed "
    "negligible gain; (ii) Row 1 is routed by the user's game-history depth "
    "(Lesson 6 routing rule) because single-domain graph models collapse to "
    "popularity at true cold-start, so mapping-CDR takes the front page when "
    "game signal is missing."
)

ROW_DESIGN_OUTRO = (
    "A few design notes worth recording. Row 4's filter to bottom-50% popularity "
    "is deliberate: Lesson 7 validated SBERT-CDR specifically on the niche "
    "subgroup, so using it as a full-catalog ranker would overstate its "
    "reach. The SBERT profile uses movies only — a pure SBERT-CDR signal — "
    "to keep the cross-domain bridge honest; mixing in game embeddings "
    "would drift it back toward in-domain similarity and dilute the "
    "cross-domain effect that Lesson 7 measured. Finally, rows 6 and 7 are "
    "reverse-direction mirrors of rows 3 and 1 respectively, so the UI is "
    "symmetric: every game row has a movie counterpart that reuses the same "
    "lesson-backed machinery."
)


def insert_row_design(doc):
    """Insert §6.3 Recommendation Rows Design right before §7."""
    ps = doc.paragraphs
    for i, p in enumerate(ps):
        if p.text.strip().startswith("7. System Design") and "Heading 1" in p.style.name:
            seven_idx = i
            break
    else:
        raise LookupError("§7 heading not found")
    anchor = ps[seven_idx - 1]

    # Bottom-up: outro paragraph → caption+table → intro → heading.
    outro_p = _insert_paragraph_after(anchor, ROW_DESIGN_OUTRO)
    cap_p = _insert_paragraph_after(anchor,
                                    "Table 7: Nine-row recommendation layout — "
                                    "each row traces back to a specific lesson.")
    for r in cap_p.runs:
        r.italic = True
        r.font.size = Pt(9)
    tbl = _add_table_after(doc, cap_p, rows=len(ROW_ROWS), cols=len(ROW_ROWS[0]))
    _fill_table(tbl, ROW_ROWS, bold_first_col=True)
    _format_zebra(tbl)
    _insert_paragraph_after(anchor, ROW_DESIGN_INTRO)
    _insert_paragraph_after(anchor, "6.4 Recommendation Rows Design", style="Heading 2")


# ─── main ───────────────────────────────────────────────────────────────

def main():
    if not BACKUP.exists():
        raise SystemExit(f"backup not found: {BACKUP}")
    shutil.copyfile(BACKUP, DOC)
    print(f"Restored {DOC} from backup.")

    doc = Document(DOC)

    fix_fig18(doc)
    print("Figure 18 swap applied.")

    # Expand each lesson bottom-up so earlier indices stay stable.
    for n in sorted(LESSONS.keys(), reverse=True):
        expand_lesson(doc, n)
        print(f"Lesson {n} expanded.")

    insert_lessons_overview(doc)
    print("Lessons-overview table inserted.")

    insert_recap(doc)
    print("§5.10 Lessons Recap inserted.")

    insert_row_design(doc)
    print("§6.3 Recommendation Rows Design inserted.")

    doc.save(DOC)
    print(f"Saved {DOC}")


if __name__ == "__main__":
    main()
