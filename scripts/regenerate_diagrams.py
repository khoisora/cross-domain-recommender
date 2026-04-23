"""Regenerate diagram-style figures for the report using Graphviz.

Produces PNGs in report_figures_v3/:
  fig04_data_pipeline.png
  fig31_routing.png
  fig32_architecture.png
  fig33_recommendation_flow.png
  fig34_frontend.png
"""

from pathlib import Path
from graphviz import Digraph

OUT = Path(__file__).resolve().parent.parent / "report_figures_v3"
OUT.mkdir(exist_ok=True)

# palette
NAVY   = "#1f3a5f"
BLUE   = "#4472C4"
TEAL   = "#2EC4B6"
ORANGE = "#ED7D31"
GREEN  = "#70AD47"
PURPLE = "#7B68EE"
GREY   = "#6B7280"
LBLUE  = "#D6E4F0"
LORANGE= "#FCE4D6"
LGREEN = "#E2EFDA"
LPURPLE= "#E8E0F0"
LYELLOW= "#FFF4CE"
LTEAL  = "#D4F1EC"


def base(name, rankdir="LR", nodesep="0.45", ranksep="0.70"):
    g = Digraph(name, format="png")
    g.attr(rankdir=rankdir, nodesep=nodesep, ranksep=ranksep,
           bgcolor="white", fontname="Helvetica", fontsize="11",
           dpi="180", pad="0.25")
    g.attr("node", shape="box", style="rounded,filled",
           fontname="Helvetica", fontsize="11",
           color=NAVY, fillcolor="white", penwidth="1.4")
    g.attr("edge", color=GREY, arrowsize="0.7", penwidth="1.2",
           fontname="Helvetica", fontsize="9")
    return g


# ─────────────────────────────────────────────────────────────
# Figure 4 — Data Processing Pipeline
# ─────────────────────────────────────────────────────────────
def fig04():
    g = base("fig04", rankdir="TB", nodesep="0.3", ranksep="0.55")

    with g.subgraph(name="cluster_raw") as s:
        s.attr(label="Raw (gzip JSONL)", style="rounded,dashed",
               color=GREY, fontname="Helvetica-Bold", fontsize="11")
        s.node("raw_m", "Movies & TV\n~4.4M reviews", fillcolor=LBLUE, color=BLUE)
        s.node("raw_g", "Video Games\n~0.47M reviews", fillcolor=LORANGE, color=ORANGE)

    g.node("filter", "Filter & Normalise\nrating ≥ 4 → positive · k-core ≥ 10 · dedup item variants",
           fillcolor=LYELLOW, color="#C9A227")

    g.node("parquet", "Parquet  (movie_game cohort)",
           fillcolor=LGREEN, color=GREEN, shape="folder")

    with g.subgraph(name="cluster_split") as s:
        s.attr(label="Splits per lesson", style="rounded,dashed",
               color=GREY, fontname="Helvetica-Bold", fontsize="11")
        s.node("llo", "Leave-Last-Out\n(target = games)", fillcolor=LTEAL, color=TEAL)
        s.node("cs",  "User-Split Cold-Start\n(80/20 warm/cold)", fillcolor=LTEAL, color=TEAL)
        s.node("ol",  "Overlap filtering\n(100% / source-rich)", fillcolor=LTEAL, color=TEAL)

    g.node("train", "Models — MF-BPR · NCF · LightGCN · CMF · EMCDR · PTUPCDR · SBERT-CDR",
           fillcolor=LPURPLE, color=PURPLE)

    g.edge("raw_m", "filter")
    g.edge("raw_g", "filter")
    g.edge("filter", "parquet")
    g.edge("parquet", "llo")
    g.edge("parquet", "cs")
    g.edge("parquet", "ol")
    g.edge("llo", "train")
    g.edge("cs",  "train")
    g.edge("ol",  "train")

    g.render(OUT / "fig04_data_pipeline", cleanup=True)


# ─────────────────────────────────────────────────────────────
# Figure 31 — Model Routing Decision Rule
# ─────────────────────────────────────────────────────────────
def fig31():
    g = base("fig31", rankdir="TB", nodesep="0.5", ranksep="0.55")

    g.node("root",
           "Target-domain (game) ratings\nfor this user?",
           fillcolor=LYELLOW, color="#C9A227", shape="diamond")

    g.node("cold",   "0 games\n· cold-start",    fillcolor=LBLUE,   color=BLUE)
    g.node("one",    "1–2 games\n· one-shot",    fillcolor=LORANGE, color=ORANGE)
    g.node("warm",   "≥ 3 games\n· warm",        fillcolor=LGREEN,  color=GREEN)

    g.edge("root", "cold", label="zero")
    g.edge("root", "one",  label="few")
    g.edge("root", "warm", label="many")

    g.node("emcdr", "EMCDR\n(global MLP mapping)\n+ universal cooc",
           fillcolor=LBLUE,   color=BLUE)
    g.node("ptup",  "PTUPCDR\n(few-shot transfer)\n+ universal cooc",
           fillcolor=LORANGE, color=ORANGE)
    g.node("lgcn",  "LightGCN\n(in-domain graph)\n+ universal cooc",
           fillcolor=LGREEN,  color=GREEN)

    g.edge("cold", "emcdr")
    g.edge("one",  "ptup")
    g.edge("warm", "lgcn")

    g.node("niche",
           "Hidden Gems row\nSBERT-CDR on bottom-50% popularity",
           fillcolor=LPURPLE, color=PURPLE)
    g.edge("emcdr", "niche", style="dashed", label="orthogonal lane")
    g.edge("ptup",  "niche", style="dashed")
    g.edge("lgcn",  "niche", style="dashed")

    g.render(OUT / "fig31_routing", cleanup=True)


# ─────────────────────────────────────────────────────────────
# Figure 32 — High-Level Architecture
# ─────────────────────────────────────────────────────────────
def fig32():
    g = base("fig32", rankdir="TB", nodesep="0.4", ranksep="0.55")

    with g.subgraph(name="cluster_fe") as s:
        s.attr(label="Frontend (jQuery single-page app)", style="rounded,dashed",
               color=BLUE, fontname="Helvetica-Bold", fontsize="11")
        s.node("ui",
               "index.html\nhash router · user picker · rating UI · row renderer",
               fillcolor=LBLUE, color=BLUE)

    with g.subgraph(name="cluster_be") as s:
        s.attr(label="Backend (FastAPI)", style="rounded,dashed",
               color=GREEN, fontname="Helvetica-Bold", fontsize="11")
        s.node("api", "main.py\nREST endpoints (/api/recommendations, /api/ratings, …)",
               fillcolor=LGREEN, color=GREEN)
        s.node("rec", "HybridRecommender\nsegment routing · row assembly · cooc post-process",
               fillcolor=LGREEN, color=GREEN)
        s.node("store", "EmbeddingStore\n7 model matrices + cooc maps (in memory)",
               fillcolor=LYELLOW, color="#C9A227")
        s.node("db",   "SQLite  —  users · ratings",
               fillcolor=LYELLOW, color="#C9A227", shape="cylinder")

    with g.subgraph(name="cluster_ml") as s:
        s.attr(label="Offline ML pipeline", style="rounded,dashed",
               color=PURPLE, fontname="Helvetica-Bold", fontsize="11")
        s.node("train", "Train & export  (bench + export_demo_artifacts.py)",
               fillcolor=LPURPLE, color=PURPLE)
        s.node("art",   "artifacts/demo/  *.npy · *.json",
               fillcolor=LPURPLE, color=PURPLE, shape="folder")

    g.edge("ui", "api", label="HTTP")
    g.edge("api", "rec")
    g.edge("rec", "store")
    g.edge("rec", "db")
    g.edge("train", "art")
    g.edge("art", "store", label="load on startup", style="dashed")

    g.render(OUT / "fig32_architecture", cleanup=True)


# ─────────────────────────────────────────────────────────────
# Figure 33 — Recommendation Flow (9 rows)
# ─────────────────────────────────────────────────────────────
def fig33():
    # Use an HTML table label to get a compact, readable 9-row layout.
    g = base("fig33", rankdir="TB", nodesep="0.35", ranksep="0.45")

    g.node("req", "GET /api/recommendations/{user_id}",
           fillcolor=LBLUE, color=BLUE, shape="parallelogram")
    g.node("seg",
           "Segment classifier\ngame_count → cold-start / one-shot / warm",
           fillcolor=LYELLOW, color="#C9A227", shape="diamond")
    g.edge("req", "seg")

    def row(label, model, color):
        return (f'<TR><TD ALIGN="LEFT" BGCOLOR="{color}">{label}</TD>'
                f'<TD ALIGN="LEFT">{model}</TD></TR>')

    game_rows = "".join([
        row("① Top Picks (routed)",     "LightGCN / PTUPCDR / EMCDR + cooc", LGREEN),
        row("② Complementary",          "opposite paradigm per segment",     LGREEN),
        row("③ Because you liked …",    "cooc standalone",                   LTEAL),
        row("④ Hidden Gems",            "SBERT-CDR on bottom-50% + cooc",    LPURPLE),
        row("⑤ Popular games",          "popularity fallback",               LYELLOW),
    ])
    movie_rows = "".join([
        row("⑥ Similar movies",         "SBERT in-domain",                   LPURPLE),
        row("⑦ LightGCN-Movies",        "collaborative in-domain",           LGREEN),
        row("⑧ Movies from your games", "reverse cooc",                      LTEAL),
        row("⑨ Popular movies",         "popularity fallback",               LYELLOW),
    ])

    game_label = (
        '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">'
        f'<TR><TD COLSPAN="2" BGCOLOR="{LBLUE}"><B>Game rows (5)</B></TD></TR>'
        f"{game_rows}</TABLE>>"
    )
    movie_label = (
        '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">'
        f'<TR><TD COLSPAN="2" BGCOLOR="{LORANGE}"><B>Movie rows (4)</B></TD></TR>'
        f"{movie_rows}</TABLE>>"
    )

    g.node("games",  label=game_label,  shape="plaintext")
    g.node("movies", label=movie_label, shape="plaintext")

    g.edge("seg", "games")
    g.edge("seg", "movies")

    g.node("post",
           "Per-row post-process:\nmask seen · λ=0.05 cooc · top-k",
           fillcolor=LORANGE, color=ORANGE)
    g.edge("games",  "post", style="dashed", color=GREY)
    g.edge("movies", "post", style="dashed", color=GREY)

    g.node("resp", "JSON response\n{segment, rows: [...]}",
           fillcolor=LBLUE, color=BLUE, shape="parallelogram")
    g.edge("post", "resp")

    g.render(OUT / "fig33_recommendation_flow", cleanup=True)


# ─────────────────────────────────────────────────────────────
# Figure 34 — Frontend Architecture
# ─────────────────────────────────────────────────────────────
def fig34():
    g = base("fig34", rankdir="TB", nodesep="0.35", ranksep="0.55")

    g.node("html", "index.html  (single file)",
           fillcolor=LBLUE, color=BLUE)

    with g.subgraph(name="cluster_pages") as s:
        s.attr(label="Hash-routed views", style="rounded,dashed",
               color=GREY, fontname="Helvetica-Bold", fontsize="11")
        s.node("home",  "#home\nuser picker\n(cold / one-shot / warm)", fillcolor=LGREEN, color=GREEN)
        s.node("recs",  "#user/{id}\nsegment banner\n9 recommendation rows", fillcolor=LGREEN, color=GREEN)
        s.node("item",  "#item/{id}\nrating widget\nitem metadata", fillcolor=LGREEN, color=GREEN)

    g.node("api", "fetch /api/*",
           fillcolor=LYELLOW, color="#C9A227", shape="parallelogram")
    g.node("row", "Row renderer\ndrag-to-scroll · chips · badges",
           fillcolor=LPURPLE, color=PURPLE)

    g.edge("html", "home")
    g.edge("html", "recs")
    g.edge("html", "item")
    g.edge("home", "api")
    g.edge("recs", "api")
    g.edge("item", "api")
    g.edge("recs", "row")

    g.render(OUT / "fig34_frontend", cleanup=True)


if __name__ == "__main__":
    fig04(); fig31(); fig32(); fig33(); fig34()
    print("Wrote figures to", OUT)
