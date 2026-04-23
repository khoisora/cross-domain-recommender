"""Modern Graphviz diagrams for the project report.

Generates 24 diagrams at 200 DPI into ./new_figures/.
Call as: python scripts/gen_graphviz.py
"""

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FONT, BG, TITLE, MUTED, EDGE, BORDER, PALETTE, node_attrs, title_label

OUT = Path(__file__).resolve().parent.parent / "new_figures"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def render(name: str, dot: str, dpi: int = 200) -> None:
    dot_path = OUT / f"{name}.dot"
    png_path = OUT / f"{name}.png"
    dot_path.write_text(dot)
    subprocess.run(
        ["dot", "-Tpng", f"-Gdpi={dpi}", "-o", str(png_path), str(dot_path)],
        check=True,
    )
    print(f"  rendered {name}.png")


def node(nid: str, label: str, color: str = "indigo", fontsize: int = 12, shape: str = "box") -> str:
    c = PALETTE[color]
    if shape == "box":
        style = 'style="rounded,filled"'
    else:
        style = 'style="filled"'
    return (
        f'{nid} [label={label} shape={shape} {style} '
        f'fillcolor="{c["fill"]}" color="{c["stroke"]}" '
        f'fontname="{FONT}" fontsize={fontsize} fontcolor="{c["text"]}" '
        f'penwidth=1.6];'
    )


def html(title: str, body: str, color: str = "indigo") -> str:
    c = PALETTE[color]
    return (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">'
        f'<TR><TD><FONT POINT-SIZE="13" COLOR="{c["text"]}"><B>{title}</B></FONT></TD></TR>'
        f'<TR><TD><FONT POINT-SIZE="10" COLOR="{MUTED}">{body}</FONT></TD></TR>'
        f'</TABLE>>'
    )


def quoted(text: str) -> str:
    return '"' + text.replace('"', '\\"').replace("\n", "\\n") + '"'


HEADER = (
    f'rankdir=TB bgcolor="{BG}" fontname="{FONT}" '
    f'labelloc="t" labeljust="c" margin=0.15 nodesep=0.45 ranksep=0.55 '
    f'splines=spline'
)

EDGE_ATTR = f'edge [color="{EDGE}" penwidth=1.3 arrowsize=0.7 fontname="{FONT}" fontsize=10 fontcolor="{MUTED}"]'


# --------------------------------------------------------------------------- #
# 1. model_family_strengths
# --------------------------------------------------------------------------- #

def model_family_strengths() -> None:
    families = [
        ("mf",   "Matrix Factorization", "Captures linear latent patterns; fast baseline", "indigo"),
        ("neu",  "Neural CF",            "Non-linear user–item interactions via MLP",     "sky"),
        ("gn",   "Graph Networks",       "Propagates preferences over bipartite graph",   "teal"),
        ("cdr",  "Cross-Domain",         "Transfers taste from movies into games",        "rose"),
    ]
    body = [f'digraph G {{ {HEADER} rankdir=LR']
    body.append(f'  label={title_label("Four Model Families", "How each family models preference")};')
    for nid, title, desc, col in families:
        lbl = html(title, desc, col)
        body.append(f'  {nid} [label={lbl} shape=box style="rounded,filled" '
                    f'fillcolor="{PALETTE[col]["fill"]}" color="{PALETTE[col]["stroke"]}" '
                    f'penwidth=1.8 margin="0.2,0.12"];')
    # invisible ordering
    body.append("  mf -> neu -> gn -> cdr [style=invis];")
    body.append("}")
    render("model_family_strengths", "\n".join(body))


# --------------------------------------------------------------------------- #
# 2. cdr_concept
# --------------------------------------------------------------------------- #

def cdr_concept() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("Cross-Domain Recommendation", "Movies (source) inform Games (target)")};')
    body.append('  subgraph cluster_src {')
    body.append(f'    label="Source: Movies" fontname="{FONT}" fontsize=13 color="{PALETTE["indigo"]["stroke"]}" style="rounded" bgcolor="{PALETTE["indigo"]["fill"]}" margin=12;')
    body.append(f'    u1 [label="User u" {_nodeattrs("indigo", 12)}];')
    body.append(f'    m1 [label="Sci-Fi Movies" {_nodeattrs("indigo", 11)}];')
    body.append(f'    m2 [label="Action Movies" {_nodeattrs("indigo", 11)}];')
    body.append(f'    u1 -> m1; u1 -> m2;')
    body.append('  }')
    body.append('  subgraph cluster_tgt {')
    body.append(f'    label="Target: Games" fontname="{FONT}" fontsize=13 color="{PALETTE["emerald"]["stroke"]}" style="rounded" bgcolor="{PALETTE["emerald"]["fill"]}" margin=12;')
    body.append(f'    g1 [label="Shooter Games?" {_nodeattrs("emerald", 11)}];')
    body.append(f'    g2 [label="Strategy Games?" {_nodeattrs("emerald", 11)}];')
    body.append('  }')
    body.append(f'  bridge [label="Transfer\\nmodel" {_nodeattrs("amber", 12)}];')
    body.append('  u1 -> bridge [label="taste signal" color="#D97706"];')
    body.append('  bridge -> g1 [label="predict" color="#059669"];')
    body.append('  bridge -> g2 [color="#059669"];')
    body.append("}")
    render("cdr_concept", "\n".join(body))


def _nodeattrs(color: str, fontsize: int = 12) -> str:
    c = PALETTE[color]
    return (f'shape=box style="rounded,filled" fillcolor="{c["fill"]}" color="{c["stroke"]}" '
            f'fontname="{FONT}" fontsize={fontsize} fontcolor="{c["text"]}" penwidth=1.4')


# --------------------------------------------------------------------------- #
# 3. lesson_flow
# --------------------------------------------------------------------------- #

def lesson_flow() -> None:
    lessons = [
        ("1", "Baselines",        "MF-BPR / NCF", "indigo"),
        ("2", "Graph",            "LightGCN",     "sky"),
        ("3", "Shared Factors",   "CMF",          "teal"),
        ("4", "Mapping",          "EMCDR",        "emerald"),
        ("5", "Personalized MoE", "PTUPCDR",      "amber"),
        ("6", "Content",          "SBERT-CDR",    "rose"),
        ("7", "Rerank",           "Co-occurrence",  "violet"),
    ]
    body = [f'digraph G {{ {HEADER} rankdir=LR {EDGE_ATTR}']
    body.append(f'  label={title_label("Lesson Progression", "Each lesson adds one mechanism")};')
    for num, title, model, col in lessons:
        lbl = f'<<FONT POINT-SIZE="10" COLOR="{MUTED}">Lesson {num}</FONT><BR/>'
        lbl += f'<FONT POINT-SIZE="13" COLOR="{PALETTE[col]["text"]}"><B>{title}</B></FONT><BR/>'
        lbl += f'<FONT POINT-SIZE="10" COLOR="{MUTED}">{model}</FONT>>'
        body.append(f'  l{num} [label={lbl} shape=box style="rounded,filled" '
                    f'fillcolor="{PALETTE[col]["fill"]}" color="{PALETTE[col]["stroke"]}" '
                    f'fontname="{FONT}" penwidth=1.5 margin="0.18,0.12"];')
    body.append("  l1 -> l2 -> l3 -> l4 -> l5 -> l6 -> l7;")
    body.append("}")
    render("lesson_flow", "\n".join(body))


# --------------------------------------------------------------------------- #
# 4. data_pipeline
# --------------------------------------------------------------------------- #

def data_pipeline() -> None:
    body = [f'digraph G {{ {HEADER} rankdir=LR {EDGE_ATTR}']
    body.append(f'  label={title_label("Data Pipeline", "Amazon 2023 → ready-to-train split")};')
    steps = [
        ("raw",      "Raw JSONL.gz",       "Amazon 2023 reviews", "slate"),
        ("filter",   "Filter + Dedup",     "rating ≥ 4, merge variants", "indigo"),
        ("kcore",    "K-core = 10",        "dense user/item graph", "sky"),
        ("pair",     "Movie ∩ Game",       "cross-domain users only", "teal"),
        ("split",    "Leave-last-out",     "train / val / test", "emerald"),
        ("ready",    "Parquet",            "train-ready tensors", "violet"),
    ]
    for nid, title, desc, col in steps:
        lbl = html(title, desc, col)
        body.append(f'  {nid} [label={lbl} shape=box style="rounded,filled" '
                    f'fillcolor="{PALETTE[col]["fill"]}" color="{PALETTE[col]["stroke"]}" '
                    f'penwidth=1.6 margin="0.18,0.1"];')
    body.append("  raw -> filter -> kcore -> pair -> split -> ready;")
    body.append("}")
    render("data_pipeline", "\n".join(body))


# --------------------------------------------------------------------------- #
# 5. eval_protocol
# --------------------------------------------------------------------------- #

def eval_protocol() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("Evaluation Protocol", "Leave-last-out, full-rank, @10")};')
    body.append('  subgraph cluster_data {')
    body.append(f'    label="User history (chronological)" fontname="{FONT}" fontsize=12 color="{BORDER}" style="rounded" bgcolor="{PALETTE["slate"]["fill"]}" margin=10;')
    for i in range(1, 6):
        body.append(f'    h{i} [label="{i}" {_nodeattrs("slate", 11)} width=0.45 height=0.45 fixedsize=true];')
    body.append(f'    h6 [label="last" {_nodeattrs("rose", 11)} width=0.6 height=0.45 fixedsize=true];')
    body.append("    h1 -> h2 -> h3 -> h4 -> h5 -> h6 [style=invis];")
    body.append("    {rank=same; h1;h2;h3;h4;h5;h6}")
    body.append('  }')
    body.append(f'  train [label="Train on prefix" {_nodeattrs("indigo", 12)}];')
    body.append(f'  rank  [label="Rank all items" {_nodeattrs("sky", 12)}];')
    body.append(f'  score [label="Recall@10\\nNDCG@10\\nHit@10" {_nodeattrs("emerald", 12)}];')
    body.append('  h1 -> train [style=dashed color="#94A3B8"];')
    body.append('  h6 -> rank [label="held out" color="#E11D48"];')
    body.append('  train -> rank -> score;')
    body.append("}")
    render("eval_protocol", "\n".join(body))


# --------------------------------------------------------------------------- #
# 6. mfbpr_training
# --------------------------------------------------------------------------- #

def mfbpr_training() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("MF-BPR Training", "Pairwise ranking of positives over negatives")};')
    body.append(f'  u  [label="User u" {_nodeattrs("indigo", 12)}];')
    body.append(f'  ip [label="Positive item i+" {_nodeattrs("emerald", 12)}];')
    body.append(f'  ineg [label="Negative item i−" {_nodeattrs("rose", 12)}];')
    body.append(f'  eu [label="User factor\\np_u ∈ R^d" {_nodeattrs("sky", 11)}];')
    body.append(f'  eip [label="Item factor\\nq_i+ ∈ R^d" {_nodeattrs("teal", 11)}];')
    body.append(f'  ein [label="Item factor\\nq_i− ∈ R^d" {_nodeattrs("amber", 11)}];')
    body.append(f'  score [label="r̂ = p_u · q" {_nodeattrs("violet", 12)}];')
    body.append(f'  bpr [label="BPR loss\\n−log σ(r̂+ − r̂−)" {_nodeattrs("pink", 12)}];')
    body.append('  u -> eu; ip -> eip; ineg -> ein;')
    body.append('  eu -> score; eip -> score; ein -> score;')
    body.append('  score -> bpr;')
    body.append("}")
    render("mfbpr_training", "\n".join(body))


# --------------------------------------------------------------------------- #
# 7. mfbpr_inference
# --------------------------------------------------------------------------- #

def mfbpr_inference() -> None:
    body = [f'digraph G {{ {HEADER} rankdir=LR {EDGE_ATTR}']
    body.append(f'  label={title_label("MF-BPR Inference", "Dot-product top-K retrieval")};')
    body.append(f'  u [label="User u" {_nodeattrs("indigo", 12)}];')
    body.append(f'  pu [label="p_u" {_nodeattrs("sky", 11)}];')
    body.append(f'  cat [label="Catalogue\\nQ ∈ R^(N×d)" {_nodeattrs("slate", 11)}];')
    body.append(f'  dot [label="p_u · Q^T" {_nodeattrs("violet", 12)}];')
    body.append(f'  topk [label="Top-K" {_nodeattrs("emerald", 12)}];')
    body.append('  u -> pu; pu -> dot; cat -> dot; dot -> topk;')
    body.append("}")
    render("mfbpr_inference", "\n".join(body))


# --------------------------------------------------------------------------- #
# 8. ncf_gmf
# --------------------------------------------------------------------------- #

def ncf_gmf() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("GMF branch", "Generalised Matrix Factorisation")};')
    body.append(f'  u [label="User u" {_nodeattrs("indigo", 12)}];')
    body.append(f'  i [label="Item i" {_nodeattrs("emerald", 12)}];')
    body.append(f'  eu [label="p_u" {_nodeattrs("sky", 11)}];')
    body.append(f'  ei [label="q_i" {_nodeattrs("teal", 11)}];')
    body.append(f'  had [label="p_u &#8857; q_i\\n(element-wise)" {_nodeattrs("violet", 12)}];')
    body.append(f'  lin [label="Linear" {_nodeattrs("amber", 11)}];')
    body.append(f'  out [label="ŷ_gmf" {_nodeattrs("pink", 12)}];')
    body.append('  u -> eu; i -> ei; eu -> had; ei -> had; had -> lin -> out;')
    body.append("}")
    render("ncf_gmf", "\n".join(body))


# --------------------------------------------------------------------------- #
# 9. ncf_mlp
# --------------------------------------------------------------------------- #

def ncf_mlp() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("MLP branch", "Non-linear interactions")};')
    body.append(f'  u [label="User u" {_nodeattrs("indigo", 12)}];')
    body.append(f'  i [label="Item i" {_nodeattrs("emerald", 12)}];')
    body.append(f'  eu [label="p_u" {_nodeattrs("sky", 11)}];')
    body.append(f'  ei [label="q_i" {_nodeattrs("teal", 11)}];')
    body.append(f'  cat [label="concat [p_u ; q_i]" {_nodeattrs("violet", 12)}];')
    body.append(f'  h1 [label="FC + ReLU (64)" {_nodeattrs("amber", 11)}];')
    body.append(f'  h2 [label="FC + ReLU (32)" {_nodeattrs("amber", 11)}];')
    body.append(f'  out [label="ŷ_mlp" {_nodeattrs("pink", 12)}];')
    body.append('  u -> eu; i -> ei; eu -> cat; ei -> cat; cat -> h1 -> h2 -> out;')
    body.append("}")
    render("ncf_mlp", "\n".join(body))


# --------------------------------------------------------------------------- #
# 10. ncf_combined
# --------------------------------------------------------------------------- #

def ncf_combined() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("NeuMF", "Combining GMF and MLP")};')
    body.append(f'  u [label="User u" {_nodeattrs("indigo", 12)}];')
    body.append(f'  i [label="Item i" {_nodeattrs("emerald", 12)}];')
    body.append('  subgraph cluster_g {')
    body.append(f'    label="GMF" fontname="{FONT}" fontsize=11 color="{PALETTE["violet"]["stroke"]}" style="rounded" bgcolor="{PALETTE["violet"]["fill"]}" margin=8;')
    body.append(f'    gemb [label="p_u &#8857; q_i" {_nodeattrs("violet", 11)}];')
    body.append('  }')
    body.append('  subgraph cluster_m {')
    body.append(f'    label="MLP" fontname="{FONT}" fontsize=11 color="{PALETTE["amber"]["stroke"]}" style="rounded" bgcolor="{PALETTE["amber"]["fill"]}" margin=8;')
    body.append(f'    memb [label="MLP(p_u, q_i)" {_nodeattrs("amber", 11)}];')
    body.append('  }')
    body.append(f'  fuse [label="concat + Linear" {_nodeattrs("teal", 12)}];')
    body.append(f'  out [label="ŷ_neumf" {_nodeattrs("pink", 12)}];')
    body.append('  u -> gemb; u -> memb; i -> gemb; i -> memb;')
    body.append('  gemb -> fuse; memb -> fuse; fuse -> out;')
    body.append("}")
    render("ncf_combined", "\n".join(body))


# --------------------------------------------------------------------------- #
# 11. lightgcn_graph
# --------------------------------------------------------------------------- #

def lightgcn_graph() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR} rankdir=LR']
    body.append(f'  label={title_label("LightGCN Bipartite Graph", "Propagate embeddings over user-item edges")};')
    # users
    body.append('  subgraph cluster_u {')
    body.append(f'    label="Users" fontname="{FONT}" fontsize=12 color="{PALETTE["indigo"]["stroke"]}" style="rounded" bgcolor="{PALETTE["indigo"]["fill"]}" margin=10;')
    for u in "ABC":
        body.append(f'    U{u} [label="u{u}" {_nodeattrs("indigo", 11)} width=0.55 height=0.55 fixedsize=true];')
    body.append('  }')
    body.append('  subgraph cluster_i {')
    body.append(f'    label="Items" fontname="{FONT}" fontsize=12 color="{PALETTE["emerald"]["stroke"]}" style="rounded" bgcolor="{PALETTE["emerald"]["fill"]}" margin=10;')
    for i in "12345":
        body.append(f'    I{i} [label="i{i}" {_nodeattrs("emerald", 11)} shape=circle width=0.55 fixedsize=true];')
    body.append('  }')
    edges = [("UA","I1"),("UA","I2"),("UB","I1"),("UB","I3"),("UB","I4"),("UC","I2"),("UC","I4"),("UC","I5")]
    for a,b in edges:
        body.append(f'  {a} -> {b} [arrowhead=none color="{EDGE}"];')
    body.append("}")
    render("lightgcn_graph", "\n".join(body))


# --------------------------------------------------------------------------- #
# 12. lightgcn_training
# --------------------------------------------------------------------------- #

def lightgcn_training() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("LightGCN Layer Propagation", "K hops, averaged into final embedding")};')
    layers = [("e0","e^(0)","slate"),("e1","e^(1) = Ã · e^(0)","indigo"),("e2","e^(2)","sky"),("e3","e^(3)","teal")]
    for nid, lbl, col in layers:
        body.append(f'  {nid} [label="{lbl}" {_nodeattrs(col, 12)}];')
    body.append(f'  avg [label="e* = (1/K) Σ e^(k)" {_nodeattrs("violet", 12)}];')
    body.append(f'  dot [label="ŷ = e*_u · e*_i" {_nodeattrs("pink", 12)}];')
    body.append('  e0 -> e1 -> e2 -> e3;')
    body.append('  e0 -> avg; e1 -> avg; e2 -> avg; e3 -> avg;')
    body.append('  avg -> dot;')
    body.append("}")
    render("lightgcn_training", "\n".join(body))


# --------------------------------------------------------------------------- #
# 13. cmf_mechanism
# --------------------------------------------------------------------------- #

def cmf_mechanism() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("Collective Matrix Factorisation", "Users share latent factors across domains")};')
    body.append(f'  Rm [label="Ratings M\\n(users × movies)" {_nodeattrs("indigo", 11)}];')
    body.append(f'  Rg [label="Ratings G\\n(users × games)" {_nodeattrs("emerald", 11)}];')
    body.append(f'  U  [label="Shared user\\nfactors  P" {_nodeattrs("violet", 12)}];')
    body.append(f'  Qm [label="Movie factors\\nQ_m" {_nodeattrs("sky", 11)}];')
    body.append(f'  Qg [label="Game factors\\nQ_g" {_nodeattrs("teal", 11)}];')
    body.append('  Rm -> U [arrowhead=none color="#4F46E5"];')
    body.append('  Rg -> U [arrowhead=none color="#059669"];')
    body.append('  U -> Qm [style=dashed color="#0284C7"];')
    body.append('  U -> Qg [style=dashed color="#0D9488"];')
    body.append("}")
    render("cmf_mechanism", "\n".join(body))


# --------------------------------------------------------------------------- #
# 14. emcdr_phases
# --------------------------------------------------------------------------- #

def emcdr_phases() -> None:
    body = [f'digraph G {{ {HEADER} rankdir=LR {EDGE_ATTR}']
    body.append(f'  label={title_label("EMCDR: Three Phases", "Pretrain → Map → Serve")};')
    body.append(f'  p1 [label="Phase 1\\nTrain MF on movies" {_nodeattrs("indigo", 12)}];')
    body.append(f'  p2 [label="Phase 2\\nTrain MF on games" {_nodeattrs("emerald", 12)}];')
    body.append(f'  p3 [label="Phase 3\\nLearn f: U_m → U_g\\non shared users" {_nodeattrs("violet", 12)}];')
    body.append(f'  p4 [label="Serve\\npredict games\\nfrom movie embedding" {_nodeattrs("amber", 12)}];')
    body.append("  p1 -> p3; p2 -> p3; p3 -> p4;")
    body.append("  p1 -> p2 [style=invis]; {rank=same; p1; p2}")
    body.append("}")
    render("emcdr_phases", "\n".join(body))


# --------------------------------------------------------------------------- #
# 15. ptupcdr_moe
# --------------------------------------------------------------------------- #

def ptupcdr_moe() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("PTUPCDR: Personalised Transfer", "A meta-network produces a per-user mapping")};')
    body.append(f'  um [label="User movie\\nembedding u_m" {_nodeattrs("indigo", 12)}];')
    body.append(f'  meta [label="Meta-network\\nMLP(u_m) → θ_u" {_nodeattrs("violet", 12)}];')
    body.append(f'  exp [label="Per-user bridge\\nf_θ_u(·)" {_nodeattrs("amber", 12)}];')
    body.append(f'  ug [label="Game-space embedding\\nû_g" {_nodeattrs("emerald", 12)}];')
    body.append('  um -> meta -> exp; um -> exp [style=dashed label="forward"]; exp -> ug;')
    body.append("}")
    render("ptupcdr_moe", "\n".join(body))


# --------------------------------------------------------------------------- #
# 16. sbert_encoding
# --------------------------------------------------------------------------- #

def sbert_encoding() -> None:
    body = [f'digraph G {{ {HEADER} rankdir=LR {EDGE_ATTR}']
    body.append(f'  label={title_label("SBERT Title Encoding", "Titles → 384-d sentence embeddings")};')
    body.append(f'  title [label="Title: &quot;Dune: Part Two&quot;" {_nodeattrs("slate", 11)}];')
    body.append(f'  tok [label="Tokenise\\n[CLS] Dune : Part Two [SEP]" {_nodeattrs("indigo", 11)}];')
    body.append(f'  bert [label="MiniLM encoder\\n(sentence-transformers)" {_nodeattrs("sky", 11)}];')
    body.append(f'  pool [label="Mean pooling" {_nodeattrs("teal", 11)}];')
    body.append(f'  emb [label="v ∈ R^384" {_nodeattrs("violet", 12)}];')
    body.append('  title -> tok -> bert -> pool -> emb;')
    body.append("}")
    render("sbert_encoding", "\n".join(body))


# --------------------------------------------------------------------------- #
# 17. sbert_cdr
# --------------------------------------------------------------------------- #

def sbert_cdr() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("SBERT-CDR Scoring", "Movie history → content-based game prediction")};')
    body.append('  subgraph cluster_m {')
    body.append(f'    label="User\'s movie history" fontname="{FONT}" fontsize=11 color="{PALETTE["indigo"]["stroke"]}" style="rounded" bgcolor="{PALETTE["indigo"]["fill"]}" margin=8;')
    for k, t in enumerate(["Dune", "Oppenheimer", "Blade Runner"], 1):
        body.append(f'    m{k} [label="{t}" {_nodeattrs("indigo", 10)}];')
    body.append('  }')
    body.append(f'  uprof [label="User profile\\nmean(v_movies)" {_nodeattrs("violet", 12)}];')
    body.append('  subgraph cluster_g {')
    body.append(f'    label="Game catalogue" fontname="{FONT}" fontsize=11 color="{PALETTE["emerald"]["stroke"]}" style="rounded" bgcolor="{PALETTE["emerald"]["fill"]}" margin=8;')
    for k, t in enumerate(["Cyberpunk 2077", "Civilization VI", "Stellaris"], 1):
        body.append(f'    g{k} [label="{t}" {_nodeattrs("emerald", 10)}];')
    body.append('  }')
    body.append(f'  cos [label="cosine(uprof, v_game)" {_nodeattrs("pink", 12)}];')
    for k in "123":
        body.append(f'  m{k} -> uprof;')
    for k in "123":
        body.append(f'  g{k} -> cos;')
    body.append('  uprof -> cos;')
    body.append("}")
    render("sbert_cdr", "\n".join(body))


# --------------------------------------------------------------------------- #
# 18. cooc_matrix
# --------------------------------------------------------------------------- #

def cooc_matrix() -> None:
    body = [f'digraph G {{ {HEADER} rankdir=LR {EDGE_ATTR}']
    body.append(f'  label={title_label("Co-occurrence Matrix", "How often movie m and game g are liked together")};')
    body.append(f'  hist [label="Shared users\\n&#123;u : liked(u,m) ∧ liked(u,g)&#125;" {_nodeattrs("indigo", 11)}];')
    body.append(f'  cooc [label="C[m, g] = |shared(m, g)|" {_nodeattrs("violet", 12)}];')
    body.append(f'  ppmi [label="PPMI[m, g] =\\nmax(0, log p(m,g) − log p(m)p(g))" {_nodeattrs("amber", 12)}];')
    body.append('  hist -> cooc -> ppmi;')
    body.append("}")
    render("cooc_matrix", "\n".join(body))


# --------------------------------------------------------------------------- #
# 19. cooc_reranking
# --------------------------------------------------------------------------- #

def cooc_reranking() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("Co-occurrence Reranking", "Cheap prior on top of any base model")};')
    body.append(f'  base [label="Base model\\nŷ_cdr" {_nodeattrs("indigo", 12)}];')
    body.append(f'  prior [label="Co-occurrence\\nPPMI prior" {_nodeattrs("violet", 12)}];')
    body.append(f'  blend [label="α · ŷ_cdr + (1−α) · PPMI" {_nodeattrs("amber", 12)}];')
    body.append(f'  out [label="Final top-K" {_nodeattrs("emerald", 12)}];')
    body.append('  base -> blend; prior -> blend; blend -> out;')
    body.append("}")
    render("cooc_reranking", "\n".join(body))


# --------------------------------------------------------------------------- #
# 20. coldstart_comparison
# --------------------------------------------------------------------------- #

def coldstart_comparison() -> None:
    body = [f'digraph G {{ {HEADER} rankdir=LR {EDGE_ATTR}']
    body.append(f'  label={title_label("Cold-start Handling", "Which models survive a new user?")};')
    body.append(f'  user [label="New user\\nonly movie history" {_nodeattrs("slate", 12)}];')
    good = [("sbert", "SBERT-CDR", "content-only ✓"),
            ("ptup", "PTUPCDR",   "meta-network ✓"),
            ("emcdr","EMCDR",     "mapping ✓")]
    bad = [("mfbpr","MF-BPR", "no game ID ✗"),
           ("cmf",  "CMF",    "needs any game rating ✗"),
           ("lgcn", "LightGCN","needs game graph ✗")]
    body.append('  subgraph cluster_good {')
    body.append(f'    label="Works for cold-start" fontname="{FONT}" fontsize=12 color="{PALETTE["emerald"]["stroke"]}" style="rounded" bgcolor="{PALETTE["emerald"]["fill"]}" margin=10;')
    for nid, name, why in good:
        lbl = html(name, why, "emerald")
        body.append(f'    {nid} [label={lbl} shape=box style="rounded,filled" '
                    f'fillcolor="{PALETTE["emerald"]["fill"]}" color="{PALETTE["emerald"]["stroke"]}" penwidth=1.4];')
    body.append('  }')
    body.append('  subgraph cluster_bad {')
    body.append(f'    label="Fails for cold-start" fontname="{FONT}" fontsize=12 color="{PALETTE["rose"]["stroke"]}" style="rounded" bgcolor="{PALETTE["rose"]["fill"]}" margin=10;')
    for nid, name, why in bad:
        lbl = html(name, why, "rose")
        body.append(f'    {nid} [label={lbl} shape=box style="rounded,filled" '
                    f'fillcolor="{PALETTE["rose"]["fill"]}" color="{PALETTE["rose"]["stroke"]}" penwidth=1.4];')
    body.append('  }')
    for nid, *_ in good + bad:
        body.append(f'  user -> {nid} [color="{EDGE}"];')
    body.append("}")
    render("coldstart_comparison", "\n".join(body))


# --------------------------------------------------------------------------- #
# 21. routing_rule
# --------------------------------------------------------------------------- #

def routing_rule() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("Routing Rule", "Pick the model for each user")};')
    body.append(f'  req [label="Incoming user" {_nodeattrs("slate", 12)}];')
    body.append(f'  chk [label="Has game interactions?" shape=diamond style="filled" fillcolor="{PALETTE["amber"]["fill"]}" color="{PALETTE["amber"]["stroke"]}" fontname="{FONT}" fontsize=12 fontcolor="{PALETTE["amber"]["text"]}" penwidth=1.6];')
    body.append(f'  warm [label="Warm start\\nPTUPCDR + rerank" {_nodeattrs("emerald", 12)}];')
    body.append(f'  cold [label="Cold start\\nSBERT-CDR + rerank" {_nodeattrs("violet", 12)}];')
    body.append('  req -> chk;')
    body.append('  chk -> warm [label="yes" color="#059669"];')
    body.append('  chk -> cold [label="no"  color="#7C3AED"];')
    body.append("}")
    render("routing_rule", "\n".join(body))


# --------------------------------------------------------------------------- #
# 22. system_architecture
# --------------------------------------------------------------------------- #

def system_architecture() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("System Architecture", "End-to-end serving stack")};')
    # client
    body.append('  subgraph cluster_client {')
    body.append(f'    label="Client" fontname="{FONT}" fontsize=12 color="{PALETTE["indigo"]["stroke"]}" style="rounded" bgcolor="{PALETTE["indigo"]["fill"]}" margin=10;')
    body.append(f'    web [label="React app" {_nodeattrs("indigo", 11)}];')
    body.append('  }')
    # api
    body.append('  subgraph cluster_api {')
    body.append(f'    label="API" fontname="{FONT}" fontsize=12 color="{PALETTE["sky"]["stroke"]}" style="rounded" bgcolor="{PALETTE["sky"]["fill"]}" margin=10;')
    body.append(f'    fastapi [label="FastAPI\\n/recommend" {_nodeattrs("sky", 11)}];')
    body.append(f'    router [label="Router\\n(warm / cold)" {_nodeattrs("sky", 11)}];')
    body.append('    fastapi -> router [color="#0284C7"];')
    body.append('  }')
    # model layer
    body.append('  subgraph cluster_models {')
    body.append(f'    label="Models" fontname="{FONT}" fontsize=12 color="{PALETTE["violet"]["stroke"]}" style="rounded" bgcolor="{PALETTE["violet"]["fill"]}" margin=10;')
    body.append(f'    ptup [label="PTUPCDR" {_nodeattrs("violet", 11)}];')
    body.append(f'    sbert [label="SBERT-CDR" {_nodeattrs("violet", 11)}];')
    body.append(f'    cooc [label="Co-occurrence\\nrerank" {_nodeattrs("violet", 11)}];')
    body.append("    {rank=same; ptup; sbert; cooc}")
    body.append("    ptup -> sbert [style=invis]; sbert -> cooc [style=invis];")
    body.append('  }')
    # data
    body.append('  subgraph cluster_data {')
    body.append(f'    label="Data" fontname="{FONT}" fontsize=12 color="{PALETTE["emerald"]["stroke"]}" style="rounded" bgcolor="{PALETTE["emerald"]["fill"]}" margin=10;')
    body.append(f'    parquet [label="Parquet\\ncatalogue + user state" {_nodeattrs("emerald", 11)}];')
    body.append(f'    embed [label="Item embeddings\\nRedis cache" {_nodeattrs("emerald", 11)}];')
    body.append("    {rank=same; parquet; embed}")
    body.append("    parquet -> embed [style=invis];")
    body.append('  }')
    body.append('  web -> fastapi;')
    body.append('  router -> ptup; router -> sbert;')
    body.append('  ptup -> cooc; sbert -> cooc;')
    body.append('  ptup -> embed [style=dashed color="#7C3AED"];')
    body.append('  sbert -> embed [style=dashed color="#7C3AED"];')
    body.append('  cooc -> parquet [style=dashed color="#7C3AED"];')
    body.append("}")
    render("system_architecture", "\n".join(body))


# --------------------------------------------------------------------------- #
# 23. recommendation_flow
# --------------------------------------------------------------------------- #

def recommendation_flow() -> None:
    body = [f'digraph G {{ {HEADER} rankdir=LR {EDGE_ATTR}']
    body.append(f'  label={title_label("Recommendation Flow", "Single /recommend call")};')
    steps = [
        ("req",   "1. POST /recommend", "user_id + optional filters", "indigo"),
        ("hist",  "2. Load history",    "user movie interactions",    "sky"),
        ("route", "3. Route",           "warm vs cold",               "amber"),
        ("score", "4. Score",           "top-200 candidate games",    "teal"),
        ("rerank","5. Rerank",          "PPMI + diversity",           "violet"),
        ("resp",  "6. Response",        "top-10 game objects",        "emerald"),
    ]
    for nid, title, desc, col in steps:
        lbl = html(title, desc, col)
        body.append(f'  {nid} [label={lbl} shape=box style="rounded,filled" '
                    f'fillcolor="{PALETTE[col]["fill"]}" color="{PALETTE[col]["stroke"]}" '
                    f'penwidth=1.5 margin="0.15,0.08"];')
    body.append("  req -> hist -> route -> score -> rerank -> resp;")
    body.append("}")
    render("recommendation_flow", "\n".join(body))


# --------------------------------------------------------------------------- #
# 24. frontend_architecture
# --------------------------------------------------------------------------- #

def frontend_architecture() -> None:
    body = [f'digraph G {{ {HEADER} {EDGE_ATTR}']
    body.append(f'  label={title_label("Frontend Architecture", "React app talks to FastAPI")};')
    body.append('  subgraph cluster_ui {')
    body.append(f'    label="UI Components" fontname="{FONT}" fontsize=12 color="{PALETTE["indigo"]["stroke"]}" style="rounded" bgcolor="{PALETTE["indigo"]["fill"]}" margin=10;')
    body.append(f'    search [label="Search bar" {_nodeattrs("indigo", 11)}];')
    body.append(f'    carousel [label="Movie carousel" {_nodeattrs("indigo", 11)}];')
    body.append(f'    grid [label="Recommendation grid" {_nodeattrs("indigo", 11)}];')
    body.append("    {rank=same; search; carousel; grid}")
    body.append("    search -> carousel [style=invis]; carousel -> grid [style=invis];")
    body.append('  }')
    body.append('  subgraph cluster_state {')
    body.append(f'    label="State" fontname="{FONT}" fontsize=12 color="{PALETTE["violet"]["stroke"]}" style="rounded" bgcolor="{PALETTE["violet"]["fill"]}" margin=10;')
    body.append(f'    tanstack [label="TanStack Query cache" {_nodeattrs("violet", 11)}];')
    body.append('  }')
    body.append('  subgraph cluster_api {')
    body.append(f'    label="API" fontname="{FONT}" fontsize=12 color="{PALETTE["emerald"]["stroke"]}" style="rounded" bgcolor="{PALETTE["emerald"]["fill"]}" margin=10;')
    body.append(f'    fastapi [label="FastAPI" {_nodeattrs("emerald", 11)}];')
    body.append('  }')
    body.append('  search -> tanstack [color="#7C3AED"];')
    body.append('  carousel -> tanstack [color="#7C3AED"];')
    body.append('  grid -> tanstack [color="#7C3AED"];')
    body.append('  tanstack -> fastapi [label="HTTP"];')
    body.append("}")
    render("frontend_architecture", "\n".join(body))


# --------------------------------------------------------------------------- #

def main():
    print(f"Rendering Graphviz diagrams to {OUT}/")
    model_family_strengths()
    cdr_concept()
    lesson_flow()
    data_pipeline()
    eval_protocol()
    mfbpr_training()
    mfbpr_inference()
    ncf_gmf()
    ncf_mlp()
    ncf_combined()
    lightgcn_graph()
    lightgcn_training()
    cmf_mechanism()
    emcdr_phases()
    ptupcdr_moe()
    sbert_encoding()
    sbert_cdr()
    cooc_matrix()
    cooc_reranking()
    coldstart_comparison()
    routing_rule()
    system_architecture()
    recommendation_flow()
    frontend_architecture()
    print(f"Done. Images in {OUT}/")


if __name__ == "__main__":
    main()
