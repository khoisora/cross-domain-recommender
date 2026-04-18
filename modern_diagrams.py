"""Modern redesign of all 37 report diagrams.

Design principles:
- Flat solid-fill boxes, no outlines, subtle drop-shadow
- Thick arrows (lw=2.2, mutation_scale=22)
- Proper spacing – zero overlaps
- Unified palette: NAVY / TEAL / GREEN / GOLD / VIOLET / CORAL
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY   = '#1B3A6B'
TEAL   = '#0F7EA8'
GREEN  = '#1A7A4A'
GOLD   = '#B8770A'
VIOLET = '#5B21B6'
CORAL  = '#C0392B'
SLATE  = '#475569'
LGRAY  = '#94A3B8'
WHITE  = '#FFFFFF'
BG     = '#FFFFFF'
PANELB = '#EEF4FF'   # blue tint panel
PANELG = '#EDFBF3'   # green tint
PANELA = '#FFFBEB'   # amber tint
PANELV = '#F5F3FF'   # violet tint
PANELR = '#FEF2F2'   # red tint
PANELN = '#F0F4FF'   # navy tint (light)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': BG})


# ── Helpers ───────────────────────────────────────────────────────────────────

def bx(ax, x, y, w, h, label, sub=None, bg=NAVY, fg=WHITE,
       fs=13, sfs=11, zo=3, lw=0, ec=None, alpha=1.0):
    """Rounded filled box with optional sub-label.
    Box is drawn at 78% of the given w×h, centred in that space,
    so boxes appear smaller while text stays large and prominent.
    """
    shrink = 0.22          # 11% inset on each side → 78% of declared size
    bx = x + w * shrink / 2
    by = y + h * shrink / 2
    bw = w * (1 - shrink)
    bh = h * (1 - shrink)
    # shadow
    sh = FancyBboxPatch((bx+.05, by-.05), bw, bh, boxstyle="round,pad=0.15",
                        fc='#00000016', ec='none', zorder=zo-1)
    ax.add_patch(sh)
    # box
    bx_ = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.15",
                         fc=bg, ec=ec or bg, lw=lw, alpha=alpha, zorder=zo)
    ax.add_patch(bx_)
    cy = y + h/2 + (h * 0.08 if sub else 0)
    ax.text(x+w/2, cy, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=fg, zorder=zo+1)
    if sub:
        ax.text(x+w/2, y + h * 0.28, sub, ha='center', va='center',
                fontsize=sfs, color=fg+'AA', zorder=zo+1)


def panel(ax, x, y, w, h, bg=PANELB, ec='#C7D8F0', lw=1.5,
          label=None, lc=NAVY, lfs=10, zo=1):
    """Background region panel."""
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                       fc=bg, ec=ec, lw=lw, alpha=0.65, zorder=zo)
    ax.add_patch(r)
    if label:
        ax.text(x+.18, y+h-.12, label, ha='left', va='top',
                fontsize=lfs, fontweight='bold', color=lc, zorder=zo+1)


def arr(ax, x1, y1, x2, y2, color=LGRAY, lw=2.2, cs=None, ms=22, style='->'):
    """Thick modern arrow."""
    kw = dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms)
    if cs:
        kw['connectionstyle'] = cs
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=kw, zorder=5)


def hdr(ax, W, H, title_text, sub_text=None):
    """Centered title + subtitle at top of axes."""
    ax.text(W/2, H-.25, title_text, ha='center', va='top',
            fontsize=16, fontweight='bold', color=NAVY)
    if sub_text:
        ax.text(W/2, H-.75, sub_text, ha='center', va='top',
                fontsize=11, color=LGRAY)


def setup(W, H):
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    fig.patch.set_facecolor(BG)
    return fig, ax


# =============================================================================
# 1. CDR CONCEPT
# =============================================================================
def fig_cdr_concept():
    W, H = 12, 6
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Cross-Domain Recommendation: Movies → Games",
        "Movie preferences transfer to game recommendations via shared user behaviour")

    panel(ax, .3, 1.2, 3.4, 3.5, PANELB, '#BBCFE8', label="Source Domain")
    panel(ax, 8.3, 1.2, 3.4, 3.5, PANELA, '#D4C09A', label="Target Domain")

    bx(ax, .5, 3.2, 3.0, 1.0, "Movie Domain", "Collaborative filtering", NAVY, WHITE, 12, 9)
    for i, m in enumerate(["Action Films", "Sci-Fi Films", "RPG Films"]):
        bx(ax, .6, 2.3-i*.65, 2.8, .50, m, bg=TEAL, fs=9)

    bx(ax, 4.4, 2.4, 3.2, 1.5, "Overlap User",
       "Watched 50 movies\nPlayed 2 games", VIOLET, WHITE, 11, 9)

    bx(ax, 8.5, 3.2, 3.0, 1.0, "Game Domain", "Target recommendations", GOLD+'CC', WHITE, 12, 9)
    for i, g in enumerate(["Action Games", "Sci-Fi Games", "RPG Games"]):
        bx(ax, 8.6, 2.3-i*.65, 2.8, .50, g, bg='#C47C10', fs=9)

    arr(ax, 3.5, 3.8, 4.4, 3.2, TEAL, 2.5)
    arr(ax, 7.6, 3.2, 8.5, 3.8, GOLD, 2.5)
    arr(ax, 3.4, 1.5, 8.3, 1.5, GREEN, 2.8, cs='arc3,rad=-0.25')
    ax.text(W/2, .9, '"Likes action movies → likely enjoys action games"',
            ha='center', fontsize=10, fontstyle='italic', color=GREEN)

    fig.tight_layout()
    fig.savefig(OUT/"cdr_concept.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ cdr_concept")


# =============================================================================
# 2. MF-BPR TRAINING
# =============================================================================
def fig_mfbpr_training():
    W, H = 11, 6.5
    fig, ax = setup(W, H)
    hdr(ax, W, H, "MF-BPR: Training with Pairwise Loss",
        "Push liked items above random items in the ranking")

    panel(ax, .3, 1.0, 10.4, 4.6, PANELB, '#BBCFE8')

    bx(ax, .6, 4.2, 2.2, 1.2, "User\nEmbedding", "64 dims", NAVY, WHITE, 12, 9)
    bx(ax, 3.5, 5.0, 2.2, 1.0, "Positive Item", "(liked, ≥4★)", GREEN, WHITE, 11, 9)
    bx(ax, 3.5, 3.5, 2.2, 1.0, "Negative Item", "(random sample)", CORAL, WHITE, 11, 9)
    bx(ax, 6.5, 5.0, 2.0, 1.0, "Score⁺", "u · i⁺", GREEN+'BB', WHITE, 12, 9)
    bx(ax, 6.5, 3.5, 2.0, 1.0, "Score⁻", "u · i⁻", CORAL+'BB', WHITE, 12, 9)
    bx(ax, 8.8, 4.0, 1.8, 1.5, "BPR\nLoss", "−log σ(s⁺−s⁻)", VIOLET, WHITE, 11, 8)

    arr(ax, 2.8, 4.8, 3.5, 5.3, NAVY)
    arr(ax, 2.8, 4.6, 3.5, 4.0, NAVY)
    arr(ax, 5.7, 5.4, 6.5, 5.4, GREEN)
    arr(ax, 5.7, 3.9, 6.5, 3.9, CORAL)
    arr(ax, 8.5, 5.3, 8.9, 5.0, GREEN)
    arr(ax, 8.5, 3.9, 8.9, 4.2, CORAL)

    bx(ax, 2.0, 1.2, 7.0, .8, "Backprop updates embeddings so liked items rank above random items",
       bg=PANELN, fg=NAVY, fs=10, lw=1.5, ec='#BBCFE8')

    fig.tight_layout()
    fig.savefig(OUT/"mfbpr_training.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ mfbpr_training")


# =============================================================================
# 3. MF-BPR INFERENCE
# =============================================================================
def fig_mfbpr_inference():
    W, H = 13, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "MF-BPR: Making Recommendations",
        "Using trained vectors to score and rank all games for a user")

    bx(ax, .4, 3.2, 2.4, 2.0, "User\nVector", "[0.8, −0.2, 0.5, …]", NAVY, WHITE, 13, 9)

    games = [("Game A", "[0.9, −0.1, 0.6, …]", GREEN,  "= 0.92", GREEN),
             ("Game B", "[0.1,  0.8, −0.3, …]", TEAL,  "= 0.35", TEAL),
             ("Game C", "[−0.5, 0.2,  0.1, …]", CORAL, "= −0.21", CORAL)]
    ys = [5.0, 3.6, 2.2]
    for (name, vec, col, score, sc), y in zip(games, ys):
        bx(ax, 3.4, y, 3.0, 1.2, name, vec, col, WHITE, 12, 9)
        ax.text(7.6, y+.6, score, ha='left', va='center',
                fontsize=14, fontweight='bold', color=sc)
        arr(ax, 2.8, 4.2, 3.4, y+.6, LGRAY, 1.8)
        arr(ax, 6.4, y+.6, 7.4, y+.6, col, 2.0)

    ax.text(6.7, 6.2, "Dot Product", ha='center', fontsize=10,
            fontweight='bold', color=LGRAY)
    arr(ax, 6.7, 6.0, 6.7, 5.7, LGRAY, 1.5)

    panel(ax, 8.6, 1.8, 3.8, 4.0, PANELG, '#9BD4B8')
    ax.text(10.5, 5.5, "Ranked List", ha='center', fontsize=13,
            fontweight='bold', color=GREEN)
    ax.text(10.5, 4.9, "1.  Game A  (0.92)", ha='center', fontsize=11, color=GREEN, fontweight='bold')
    ax.text(10.5, 4.3, "2.  Game B  (0.35)", ha='center', fontsize=11, color=TEAL)
    ax.text(10.5, 3.7, "3.  Game C  (−0.21)", ha='center', fontsize=11, color=CORAL)
    arr(ax, 7.8, 3.6, 8.6, 3.6, TEAL, 2.5)

    panel(ax, 1.5, .15, 10.0, .9, PANELB, '#BBCFE8')
    ax.text(W/2, .65, "score(user, game) = user_vector · game_vector   (dot product of 64 numbers)",
            ha='center', fontsize=11, fontweight='bold', color=NAVY)

    fig.tight_layout()
    fig.savefig(OUT/"mfbpr_inference.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ mfbpr_inference")


# =============================================================================
# 4. NCF GMF
# =============================================================================
def fig_ncf_gmf():
    W, H = 11, 6
    fig, ax = setup(W, H)
    hdr(ax, W, H, "NCF — Branch 1: GMF (Generalized Matrix Factorization)",
        "Element-wise multiplication captures linear feature interactions")

    bx(ax, .5, 2.8, 2.4, 1.6, "User\nEmbedding", "64 dimensions", NAVY, WHITE, 12, 9)
    bx(ax, .5, .8,  2.4, 1.6, "Item\nEmbedding", "64 dimensions", TEAL, WHITE, 12, 9)

    bx(ax, 4.3, 1.4, 2.8, 2.2, "Element-wise\nMultiply", "user[i] × item[i]", GOLD, WHITE, 12, 9)
    ax.text(W/2, .9, "[0.8, −0.2, 0.5] × [0.9, 0.3, −0.1]",
            ha='center', fontsize=10, color=SLATE)
    ax.text(W/2, .4, "= [0.72, −0.06, −0.05]",
            ha='center', fontsize=11, fontweight='bold', color=GOLD)

    bx(ax, 8.3, 1.8, 2.2, 1.4, "GMF\nOutput", "64-dim vector", GREEN, WHITE, 12, 9)

    arr(ax, 2.9, 3.6, 4.3, 2.9, NAVY, 2.4)
    arr(ax, 2.9, 1.6, 4.3, 2.0, TEAL, 2.4)
    arr(ax, 7.1, 2.5, 8.3, 2.5, GOLD, 2.4)

    fig.tight_layout()
    fig.savefig(OUT/"ncf_gmf.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ ncf_gmf")


# =============================================================================
# 5. NCF MLP
# =============================================================================
def fig_ncf_mlp():
    W, H = 13, 6
    fig, ax = setup(W, H)
    hdr(ax, W, H, "NCF — Branch 2: MLP (Multi-Layer Perceptron)",
        "Neural network layers learn non-linear interaction patterns")

    panel(ax, 2.2, 1.8, 9.8, 2.8, PANELB, '#BBCFE8')

    bx(ax, .3, 3.6, 2.0, 1.2, "User\nEmb.", "64-dim", NAVY, WHITE, 12, 9)
    bx(ax, .3, 2.0, 2.0, 1.2, "Item\nEmb.", "64-dim", SLATE, WHITE, 12, 9)

    bx(ax, 2.6, 2.4, 2.0, 1.6, "Concat",  "128-dim", TEAL, WHITE, 12, 9)
    bx(ax, 5.2, 2.4, 2.0, 1.6, "Layer 1", "128 neurons", NAVY, WHITE, 12, 9)
    bx(ax, 7.8, 2.4, 2.0, 1.6, "Layer 2", "64 neurons",  NAVY, WHITE, 12, 9)
    bx(ax, 10.4,2.4, 1.8, 1.6, "MLP\nOut", "32-dim",     GREEN, WHITE, 12, 9)

    arr(ax, 2.3, 4.2, 2.6, 3.6, NAVY, 2.2)
    arr(ax, 2.3, 2.6, 2.6, 2.8, SLATE, 2.2)
    arr(ax, 4.6, 3.2, 5.2, 3.2, TEAL, 2.2)
    arr(ax, 7.2, 3.2, 7.8, 3.2, NAVY, 2.2)
    arr(ax, 9.8, 3.2, 10.4,3.2, NAVY, 2.2)

    panel(ax, 1.8, .25, 9.4, .9, PANELB, '#BBCFE8')
    ax.text(W/2, .72, "Each layer:  output = ReLU( W × input + b )   — learns complex non-linear patterns",
            ha='center', fontsize=11, fontweight='bold', color=NAVY)

    fig.tight_layout()
    fig.savefig(OUT/"ncf_mlp.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ ncf_mlp")


# =============================================================================
# 6. NCF COMBINED (NeuMF)
# =============================================================================
def fig_ncf_combined():
    W, H = 13, 9
    fig, ax = setup(W, H)
    hdr(ax, W, H, "NCF — Combined NeuMF Architecture",
        "GMF and MLP outputs are concatenated, then a final layer produces the score")

    # GMF branch panel
    panel(ax, .4, 5.6, 7.8, 2.4, PANELA, '#D4C49A', label="GMF Branch", lc=GOLD)
    bx(ax, .7, 6.0, 2.0, 1.5, "User\nEmb.", "64-dim", NAVY, WHITE, 11, 9)
    bx(ax, 3.0, 6.0, 2.0, 1.5, "Item\nEmb.", "64-dim", TEAL, WHITE, 11, 9)
    bx(ax, 5.3, 5.9, 2.4, 1.7, "u ⊙ i", "64-dim output", GOLD, WHITE, 12, 9)

    arr(ax, 2.7, 6.75, 3.0, 6.75, NAVY, 2.2)
    arr(ax, 5.0, 6.75, 5.3, 6.75, TEAL, 2.2)

    # MLP branch panel
    panel(ax, .4, 2.8, 7.8, 2.4, PANELV, '#C4B0E8', label="MLP Branch", lc=VIOLET)
    bx(ax, .7, 3.2, 2.0, 1.5, "User\nEmb.", "64-dim", NAVY, WHITE, 11, 9)
    bx(ax, 3.0, 3.2, 2.0, 1.5, "Item\nEmb.", "64-dim", TEAL, WHITE, 11, 9)
    bx(ax, 5.3, 3.1, 2.4, 1.7, "MLP", "32-dim output", VIOLET, WHITE, 12, 9)

    arr(ax, 2.7, 3.95, 3.0, 3.95, NAVY, 2.2)
    arr(ax, 5.0, 3.95, 5.3, 3.95, TEAL, 2.2)

    # Merge
    bx(ax, 9.0, 4.6, 2.0, 2.0, "Concat", "96-dim", SLATE, WHITE, 12, 9)
    bx(ax, 11.3,4.8, 1.5, 1.6, "Score", "0 to 1",  GREEN, WHITE, 13, 9)

    arr(ax, 7.7, 6.75, 9.0, 5.8, GOLD, 2.4)
    arr(ax, 7.7, 3.95, 9.0, 4.8, VIOLET, 2.4)
    arr(ax, 11.0,5.6, 11.3,5.6, SLATE, 2.4)

    panel(ax, 1.5, .5, 10.0, 1.1, PANELB, '#BBCFE8')
    ax.text(W/2, 1.1, "score = sigmoid( hᵀ × [GMF_output ; MLP_output] )     trained with BCE loss",
            ha='center', fontsize=11, fontweight='bold', color=NAVY)

    fig.tight_layout()
    fig.savefig(OUT/"ncf_combined.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ ncf_combined")


# =============================================================================
# 7. LIGHTGCN GRAPH
# =============================================================================
def fig_lightgcn_graph():
    W, H = 14, 8
    fig, ax = setup(W, H)
    hdr(ax, W, H, "LightGCN: Graph Convolution for Recommendations",
        "Each user/item learns from its neighbours, and their neighbours, and so on")

    # Bipartite graph (left side)
    ax.text(2.5, 6.8, "User-Item Bipartite Graph", ha='center',
            fontsize=12, fontweight='bold', color=NAVY)
    users_y = [5.8, 4.5, 3.2]
    items_y = [6.2, 5.2, 4.0, 2.9]
    for i, y in enumerate(users_y):
        circ = plt.Circle((1.5, y), .38, color=NAVY, zorder=3)
        ax.add_patch(circ)
        ax.text(1.5, y, f"U{i+1}", ha='center', va='center',
                fontsize=10, color=WHITE, fontweight='bold', zorder=4)
    for i, y in enumerate(items_y):
        circ = plt.Circle((3.5, y), .38, color=TEAL, zorder=3)
        ax.add_patch(circ)
        ax.text(3.5, y, f"G{i+1}", ha='center', va='center',
                fontsize=10, color=WHITE, fontweight='bold', zorder=4)
    edges = [(0,0),(0,1),(1,1),(1,2),(2,2),(2,3),(0,3)]
    for ui, gi in edges:
        ax.plot([1.88, 3.12], [users_y[ui], items_y[gi]], '-',
                color='#94A3B8', lw=1.5, zorder=1, alpha=.7)

    # Information flow (right side)
    ax.text(9.5, 6.8, "Information Flow for U1", ha='center',
            fontsize=12, fontweight='bold', color=TEAL)

    bx(ax, 5.8, 5.8, 2.4, 1.2, "Layer 0", "U1 initial emb  e_U1^(0)", NAVY, WHITE, 11, 8)
    bx(ax, 9.0, 5.8, 2.4, 1.2, "Layer 1", "Avg(G1,G2) embs  e_U1^(1)", TEAL, WHITE, 11, 8)
    bx(ax, 7.0, 3.8, 2.4, 1.2, "Layer 0\n(items)", "G1, G2 embeddings", SLATE, WHITE, 10, 8)
    bx(ax, 9.0, 3.8, 2.4, 1.2, "Layer 2", "2-hop neighbours", GOLD, WHITE, 11, 8)

    arr(ax, 8.2, 6.4, 9.0, 6.4, NAVY, 2.4)
    arr(ax, 7.2, 4.4, 7.8, 6.2, SLATE, 2.0, cs='arc3,rad=-0.3')
    arr(ax, 9.4, 5.8, 9.6, 5.0, LGRAY, 2.0)
    ax.text(10.0, 5.3, "through G1,G2", fontsize=8, color=LGRAY)

    panel(ax, 5.4, 1.4, 8.0, 1.6, PANELG, '#9BD4B8')
    ax.text(9.4, 2.4, "Final U1 Embedding", ha='center',
            fontsize=12, fontweight='bold', color=GREEN)
    ax.text(9.4, 1.9, "e_U1 = ( e^(0) + e^(1) + e^(2) ) / 3",
            ha='center', fontsize=12, fontweight='bold', color=GREEN)
    ax.text(9.4, 1.55, "Average across all layers captures local + global structure",
            ha='center', fontsize=9, color=LGRAY)

    arr(ax, 10.2, 3.8, 11.0, 3.0, GOLD, 2.2)
    arr(ax, 11.0, 3.0, 11.0, 1.8, GREEN, 2.2)

    fig.tight_layout()
    fig.savefig(OUT/"lightgcn_graph.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ lightgcn_graph")


# =============================================================================
# 8. LIGHTGCN TRAINING
# =============================================================================
def fig_lightgcn_training():
    W, H = 13, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "LightGCN: Training Loop",
        "Same BPR loss as MF-BPR, but embeddings are enriched by graph convolution first")

    panel(ax, .5, 1.4, 12.0, 4.0, PANELB, '#BBCFE8')

    steps = [
        ("Graph\nConvolution", "K=3 hops", TEAL),
        ("Get Final\nEmbeddings", "avg all layers", NAVY),
        ("BPR Loss", "liked > random", GREEN),
        ("Backprop\n+ Update", "Only Layer-0 embs", CORAL),
    ]
    xs = [.9, 3.7, 6.5, 9.3]
    for (lbl, sub, col), x in zip(steps, xs):
        bx(ax, x, 2.2, 2.4, 2.2, lbl, sub, col, WHITE, 12, 9)

    for i in range(3):
        arr(ax, xs[i]+2.4, 3.3, xs[i+1], 3.3, LGRAY, 2.5)

    # Loop back arrow
    arr(ax, 11.7, 4.4, 1.4, 4.4, GOLD, 2.2,
        cs='arc3,rad=-0.4', style='->')
    ax.text(6.5, 5.35, "Training Loop  (Adam optimizer, lr=0.001)",
            ha='center', fontsize=11, fontweight='bold', color=GOLD)

    panel(ax, 1.5, .25, 10.0, .9, PANELB, '#BBCFE8')
    ax.text(W/2, .72, "Key: Gradients flow BACK through the graph convolution to update Layer-0 embeddings only",
            ha='center', fontsize=10, fontweight='bold', color=NAVY)

    fig.tight_layout()
    fig.savefig(OUT/"lightgcn_training.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ lightgcn_training")


# =============================================================================
# 9. MODEL CMF
# =============================================================================
def fig_model_cmf():
    W, H = 13, 8
    fig, ax = setup(W, H)
    hdr(ax, W, H, "CMF: Collective Matrix Factorization",
        "One shared user embedding trained on both movie and game data simultaneously")

    panel(ax, .4, 3.2, 4.6, 3.2, PANELB, '#BBCFE8', label="Movie Domain", lc=NAVY)
    bx(ax, .7, 4.8, 1.9, 1.5, "Movie\nItems", "embeddings", NAVY, WHITE, 11, 9)
    bx(ax, 2.9, 4.8, 1.9, 1.5, "BPR\nLoss", "L_movie", SLATE, WHITE, 11, 9)
    ax.text(2.5, 3.4, "Movie ratings", ha='center', fontsize=9,
            fontstyle='italic', color=LGRAY)

    panel(ax, 8.0, 3.2, 4.6, 3.2, PANELA, '#D4C49A', label="Game Domain", lc=GOLD)
    bx(ax, 8.3, 4.8, 1.9, 1.5, "Game\nItems", "embeddings", TEAL, WHITE, 11, 9)
    bx(ax, 10.2,4.8, 1.9, 1.5, "BPR\nLoss", "L_game",   SLATE, WHITE, 11, 9)
    ax.text(10.5,3.4, "Game ratings",  ha='center', fontsize=9,
            fontstyle='italic', color=LGRAY)

    bx(ax, 4.9, 4.6, 3.2, 1.9, "Shared\nUser Emb.", "Same vector for both", GOLD, WHITE, 13, 9)

    arr(ax, 4.9, 5.5, 4.8, 5.5, GOLD, 2.4)
    arr(ax, 8.0, 5.5, 8.1, 5.5, GOLD, 2.4)

    panel(ax, 1.5, 1.2, 10.0, 1.4, PANELB, '#BBCFE8')
    ax.text(W/2, 2.1, "Joint Loss = 0.05 × L_movie  +  0.95 × L_game",
            ha='center', fontsize=13, fontweight='bold', color=NAVY)
    ax.text(W/2, 1.4, "Heavily weighted toward games (target domain); movies provide supplementary signal",
            ha='center', fontsize=9, color=LGRAY)

    fig.tight_layout()
    fig.savefig(OUT/"model_cmf.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ model_cmf")


# =============================================================================
# 10. EMCDR PHASES
# =============================================================================
def fig_emcdr_phases():
    W, H = 14, 9
    fig, ax = setup(W, H)
    hdr(ax, W, H, "EMCDR: Three-Phase Training",
        "Train two separate models, then learn a mapping between their embedding spaces")

    # Phase boxes
    phases = [
        ("Phase 1\n(20 epochs)", "Movie Ratings\n→ MF-BPR Model", PANELB, '#BBCFE8', NAVY, NAVY),
        ("Phase 2\n(20 epochs)", "Game Ratings\n→ MF-BPR Model",  PANELG, '#9BD4B8', TEAL, TEAL),
        ("Phase 3\n(10 epochs)", "Mapping MLP\n+ MSE Loss",        PANELA, '#D4C49A', GOLD, GOLD),
    ]
    px = [.5, 4.5, 8.5]
    for (ph, desc, bg, ec, lc, ac), x in zip(phases, px):
        panel(ax, x, 5.8, 3.6, 2.6, bg, ec, label=ph, lc=lc)
        bx(ax, x+.3, 6.1, 1.3, 1.9, desc.split('\n')[0], None, ac, WHITE, 10)
        bx(ax, x+1.9, 6.1, 1.4, 1.9, desc.split('\n')[1], None, SLATE, WHITE, 9)
        if desc.split('\n')[0] == "Mapping MLP":
            ax.text(x+1.9+.7, 5.95, "Trained on\noverlap users",
                    ha='center', fontsize=8, color=LGRAY)

    arr(ax, 4.1, 7.1, 4.5, 7.1, NAVY, 2.4)
    arr(ax, 8.1, 7.1, 8.5, 7.1, TEAL, 2.4)

    # Inference section
    panel(ax, .4, 1.5, 13.2, 3.5, PANELG, '#9BD4B8', label="Inference (Cold-Start User)", lc=GREEN)

    inf_boxes = [
        ("User's Movie\nEmbedding", None, NAVY),
        ("Mapping\nMLP", None, GOLD),
        ("Predicted\nGame Emb.", None, TEAL),
        ("Rank\nGames", None, GREEN),
    ]
    ix = [.7, 3.7, 6.7, 9.7]
    for (lbl, sub, col), x in zip(inf_boxes, ix):
        bx(ax, x, 2.4, 2.6, 1.9, lbl, sub, col, WHITE, 12)
    for i in range(3):
        arr(ax, ix[i]+2.6, 3.35, ix[i+1], 3.35,
            [NAVY, GOLD, TEAL][i], 2.5)

    ax.text(W/2, 1.75,
            "game_emb = MLP(movie_emb)   then   score = dot(game_emb, game_item)",
            ha='center', fontsize=10, color=SLATE)

    fig.tight_layout()
    fig.savefig(OUT/"emcdr_phases.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ emcdr_phases")


# =============================================================================
# 11. MODEL EMCDR
# =============================================================================
def fig_model_emcdr():
    W, H = 13, 8
    fig, ax = setup(W, H)
    hdr(ax, W, H, "EMCDR: Embedding & Mapping Cross-Domain Recommendation",
        "Train two separate models, then learn a bridge function for cold-start users")

    bx(ax, .4, 5.2, 2.8, 1.8, "Movie MF\nModel", "e_movie(u)", NAVY, WHITE, 12, 9)
    bx(ax, 9.8, 5.2, 2.8, 1.8, "Game MF\nModel",  "e_game(u)", TEAL, WHITE, 12, 9)
    bx(ax, 4.5, 5.2, 4.0, 1.8, "Global Mapping\nMLP",
       "f: movie_space → game_space", GREEN, WHITE, 12, 9)

    arr(ax, 3.2, 6.1, 4.5, 6.1, NAVY, 2.5)
    arr(ax, 9.8, 6.1, 8.5, 6.1, TEAL, 2.5)

    bx(ax, 2.0, 2.6, 9.0, 1.6,
       "Inference:  ê_game(u) = MLP( e_movie(u) )   →   score = dot(ê_game, game_item)",
       bg=PANELG, fg=NAVY, fs=11, lw=1.5, ec='#9BD4B8')
    arr(ax, 6.5, 5.2, 6.5, 4.2, GREEN, 2.4)

    ax.text(1.8, 7.6, "① Train\n(20 epochs)", ha='center',
            fontsize=9, color=NAVY, fontweight='bold')
    ax.text(11.2, 7.6, "② Train\n(20 epochs)", ha='center',
            fontsize=9, color=TEAL, fontweight='bold')
    ax.text(6.5, 7.6, "③ Map\n(10 epochs)", ha='center',
            fontsize=9, color=GREEN, fontweight='bold')

    ax.text(.7, 4.8, "Train on overlap users\nMSE loss in embedding space",
            ha='left', fontsize=8, color=LGRAY)

    fig.tight_layout()
    fig.savefig(OUT/"model_emcdr.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ model_emcdr")


# =============================================================================
# 12. PTUPCDR MOE
# =============================================================================
def fig_ptupcdr_moe():
    W, H = 14, 9
    fig, ax = setup(W, H)
    hdr(ax, W, H, "PTUPCDR: Personalized Transfer via Mixture of Experts",
        "Each user gets a custom translator built by blending 8 expert networks")

    bx(ax, .4, 3.8, 2.4, 2.8, "User's\nMovie Emb.", "from Phase 1", NAVY, WHITE, 12, 9)

    panel(ax, 3.2, 2.2, 4.4, 5.0, PANELV, '#C4B0E8', label="8 Expert Networks", lc=VIOLET)
    ax.text(5.4, 6.8, "Each expert is a small MLP\nthat maps movie→game space",
            ha='center', fontsize=8, color=LGRAY)
    expert_ys = [6.1, 5.2, 4.3, 3.2]
    expert_labels = ["Expert 1", "Expert 2", "Expert 3", "Expert 8"]
    for y, lbl in zip(expert_ys, expert_labels):
        col = VIOLET if lbl != "Expert 8" else SLATE
        bx(ax, 3.5, y-.35, 3.8, .8, lbl, bg=col, fg=WHITE, fs=10)
    ax.text(5.4, 3.7, "· · ·", ha='center', fontsize=16, color=VIOLET)

    bx(ax, 8.4, 5.2, 2.4, 1.8, "Gate\nNetwork", "Scorer weights\nper expert", TEAL, WHITE, 12, 9)
    bx(ax, 8.4, 3.0, 2.4, 1.8, "Actual Game\nEmb.", "(if user has games)", TEAL, WHITE, 11, 9)

    bx(ax, 11.2,5.8, 2.0, 1.8, "Mapped\nGame Emb.", None, GOLD, WHITE, 11, 9)
    bx(ax, 11.2,3.3, 2.0, 1.6, "Blend", None, '#D4690A', WHITE, 13, 9)

    arr(ax, 2.8, 5.2, 3.5, 5.8, NAVY, 2.2)
    arr(ax, 2.8, 5.0, 3.5, 5.0, NAVY, 2.2)
    arr(ax, 2.8, 4.8, 3.5, 4.0, NAVY, 2.2)
    for y in [6.45, 5.6, 4.65]:
        arr(ax, 7.3, y, 8.4, 6.1, LGRAY, 1.5)
    arr(ax, 10.8, 6.1, 11.2, 6.7, TEAL, 2.2)
    arr(ax, 10.8, 3.9, 11.2, 4.1, TEAL, 2.2)
    arr(ax, 12.2, 5.8, 12.2, 4.9, GOLD, 2.2)

    panel(ax, .4, .3, 13.2, 1.5, PANELB, '#BBCFE8')
    ax.text(W/2, 1.4, "bridge(u) = Σₖ  gate_k(u_movie) × Expert_k(u_movie)",
            ha='center', fontsize=12, fontweight='bold', color=NAVY)
    ax.text(W/2, .65, "final = w × bridge(u_movie) + (1−w) × u_game      (w learned per user)",
            ha='center', fontsize=10, color=SLATE)

    fig.tight_layout()
    fig.savefig(OUT/"ptupcdr_moe.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ ptupcdr_moe")


# =============================================================================
# 13. MODEL PTUPCDR
# =============================================================================
def fig_model_ptupcdr():
    W, H = 13, 8
    fig, ax = setup(W, H)
    hdr(ax, W, H, "PTUPCDR: Personalized Transfer via Mixture of Experts",
        "Hypernetwork generates user-specific mapping functions from movie to game space")

    bx(ax, .4, 4.8, 2.4, 2.0, "Movie MF\nModel", "e_movie(u)", NAVY, WHITE, 12, 9)
    bx(ax, 9.8, 4.8, 2.8, 2.0, "Game MF\nModel", "e_game(u)", TEAL, WHITE, 12, 9)

    panel(ax, 3.2, 3.8, 5.6, 3.4, PANELV, '#C4B0E8', label="MoE Hypernetwork", lc=VIOLET)
    ax.text(6.0, 6.7, "K Expert MLPs  +  Gate Network", ha='center',
            fontsize=10, color=VIOLET)
    ax.text(6.0, 6.2, "Personalised per user", ha='center', fontsize=9, color=LGRAY)
    bx(ax, 3.5, 4.2, 4.8, 1.8, "bridge(u) = Σ gate_k × Expert_k(e_movie)",
       bg=PANELV, fg=VIOLET, fs=10, lw=1.2, ec='#C4B0E8')

    bx(ax, 1.5, 2.0, 10.0, 1.4,
       "Blend:  w × bridge(e_movie) + (1−w) × e_game   |   w = 1/(1+k_games)",
       bg=PANELG, fg=NAVY, fs=11, lw=1.5, ec='#9BD4B8')

    arr(ax, 2.8, 5.8, 3.5, 5.2, NAVY, 2.4)
    arr(ax, 9.8, 5.8, 8.8, 5.2, TEAL, 2.4)
    arr(ax, 6.0, 3.8, 6.0, 3.4, VIOLET, 2.4)
    arr(ax, 10.2,4.8, 10.2, 3.4, TEAL, 2.2)

    fig.tight_layout()
    fig.savefig(OUT/"model_ptupcdr.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ model_ptupcdr")


# =============================================================================
# 14. SBERT CDR
# =============================================================================
def fig_sbert_cdr():
    W, H = 14, 8
    fig, ax = setup(W, H)
    hdr(ax, W, H, "SBERT-CDR: Cross-Domain Recommendation via Text Similarity",
        "Build user profile from movies, use it to rank games by text similarity")

    panel(ax, .3, 3.0, 3.8, 3.8, PANELB, '#BBCFE8', label="User's Watched Movies", lc=NAVY)
    for i, m in enumerate(["Sci-fi Movie", "Action Movie", "Thriller"]):
        bx(ax, .5, 5.4-i*1.1, 3.4, .85, m, bg=NAVY, fg=WHITE, fs=10)

    bx(ax, 5.0, 4.0, 2.8, 2.2, "Average\nVectors", "user profile", GOLD, WHITE, 12, 9)
    arr(ax, 4.1, 5.0, 5.0, 5.1, NAVY, 2.4)

    panel(ax, 8.4, 2.8, 3.8, 4.2, PANELG, '#9BD4B8', label="All Candidate Games", lc=GREEN)
    game_data = [("Sci-fi RPG", GREEN, "= 0.89"),
                 ("Sports Game", CORAL, "= 0.23"),
                 ("Space Shooter", GREEN, "= 0.85")]
    for i, (name, col, score) in enumerate(game_data):
        bx(ax, 8.6, 5.5-i*1.15, 3.4, .88, name, bg=col, fg=WHITE, fs=10)
        ax.text(12.5, 5.95-i*1.15, score, ha='left', fontsize=12,
                fontweight='bold', color=col)
        arr(ax, 7.8, 5.0, 8.6, 5.93-i*1.15, GOLD, 1.8)

    bx(ax, 12.6, 4.0, 1.2, 2.2, "Top-K\nResult", None, GREEN, WHITE, 10)
    arr(ax, 12.0, 5.0, 12.6, 5.0, GREEN, 2.5)

    panel(ax, 1.0, .3, 12.0, 1.6, PANELB, '#BBCFE8')
    ax.text(W/2, 1.5, "user_profile = mean( SBERT(movie_1), SBERT(movie_2), … )",
            ha='center', fontsize=11, fontweight='bold', color=NAVY)
    ax.text(W/2, .75, "score(game) = cosine_sim( user_profile,  SBERT(game) )",
            ha='center', fontsize=11, color=TEAL, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT/"sbert_cdr.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ sbert_cdr")


# =============================================================================
# 15. SYSTEM ARCHITECTURE
# =============================================================================
def fig_system_architecture():
    W, H = 16, 13
    fig, ax = setup(W, H)
    hdr(ax, W, H, "System Architecture",
        "Three-tier design: Next.js frontend, FastAPI backend, offline ML pipeline")

    # ── FRONTEND ──
    panel(ax, .4, 10.2, 15.2, 2.0, PANELB, '#BBCFE8', label="FRONTEND  (Next.js)", lc=NAVY)
    fe = [("Home Page", "User picker + groups"),
          ("Recommendations", "9 model rows (5G + 4M)"),
          ("Item Detail", "Similar items + rating")]
    for i, (lbl, sub) in enumerate(fe):
        bx(ax, .7+i*5.0, 10.5, 4.6, 1.5, lbl, sub, NAVY, WHITE, 12, 9)

    # ── BACKEND ──
    panel(ax, .4, 7.6, 15.2, 2.4, PANELG, '#9BD4B8', label="BACKEND  (FastAPI, port 8000)", lc=TEAL)
    be = [("User API", "/api/users"),
          ("Rec Engine", "/api/recommendations"),
          ("Item API", "/api/items/{id}"),
          ("Retrain", "/scheduler")]
    be_cols = [TEAL, TEAL, TEAL, GOLD]
    for i, ((lbl, sub), col) in enumerate(zip(be, be_cols)):
        bx(ax, .7+i*3.8, 7.9, 3.4, 1.8, lbl, sub, col, WHITE, 11, 9)
    ax.text(8.3, 10.2, "REST API", ha='center', fontsize=9,
            color=LGRAY, fontstyle='italic')
    arr(ax, 8.3, 10.2, 5.5, 9.7, LGRAY, 1.8)

    # ── IN-MEMORY STORE ──
    panel(ax, .4, 5.2, 15.2, 2.2, PANELA, '#D4C49A', label="IN-MEMORY EMBEDDING STORE", lc=GOLD)
    emb = ["LightGCN\nGames", "LightGCN\nMovies", "EMCDR", "PTUPCDR",
           "NCF", "SBERT", "Co-occ\nMatrix"]
    emb_cols = [NAVY, NAVY, TEAL, TEAL, SLATE, VIOLET, GOLD]
    ew = 1.85; gap = .23; ex0 = .55
    for i, (lbl, col) in enumerate(zip(emb, emb_cols)):
        bx(ax, ex0+i*(ew+gap), 5.5, ew, 1.6, lbl, bg=col, fg=WHITE, fs=9)
    arr(ax, 12.8, 7.6, 14.0, 7.2, GOLD, 1.8)

    # ── STORAGE ──
    panel(ax, .4, 2.0, 6.0, 3.0, PANELV, '#C4B0E8', label="STORAGE", lc=VIOLET)
    bx(ax, .6, 2.3, 2.6, 2.3, "SQLite DB", "Users + Ratings", GOLD, WHITE, 10, 9)
    bx(ax, 3.4, 2.3, 2.6, 2.3, "Artifacts", ".npz files", GOLD, WHITE, 10, 9)

    # ── ML PIPELINE ──
    panel(ax, 6.8, 2.0, 8.8, 3.0, PANELG, '#9BD4B8', label="ML PIPELINE  (Offline)", lc=GREEN)
    ml = [("Raw Data", "JSONL.gz"), ("Process", "Parquet"), ("Train", "Models"), ("Export", "Embeddings")]
    ml_cols = [NAVY, TEAL, GREEN, GOLD]
    for i, ((lbl, sub), col) in enumerate(zip(ml, ml_cols)):
        bx(ax, 7.0+i*2.1, 2.4, 1.8, 2.1, lbl, sub, col, WHITE, 9, 8)
        if i < 3:
            arr(ax, 7.0+i*2.1+1.8, 3.45, 7.0+(i+1)*2.1, 3.45, col, 1.8)

    # ── REFRESH TIERS ──
    tiers = [("Instant (~1 ms)", "Co-occurrence, SBERT: Precomputed", GREEN),
             ("Fast (~50 ms)",   "In-memory dot products", TEAL),
             ("Batch (~40 s)",   "LightGCN retrain: hourly scheduler", GOLD)]
    for i, (lbl, sub, col) in enumerate(tiers):
        tx = 1.0 + i*5.0
        circ = plt.Circle((tx-.4, .65), .18, color=col, zorder=3)
        ax.add_patch(circ)
        ax.text(tx, .82, lbl, fontsize=10, fontweight='bold', color=col)
        ax.text(tx, .38, sub, fontsize=8, color=LGRAY)

    fig.tight_layout()
    fig.savefig(OUT/"system_architecture.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ system_architecture")


# =============================================================================
# 16. RECOMMENDATION FLOW
# =============================================================================
def fig_recommendation_flow():
    W, H = 13, 10
    fig, ax = setup(W, H)
    hdr(ax, W, H, "End-to-End Recommendation Flow",
        "From user request to ranked game list in under 50ms")

    steps = [
        ("User\nRequest",       "GET /api/recs?user=42", NAVY),
        ("Profile\nLookup",     "Load game history", TEAL),
        ("Route\nDecision",     "Cold / Warm / Hot", VIOLET),
        ("Model\nInference",    "Dot product ranking", GREEN),
        ("Co-occ\nBoost",       "Add cooc scores ×λ", GOLD),
        ("Return\nTop-10",      "JSON response", GREEN),
    ]
    ys = [8.2, 6.8, 5.4, 4.0, 2.6, 1.2]
    for (lbl, sub, col), y in zip(steps, ys):
        bx(ax, 2.0, y, 9.0, 1.0, lbl, sub, col, WHITE, 13, 9, zo=4)

    for i in range(len(ys)-1):
        arr(ax, 6.5, ys[i], 6.5, ys[i+1]+1.0, LGRAY, 2.5)

    # Side notes
    notes = [
        (3, "< 1 ms",  CORAL),
        (4, "LightGCN / EMCDR / PTUPCDR\ndepending on history depth", TEAL),
        (5, "Model-agnostic, test-time only", GOLD),
    ]
    for idx, note, col in notes:
        ax.text(11.6, ys[idx]+.5, note, ha='center', va='center',
                fontsize=8, color=col,
                bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec=col, lw=1))
        arr(ax, 11.0, ys[idx]+.5, 11.0, ys[idx]+.5, col, 1.0)

    fig.tight_layout()
    fig.savefig(OUT/"recommendation_flow.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ recommendation_flow")


# =============================================================================
# 17. DATA PIPELINE
# =============================================================================
def fig_data_pipeline():
    W, H = 13, 5
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Data Processing Pipeline",
        "Raw Amazon 2023 reviews → train/val/test splits ready for model training")

    steps = [
        ("Raw JSONL\n(Amazon 2023)",   '#94A3B8', BG, True),
        ("Rating ≥ 4\n→ Implicit",     NAVY,  PANELB, False),
        ("Item K-Core\nM≥20, G≥10",    TEAL,  PANELG, False),
        ("User Filter\n(per lesson)",   GOLD,  PANELA, False),
        ("LLO Split\ntrain/val/test",   VIOLET,PANELV, False),
        ("Evaluate\n@10",               GREEN, PANELG, False),
    ]
    xs = [.4 + i*2.15 for i in range(6)]
    for (txt, col, bg, gray), x in zip(steps, xs):
        bx(ax, x, 1.6, 1.95, 2.0, txt, bg=col if not gray else '#94A3B8',
           fg=WHITE, fs=9, zo=3)
        if x != xs[-1]:
            arr(ax, x+1.95, 2.6, x+2.15, 2.6, col, 2.2)

    for i, (_, col, _, _) in enumerate(steps):
        ax.text(xs[i]+.975, 1.4, ["Amazon", "Explicit→\nImplicit",
                                    "Density\nFilter", "Cohort\nFilter",
                                    "Temporal\nSplit", "Metrics"][i],
                ha='center', va='top', fontsize=7.5, color=LGRAY)

    fig.tight_layout()
    fig.savefig(OUT/"data_pipeline.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ data_pipeline")


# =============================================================================
# 18. LESSON FLOW
# =============================================================================
def fig_lesson_flow():
    W, H = 14, 6.5
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Experiment Progression: Each Lesson Isolates One Variable",
        "Systematic ablation from single-domain baselines to advanced CDR models")

    lessons = [
        ("L1", "BPR vs\nExplicit",     "Loss fn",     LGRAY),
        ("L2", "Low overlap\n(5.8%)",   "Overlap",     NAVY),
        ("L3", "100%\noverlap",          "Users",       NAVY),
        ("L4", "Source-rich\nmovies≥10","Source",      GREEN),
        ("L5", "Catalog\nfilter",        "Items",       GOLD),
        ("L6", "Cold-start\n(0 games)", "Target",      CORAL),
        ("L7", "SBERT\ncontent",         "Signal",      VIOLET),
        ("L8", "Cooc\nrerank",           "Post-proc",   TEAL),
    ]
    xs = [.3 + i*1.7 for i in range(8)]
    for (num, desc, var, col), x in zip(lessons, xs):
        bx(ax, x, 2.4, 1.5, 2.8, f"{num}\n\n{desc}", bg=col, fg=WHITE, fs=9, zo=3)
        bx(ax, x, 1.0, 1.5, 1.1, var, bg=LGRAY, fg=WHITE, fs=8, zo=3)
        if x != xs[-1]:
            arr(ax, x+1.5, 3.8, x+1.7, 3.8, LGRAY, 1.8)

    ax.text(.3, 5.6, "Controlled ablation:", fontsize=11, fontweight='bold', color=NAVY)
    ax.text(.3, .3,
            "→ Each lesson changes exactly one variable; all others held constant",
            fontsize=9, fontstyle='italic', color=LGRAY)

    fig.tight_layout()
    fig.savefig(OUT/"lesson_flow.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ lesson_flow")


# =============================================================================
# 19. ROUTING RULE
# =============================================================================
def fig_routing_rule():
    W, H = 13, 9
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Model Routing Decision Rule",
        "Automatically select the best model based on user history depth")

    bx(ax, 4.5, 7.5, 4.0, 1.0, "New user arrives", bg=SLATE, fg=WHITE, fs=13)
    arr(ax, 6.5, 7.5, 6.5, 7.0, LGRAY, 2.2)

    bx(ax, 4.0, 5.8, 5.0, 1.0, "Has game history?", bg=VIOLET, fg=WHITE, fs=12)
    arr(ax, 6.5, 5.8, 6.5, 5.3, LGRAY, 2.2)

    # No branch
    ax.text(2.5, 5.5, "No", fontsize=11, fontweight='bold', color=CORAL, ha='center')
    arr(ax, 4.0, 6.3, 2.8, 5.0, CORAL, 2.2)
    bx(ax, .4, 3.8, 3.6, 1.0, "EMCDR + Cooc", "cold-start CDR", CORAL, WHITE, 12, 9)
    arr(ax, 2.2, 3.8, 2.2, 3.3, CORAL, 2.0)
    bx(ax, .4, 2.1, 3.6, .95, "Pure movie history\n→ game_emb via MLP", bg=PANELR, fg=CORAL,
       fs=9, lw=1.2, ec=CORAL)

    # Yes branch
    ax.text(9.8, 5.5, "Yes", fontsize=11, fontweight='bold', color=GREEN, ha='center')
    arr(ax, 9.0, 6.3, 9.8, 5.0, GREEN, 2.2)
    bx(ax, 8.0, 3.8, 4.6, 1.0, "How many games?", bg=GREEN, fg=WHITE, fs=12)

    # Sub-branches
    ax.text(7.2, 3.3, "1–2 games", fontsize=9, color=GOLD, ha='center', fontweight='bold')
    bx(ax, 5.6, 1.8, 3.6, 1.1, "PTUPCDR + Cooc", "few-shot blend", GOLD, WHITE, 11, 9)
    arr(ax, 8.0, 4.3, 7.5, 2.9, GOLD, 2.0)

    ax.text(11.5, 3.3, "3+ games", fontsize=9, color=TEAL, ha='center', fontweight='bold')
    bx(ax, 9.6, 1.8, 3.6, 1.1, "LightGCN + Cooc", "graph collaborative filtering", TEAL, WHITE, 11, 9)
    arr(ax, 10.3, 3.8, 10.5, 2.9, TEAL, 2.0)

    panel(ax, .4, .25, 12.2, 1.0, PANELV, '#C4B0E8')
    ax.text(.7, .95, "Always check:", fontsize=9, color=VIOLET, fontweight='bold')
    ax.text(.7, .5, "If item is niche (< 5 ratings)  →  fallback to SBERT-CDR + Cooc  (content-based)",
            fontsize=9, color=VIOLET)

    fig.tight_layout()
    fig.savefig(OUT/"routing_rule.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ routing_rule")


# =============================================================================
# 20. COOC MECHANISM
# =============================================================================
def fig_cooc_mechanism():
    W, H = 14, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Cross-Domain Item Co-occurrence Reranking",
        "Training-free, model-agnostic boost applied at test time only")

    bx(ax, .3, 3.5, 3.0, 2.5, "User's Movies\n\nAction Movie A\nSci-Fi Movie B\nRPG Movie C",
       bg=NAVY, fg=WHITE, fs=10)

    bx(ax, 4.4, 2.8, 3.8, 3.2, "Co-occurrence\nMatrix",
       "cooc[movie][game]\n= log(1 + #users who\n  liked both)",
       bg=GREEN, fg=WHITE, fs=11, sfs=9)

    bx(ax, 9.4, 3.5, 4.0, 2.5, "Game Score Bonus\n\nAction Game:  +0.8\nRPG Game:       +0.5\nPuzzle Game:  +0.1",
       bg=GOLD, fg=WHITE, fs=10)

    bx(ax, 5.0, .5, 7.0, 1.4,
       "Final score = Base model score  +  λ × cooc_bonus",
       bg=PANELB, fg=NAVY, fs=12, lw=1.5, ec='#BBCFE8')

    arr(ax, 3.3, 4.75, 4.4, 4.75, NAVY, 2.5)
    arr(ax, 8.2, 4.75, 9.4, 4.75, GREEN, 2.5)
    arr(ax, 11.4, 3.5, 9.5, 1.9, GOLD, 2.2)

    ax.text(3.8, 5.4, "① Look up",      fontsize=10, color=NAVY, fontweight='bold')
    ax.text(8.8, 5.4, "② Sum bonuses",  fontsize=10, color=GREEN, fontweight='bold')
    ax.text(10.2, 2.6, "③ Add to\nbase scores", fontsize=9, color=GOLD, fontweight='bold')

    ax.text(.3, .65, "Training-free  ·  Model-agnostic  ·  Test-time only",
            fontsize=10, fontstyle='italic', color=LGRAY)

    fig.tight_layout()
    fig.savefig(OUT/"cooc_mechanism.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ cooc_mechanism")


# =============================================================================
# 21. MODEL COOC
# =============================================================================
def fig_model_cooc():
    W, H = 13, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Co-occurrence Model: Score Boosting",
        "Base model scores are boosted by cross-domain co-occurrence statistics")

    bx(ax, .4, 4.0, 3.0, 2.0, "Base Model\nScores", "LightGCN / EMCDR\n/ PTUPCDR / NCF", NAVY, WHITE, 11, 9)
    bx(ax, .4, 1.5, 3.0, 2.0, "Cooc\nMatrix", "Built from training\noverlap users", GREEN, WHITE, 11, 9)
    bx(ax, 4.5, 2.8, 3.2, 1.8, "Cooc\nBoost", "+  λ × cooc_bonus", GOLD, WHITE, 12, 9)
    bx(ax, 9.0, 2.8, 3.6, 1.8, "Boosted\nScores", "Re-ranked Top-K", TEAL, WHITE, 12, 9)

    ax.text(7.0, 2.0, "+", ha='center', fontsize=32, color=GOLD, fontweight='bold', zorder=5)

    arr(ax, 3.4, 5.0, 4.5, 4.0, NAVY, 2.4)
    arr(ax, 3.4, 2.5, 4.5, 3.4, GREEN, 2.4)
    arr(ax, 7.7, 3.7, 9.0, 3.7, GOLD, 2.6)

    panel(ax, 1.5, .4, 10.0, .9, PANELB, '#BBCFE8')
    ax.text(W/2, .88, "score_final(user, game) = score_base + λ × Σ_movies  cooc[movie][game]",
            ha='center', fontsize=11, fontweight='bold', color=NAVY)

    fig.tight_layout()
    fig.savefig(OUT/"model_cooc.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ model_cooc")


# =============================================================================
# 22. EVAL PROTOCOL
# =============================================================================
def fig_eval_protocol():
    W, H = 13, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Evaluation Protocol: Leave-Last-Out + Full Ranking",
        "Standard information retrieval protocol for implicit feedback datasets")

    panel(ax, .4, 4.2, 12.2, 2.5, PANELB, '#BBCFE8', label="Data Split (per user)", lc=NAVY)
    bx(ax, .7, 4.5, 5.8, 1.9, "Train interactions\n(all but last 2)",
       "(temporal order)", NAVY, WHITE, 12, 9)
    bx(ax, 7.0, 4.5, 2.5, 1.9, "Validation\n(second-last)", "(1 item)", TEAL, WHITE, 11, 9)
    bx(ax, 9.8, 4.5, 2.5, 1.9, "Test\n(last item)", "(1 item)", GREEN, WHITE, 11, 9)

    panel(ax, .4, 1.4, 12.2, 2.5, PANELG, '#9BD4B8', label="Evaluation (Full Ranking)", lc=GREEN)
    bx(ax, .7, 1.7, 3.8, 1.9, "All game items\n(candidates)",
       "~5,000 games", SLATE, WHITE, 11, 9)
    bx(ax, 5.0, 1.7, 3.4, 1.9, "Model ranks\nall candidates",
       "score = dot(u, i)", TEAL, WHITE, 11, 9)
    bx(ax, 8.9, 1.7, 3.5, 1.9, "Metrics @10\nRecall, NDCG, HR",
       "where is the test item?", GREEN, WHITE, 11, 9)

    arr(ax, 4.5, 2.65, 5.0, 2.65, SLATE, 2.4)
    arr(ax, 8.4, 2.65, 8.9, 2.65, TEAL, 2.4)

    fig.tight_layout()
    fig.savefig(OUT/"eval_protocol.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ eval_protocol")


# =============================================================================
# 23. SBERT CONCEPT
# =============================================================================
def fig_sbert_concept():
    W, H = 13, 5.5
    fig, ax = setup(W, H)
    hdr(ax, W, H, "SBERT-CDR: Content-Based Cross-Domain Transfer",
        "Movies and games share the same semantic embedding space via Sentence-BERT")

    bx(ax, .3, 2.2, 2.4, 1.8, "Movie Titles\n& Descriptions", bg=NAVY, fg=WHITE, fs=10)
    bx(ax, 3.2, 2.2, 2.6, 1.8, "SBERT\nEncoder", "384-dim", GREEN, WHITE, 12, 9)
    bx(ax, 6.4, 2.2, 2.8, 1.8, "User Profile\n= mean of\nmovie vectors", bg=VIOLET, fg=WHITE, fs=10)
    bx(ax, 9.8, 2.2, 2.8, 1.8, "Rank Games\nby cosine sim", bg=GOLD, fg=WHITE, fs=10)

    arr(ax, 2.7, 3.1, 3.2, 3.1, NAVY, 2.4)
    arr(ax, 5.8, 3.1, 6.4, 3.1, GREEN, 2.4)
    arr(ax, 9.2, 3.1, 9.8, 3.1, VIOLET, 2.4)

    bx(ax, 3.2, .4, 2.6, 1.5, "Game Titles\n& Descriptions", bg='#C47C10', fg=WHITE, fs=10)
    arr(ax, 5.8, 1.1, 10.5, 2.2, GOLD, 2.2, cs='arc3,rad=-0.25')
    ax.text(8.0, 1.3, "Same SBERT space", fontsize=9,
            fontstyle='italic', color=LGRAY, ha='center')

    fig.tight_layout()
    fig.savefig(OUT/"sbert_concept.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ sbert_concept")


# =============================================================================
# 24. SBERT ENCODING
# =============================================================================
def fig_sbert_encoding():
    W, H = 13, 6
    fig, ax = setup(W, H)
    hdr(ax, W, H, "SBERT: Sentence Embedding Process",
        "Text is tokenised, encoded via transformer, then mean-pooled to a fixed vector")

    items = [
        ("Input Text", '"The Matrix:\nSci-fi action\nthriller"', SLATE),
        ("Tokenise", "[CLS] The\nMatrix sci-fi\n[SEP]", NAVY),
        ("Transformer\n(12 layers)", "Contextual\nrepresentations", TEAL),
        ("Mean Pool", "Average\ntoken vecs", GREEN),
        ("384-dim\nVector", "[0.31, −0.12,\n0.87, …]", GOLD),
    ]
    xs = [.3 + i*2.55 for i in range(5)]
    for (lbl, sub, col), x in zip(items, xs):
        bx(ax, x, 1.8, 2.25, 2.8, lbl, sub, col, WHITE, 11, 9)
        if x != xs[-1]:
            arr(ax, x+2.25, 3.2, x+2.55, 3.2, col, 2.2)

    fig.tight_layout()
    fig.savefig(OUT/"sbert_encoding.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ sbert_encoding")


# =============================================================================
# 25. MODEL SBERT
# =============================================================================
def fig_model_sbert():
    W, H = 13, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "SBERT-CDR: Full Inference Pipeline",
        "No training required — pure content similarity for cross-domain recommendation")

    panel(ax, .3, 1.8, 5.6, 4.2, PANELB, '#BBCFE8', label="Source (Movies)", lc=NAVY)
    bx(ax, .5, 4.5, 5.0, 1.0, "Movie titles → SBERT encoder",   bg=NAVY, fg=WHITE, fs=11)
    bx(ax, .5, 3.2, 5.0, 1.0, "Per-movie vectors (384-dim each)", bg=TEAL, fg=WHITE, fs=11)
    bx(ax, .5, 1.9, 5.0, 1.0, "User profile = mean of all movie vecs", bg=SLATE, fg=WHITE, fs=11)
    arr(ax, 3.0, 4.5, 3.0, 4.2, NAVY, 2.2)
    arr(ax, 3.0, 3.2, 3.0, 2.9, TEAL, 2.2)

    panel(ax, 7.1, 1.8, 5.6, 4.2, PANELG, '#9BD4B8', label="Target (Games)", lc=GREEN)
    bx(ax, 7.3, 4.5, 5.0, 1.0, "Game titles → SBERT encoder", bg=GREEN, fg=WHITE, fs=11)
    bx(ax, 7.3, 3.2, 5.0, 1.0, "Per-game vectors (384-dim each)", bg=TEAL, fg=WHITE, fs=11)
    bx(ax, 7.3, 1.9, 5.0, 1.0, "Rank by cosine_sim(user_profile, game_vec)", bg=GOLD, fg=WHITE, fs=11)
    arr(ax, 9.8, 4.5, 9.8, 4.2, GREEN, 2.2)
    arr(ax, 9.8, 3.2, 9.8, 2.9, TEAL, 2.2)

    arr(ax, 5.9, 2.4, 7.1, 2.4, VIOLET, 2.8)
    ax.text(6.5, 2.8, "cosine\nsimilarity", ha='center', fontsize=9,
            color=VIOLET, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT/"model_sbert.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ model_sbert")


# =============================================================================
# 26. MODEL MF-BPR
# =============================================================================
def fig_model_mfbpr():
    W, H = 11, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "MF-BPR: Matrix Factorization with BPR Loss",
        "Learn low-dimensional representations via pairwise ranking optimisation")

    bx(ax, .4, 4.4, 2.4, 1.8, "User\nEmbedding U", "N × 64 matrix", NAVY, WHITE, 12, 9)
    bx(ax, 4.3, 5.2, 2.4, 1.4, "Positive Item i⁺", "64-dim", GREEN, WHITE, 11, 9)
    bx(ax, 4.3, 3.5, 2.4, 1.4, "Negative Item i⁻", "64-dim", CORAL, WHITE, 11, 9)

    bx(ax, 7.4, 5.2, 2.4, 1.4, "Score⁺ = u·i⁺", None, GREEN+'BB', WHITE, 11)
    bx(ax, 7.4, 3.5, 2.4, 1.4, "Score⁻ = u·i⁻", None, CORAL+'BB', WHITE, 11)

    bx(ax, 4.0, 1.2, 6.2, 1.4,
       "BPR Loss = −log σ(Score⁺ − Score⁻)   →   maximise ranking gap",
       bg=PANELB, fg=NAVY, fs=11, lw=1.5, ec='#BBCFE8')

    arr(ax, 2.8, 5.3, 4.3, 5.7, NAVY, 2.2)
    arr(ax, 2.8, 5.0, 4.3, 4.2, NAVY, 2.2)
    arr(ax, 6.7, 5.6, 7.4, 5.7, GREEN, 2.2)
    arr(ax, 6.7, 4.2, 7.4, 4.1, CORAL, 2.2)
    arr(ax, 8.8, 5.2, 7.6, 2.6, GREEN, 2.0)
    arr(ax, 9.0, 3.5, 8.4, 2.6, CORAL, 2.0)

    fig.tight_layout()
    fig.savefig(OUT/"model_mfbpr.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ model_mfbpr")


# =============================================================================
# 27. MODEL NCF
# =============================================================================
def fig_model_ncf():
    W, H = 12, 8
    fig, ax = setup(W, H)
    hdr(ax, W, H, "NCF: Neural Collaborative Filtering (NeuMF)",
        "Combines GMF (linear) and MLP (non-linear) branches for richer interaction modelling")

    bx(ax, .4, 5.6, 2.2, 1.6, "User\nEmbedding", "64-dim", NAVY, WHITE, 12, 9)
    bx(ax, 9.4, 5.6, 2.2, 1.6, "Item\nEmbedding", "64-dim", TEAL, WHITE, 12, 9)

    panel(ax, .3, 3.0, 4.2, 2.3, PANELA, '#D4C49A', label="GMF Branch", lc=GOLD)
    bx(ax, .5, 3.2, 3.7, 1.8, "Element-wise Multiply\nu ⊙ i  →  64-dim", bg=GOLD, fg=WHITE, fs=11)

    panel(ax, 7.5, 3.0, 4.2, 2.3, PANELV, '#C4B0E8', label="MLP Branch", lc=VIOLET)
    bx(ax, 7.7, 3.2, 3.7, 1.8, "MLP Layers\n[u; i] → 32-dim", bg=VIOLET, fg=WHITE, fs=11)

    bx(ax, 4.0, 1.4, 4.0, 1.6, "Concat + Predict\nσ(hᵀ[GMF⊕MLP])", bg=GREEN, fg=WHITE, fs=12)

    arr(ax, 1.5, 5.6, 1.5, 5.0, NAVY, 2.2)
    arr(ax, 10.5, 5.6, 10.5, 5.0, TEAL, 2.2)
    arr(ax, 2.4, 3.0, 4.2, 2.0, GOLD, 2.4)
    arr(ax, 9.6, 3.0, 7.8, 2.0, VIOLET, 2.4)

    fig.tight_layout()
    fig.savefig(OUT/"model_ncf.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ model_ncf")


# =============================================================================
# 28. MODEL LIGHTGCN
# =============================================================================
def fig_model_lightgcn():
    W, H = 12, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "LightGCN: Light Graph Convolution Network",
        "No feature transformation — pure neighbourhood aggregation for collaborative filtering")

    bx(ax, .4, 4.2, 3.2, 2.2, "User-Item\nBipartite Graph", "Adjacency matrix A", NAVY, WHITE, 12, 9)
    bx(ax, 4.6, 5.0, 2.8, 1.4, "Layer 0\nEmbeddings", "Random init", SLATE, WHITE, 11, 9)
    bx(ax, 4.6, 3.2, 2.8, 1.4, "Layer k Aggr.",
       "e^(k) = Â·e^(k-1)", TEAL, WHITE, 11, 9)
    bx(ax, 8.4, 4.0, 3.0, 2.0, "Final Emb.\ne = (1/K)Σe^(k)",
       "K=3 layers\naverage pooling", GREEN, WHITE, 12, 9)

    arr(ax, 3.6, 5.3, 4.6, 5.5, NAVY, 2.2)
    arr(ax, 3.6, 4.6, 4.6, 3.8, NAVY, 2.2)
    arr(ax, 5.4, 4.6, 5.4, 4.6, TEAL, 2.0)
    arr(ax, 7.4, 4.2, 8.4, 4.8, TEAL, 2.2)

    panel(ax, 1.0, .4, 10.0, 1.2, PANELB, '#BBCFE8')
    ax.text(W/2, 1.1, "BPR Loss  +  Backprop  →  only Layer-0 embeddings are learnable parameters",
            ha='center', fontsize=11, fontweight='bold', color=NAVY)

    fig.tight_layout()
    fig.savefig(OUT/"model_lightgcn.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ model_lightgcn")


# =============================================================================
# 29. CMF MECHANISM
# =============================================================================
def fig_cmf_mechanism():
    W, H = 13, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "CMF: How Joint Training Works",
        "The shared user vector must simultaneously explain movie AND game interactions")

    bx(ax, .4, 4.0, 3.2, 2.5, "Movie Data\n\nUser → Movie\ninteractions\n(ratings ≥ 4)", bg=NAVY, fg=WHITE, fs=10)
    bx(ax, 4.9, 3.8, 3.2, 2.9, "Shared\nUser Emb.\n\nOne vector u\nfor both domains", bg=GOLD, fg=WHITE, fs=12, sfs=9)
    bx(ax, 9.4, 4.0, 3.2, 2.5, "Game Data\n\nUser → Game\ninteractions\n(ratings ≥ 4)", bg=TEAL, fg=WHITE, fs=10)

    bx(ax, .4, 1.4, 3.2, 1.8, "Movie Loss\nL_movie (BPR)", bg=PANELB, fg=NAVY, fs=11, lw=1.5, ec='#BBCFE8')
    bx(ax, 9.4, 1.4, 3.2, 1.8, "Game Loss\nL_game (BPR)", bg=PANELG, fg=GREEN, fs=11, lw=1.5, ec='#9BD4B8')

    bx(ax, 3.8, .35, 5.4, 1.1,
       "Joint = 0.05×L_movie + 0.95×L_game",
       bg=PANELA, fg=GOLD, fs=11, lw=1.5, ec='#D4C49A')

    arr(ax, 3.6, 5.2, 4.9, 5.2, NAVY, 2.4)
    arr(ax, 8.1, 5.2, 9.4, 5.2, GOLD, 2.4)
    arr(ax, 2.0, 4.0, 2.0, 3.2, NAVY, 2.0)
    arr(ax, 11.0, 4.0, 11.0, 3.2, TEAL, 2.0)
    arr(ax, 2.0, 1.4, 5.0, 1.2, NAVY, 1.8)
    arr(ax, 11.0, 1.4, 8.4, 1.2, TEAL, 1.8)

    fig.tight_layout()
    fig.savefig(OUT/"cmf_mechanism.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ cmf_mechanism")


# =============================================================================
# 30. COOC MATRIX
# =============================================================================
def fig_cooc_matrix():
    W, H = 11, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Co-occurrence Matrix: Construction",
        "Built once from training data — stores how often users like both a movie and a game")

    # Matrix visualisation
    movies = ["Action\nFilm", "Sci-Fi\nFilm", "RPG\nFilm"]
    games  = ["Action\nGame", "Sci-Fi\nGame", "RPG\nGame", "Puzzle\nGame"]
    vals = np.array([[0.8, 0.2, 0.4, 0.1],
                     [0.2, 0.9, 0.3, 0.2],
                     [0.3, 0.2, 0.7, 0.1]])

    for i, g in enumerate(games):
        ax.text(3.2+i*1.6, 5.6, g, ha='center', va='center',
                fontsize=9, color=NAVY, fontweight='bold')
    for j, m in enumerate(movies):
        ax.text(2.2, 4.5-j*1.1, m, ha='center', va='center',
                fontsize=9, color=TEAL, fontweight='bold')

    for j in range(3):
        for i in range(4):
            v = vals[j, i]
            col = plt.cm.YlGn(v)
            rect = FancyBboxPatch((3.0+i*1.6-0.6, 3.9-j*1.1), 1.1, .9,
                                  boxstyle="round,pad=0.05",
                                  fc=col, ec='white', lw=1.5, zorder=3)
            ax.add_patch(rect)
            ax.text(3.0+i*1.6-.05, 4.35-j*1.1, f"{v:.1f}",
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color=NAVY if v < .7 else WHITE, zorder=4)

    panel(ax, .4, .4, 10.2, 1.4, PANELB, '#BBCFE8')
    ax.text(W/2, 1.3, "cooc[movie_m][game_g] = log( 1 + |{users who liked both m and g}| )",
            ha='center', fontsize=11, fontweight='bold', color=NAVY)
    ax.text(W/2, .65, "Darker = more co-occurring  |  Built once, reused at inference",
            ha='center', fontsize=9, color=LGRAY)

    fig.tight_layout()
    fig.savefig(OUT/"cooc_matrix.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ cooc_matrix")


# =============================================================================
# 31. COOC RERANKING
# =============================================================================
def fig_cooc_reranking():
    W, H = 13, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Co-occurrence Reranking: Step by Step",
        "Apply cooc boost at test time on top of any base model's scores")

    steps = [
        ("Base model\nscores", "rank all games\nfor user u", NAVY),
        ("Lookup\ncooc bonus", "for each game,\nsum movie bonuses", GREEN),
        ("Compute\nfinal score", "base + λ × bonus", GOLD),
        ("Re-rank\nTop-10", "return best 10\ngames", TEAL),
    ]
    xs = [.4, 3.5, 6.6, 9.7]
    for (lbl, sub, col), x in zip(steps, xs):
        bx(ax, x, 2.8, 2.7, 2.8, lbl, sub, col, WHITE, 12, 9)
        if x != xs[-1]:
            arr(ax, x+2.7, 4.2, x+3.5, 4.2, col, 2.4)

    panel(ax, .4, .5, 12.2, 1.5, PANELA, '#D4C49A')
    ax.text(W/2, 1.5, "λ (lambda) controls the strength of the cooc signal",
            ha='center', fontsize=11, fontweight='bold', color=GOLD)
    ax.text(W/2, .75, "λ=0 → pure base model    λ=1 → equal weight    λ=0.3 → best empirically",
            ha='center', fontsize=9, color=SLATE)

    fig.tight_layout()
    fig.savefig(OUT/"cooc_reranking.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ cooc_reranking")


# =============================================================================
# 32. FRONTEND ARCHITECTURE
# =============================================================================
def fig_frontend_architecture():
    W, H = 13, 8
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Frontend Architecture: Next.js Application",
        "Three main views connected to FastAPI backend via REST endpoints")

    pages = [
        ("Home Page", [("User picker", TEAL), ("Group selector\n(5 users / group)", TEAL),
                       ("Switch model", SLATE)]),
        ("Recommendations", [("9 model rows", NAVY), ("5 game + 4 movie\nrows per user", NAVY),
                              ("Item cards\nwith rating", SLATE)]),
        ("Item Detail", [("Similar items", GREEN), ("Cross-domain\nlinks", GREEN),
                          ("Rating + meta", SLATE)]),
    ]
    xs = [.4, 4.6, 8.8]
    for (title_txt, items), x in zip(pages, xs):
        bx(ax, x, 5.0, 3.8, 2.2, title_txt, bg=NAVY, fg=WHITE, fs=13)
        for i, (lbl, col) in enumerate(items):
            bx(ax, x+.15, 3.4-i*1.15, 3.5, .95, lbl, bg=col, fg=WHITE, fs=9)

    panel(ax, .3, .6, 12.4, 1.8, PANELB, '#BBCFE8', label="FastAPI Backend", lc=TEAL)
    be_items = ["/api/users", "/api/recommendations", "/api/items/{id}", "/scheduler/retrain"]
    for i, ep in enumerate(be_items):
        bx(ax, .5+i*3.1, .8, 2.8, 1.2, ep, bg=TEAL, fg=WHITE, fs=9)

    for i, x in enumerate(xs):
        arr(ax, x+1.9, 5.0, 1.5+i*3.1+1.4, 2.0, LGRAY, 1.8)

    fig.tight_layout()
    fig.savefig(OUT/"frontend_architecture.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ frontend_architecture")


# =============================================================================
# 33. MODEL FAMILY STRENGTHS
# =============================================================================
def fig_model_family_strengths():
    W, H = 13, 7
    fig, ax = setup(W, H)
    hdr(ax, W, H, "Model Family Comparison",
        "Each approach makes different assumptions about user behaviour and data availability")

    families = [
        ("Single-Domain\nCF", "MF-BPR\nNCF\nLightGCN",
         "✓ Rich game history\n✗ Cold-start users", NAVY),
        ("Collective\nMF", "CMF",
         "✓ Shared user space\n✗ Domain mismatch", TEAL),
        ("Embedding\nTransfer", "EMCDR\nPTUPCDR",
         "✓ Cold-start CDR\n✓ Movie→game bridge", GREEN),
        ("Content-\nBased", "SBERT-CDR",
         "✓ No training needed\n✓ Handles niche items", VIOLET),
        ("Hybrid\nBoost", "Cooc\nReranking",
         "✓ Model-agnostic\n✓ Test-time only", GOLD),
    ]
    xs = [.3 + i*2.55 for i in range(5)]
    for (family, models, strengths, col), x in zip(families, xs):
        bx(ax, x, 4.2, 2.3, 2.0, family, bg=col, fg=WHITE, fs=11)
        bx(ax, x, 2.8, 2.3, 1.1, models, bg=LGRAY, fg=WHITE, fs=9)
        ax.text(x+1.15, 2.4, strengths, ha='center', va='top',
                fontsize=8, color=SLATE,
                bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec=col+'88', lw=1))

    ax.text(.3, 6.6, "Deployment strategy: route users dynamically based on history depth",
            fontsize=10, color=SLATE, fontstyle='italic')

    fig.tight_layout()
    fig.savefig(OUT/"model_family_strengths.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ model_family_strengths")


# =============================================================================
# 34. RESULTS SUMMARY
# =============================================================================
def fig_results_summary():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Key Results: Recall@10 Across Evaluation Regimes",
                 fontsize=14, fontweight='bold', color=NAVY, y=1.01)

    models_llo = ['LightGCN', 'MF-BPR', 'CMF', 'PTUPCDR', 'NCF', 'EMCDR']
    base_llo   = [0.0595, 0.0445, 0.0395, 0.0315, 0.0255, 0.0225]
    cooc_llo   = [0.0625, 0.0520, 0.0380, 0.0355, 0.0370, 0.0335]

    x1 = np.arange(len(models_llo)); w = 0.38
    ax1.bar(x1-w/2, base_llo, w, label='Base',  color=NAVY, alpha=.85)
    ax1.bar(x1+w/2, cooc_llo, w, label='+Cooc', color=GOLD, alpha=.85)
    ax1.set_xticks(x1); ax1.set_xticklabels(models_llo, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('Recall@10'); ax1.set_title('Standard (LLO, L3)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9); ax1.set_ylim(0, .08); ax1.spines[['top','right']].set_visible(False)

    models_cs = ['Popularity', 'EMCDR', 'PTUPCDR', 'LightGCN', 'CMF', 'MF-BPR']
    base_cs   = [0.0381, 0.0334, 0.0301, 0.0067, 0.0020, 0.0007]
    cooc_cs   = [0.0387, 0.0341, 0.0301, 0.0261, 0.0321, 0.0007]

    x2 = np.arange(len(models_cs))
    ax2.bar(x2-w/2, base_cs, w, label='Base',  color=NAVY, alpha=.85)
    ax2.bar(x2+w/2, cooc_cs, w, label='+Cooc', color=GOLD, alpha=.85)
    ax2.set_xticks(x2); ax2.set_xticklabels(models_cs, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Recall@10'); ax2.set_title('Cold-Start (L6, zero game history)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9); ax2.set_ylim(0, .05); ax2.spines[['top','right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT/"results_summary.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ results_summary")


# =============================================================================
# 35. OVERLAP IMPACT
# =============================================================================
def fig_overlap_impact():
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    models = ['LightGCN', 'EMCDR', 'PTUPCDR', 'CMF']
    l2 = [0.0290, 0.0160, 0.0085, 0.0050]
    l3 = [0.0595, 0.0225, 0.0315, 0.0395]

    x = np.arange(len(models)); w = 0.38
    ax.bar(x-w/2, l2, w, label='L2: 5.8% overlap', color=CORAL, alpha=.85)
    ax.bar(x+w/2, l3, w, label='L3: 100% overlap', color=GREEN,  alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('Recall@10', fontsize=11)
    ax.set_title('Impact of User Overlap on Performance', fontsize=13, fontweight='bold', color=NAVY)
    ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
    for i in [1, 2, 3]:
        pct = (l3[i]-l2[i])/l2[i]*100
        ax.annotate(f'+{pct:.0f}%', xy=(i+w/2, l3[i]),
                    xytext=(i+w/2, l3[i]+.003), ha='center',
                    fontsize=9, fontweight='bold', color=GREEN)
    fig.tight_layout()
    fig.savefig(OUT/"overlap_impact.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ overlap_impact")


# =============================================================================
# 36. COLDSTART COMPARISON
# =============================================================================
def fig_coldstart_comparison():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG)
    models  = ['Popularity', 'EMCDR', 'PTUPCDR', 'LightGCN', 'CMF', 'NCF', 'MF-BPR']
    recall  = [0.0381, 0.0334, 0.0301, 0.0067, 0.0020, 0.0020, 0.0007]
    colors_ = [LGRAY, TEAL, TEAL, NAVY, TEAL, NAVY, NAVY]

    bars = ax.barh(models[::-1], recall[::-1], color=colors_[::-1], alpha=.85, height=.6)
    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_title('Cold-Start Performance (Zero Game History)', fontsize=12,
                 fontweight='bold', color=NAVY)
    for bar, val in zip(bars, recall[::-1]):
        ax.text(bar.get_width()+.0005, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/"coldstart_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ coldstart_comparison")


# =============================================================================
# 37. MODEL ARCHITECTURES (6-panel overview)
# =============================================================================
def fig_model_architectures():
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle("Model Architectures Overview", fontsize=15, fontweight='bold',
                 color=NAVY, y=1.0)
    fig.patch.set_facecolor(BG)

    def sb(ax, x, y, w, h, lbl, sub=None, bg=NAVY, fg=WHITE, fs=9, sfs=7):
        sh = FancyBboxPatch((x+.04,y-.04),w,h,boxstyle="round,pad=0.1",
                            fc='#00000018',ec='none',zorder=1)
        ax.add_patch(sh)
        r = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",
                           fc=bg,ec=bg,lw=0,zorder=2)
        ax.add_patch(r)
        ty = y+h/2+(h*.1 if sub else 0)
        ax.text(x+w/2,ty,lbl,ha='center',va='center',
                fontsize=fs,fontweight='bold',color=fg,zorder=3)
        if sub:
            ax.text(x+w/2,y+h*.28,sub,ha='center',va='center',
                    fontsize=sfs,color=fg+'AA',zorder=3)

    def sa(ax,x1,y1,x2,y2,col=LGRAY,lw=1.8):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->',color=col,lw=lw,mutation_scale=16),zorder=4)

    # ── MF-BPR ──
    ax = axes[0,0]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    ax.set_title("MF-BPR  (Single-Domain)", fontsize=11, fontweight='bold', color=NAVY)
    sb(ax,1,7,3,1.5,"User Emb\n64-dim",bg=NAVY); sb(ax,6,7,3,1.5,"Item Emb\n64-dim",bg=TEAL)
    sb(ax,2.5,4.5,5,1.5,"Dot Product\nr̂ = uᵀ·i",bg=GREEN)
    sb(ax,2,1.5,6,1.5,"BPR Loss\n−log σ(r̂⁺−r̂⁻)",bg=CORAL)
    sa(ax,2.5,7,4,6,NAVY); sa(ax,7.5,7,6,6,TEAL); sa(ax,5,4.5,5,3,GREEN)

    # ── NCF ──
    ax = axes[0,1]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    ax.set_title("NCF NeuMF  (Single-Domain)", fontsize=11, fontweight='bold', color=NAVY)
    sb(ax,.5,8,2,1.2,"User",bg=NAVY); sb(ax,7.5,8,2,1.2,"Item",bg=TEAL)
    sb(ax,.3,5.5,2.5,1.2,"GMF\nu⊙i",bg=GOLD); sb(ax,7.2,5.5,2.5,1.2,"MLP\n[u;i]→h",bg=VIOLET)
    sb(ax,3,3,4,1.2,"Concat+Predict",bg=GREEN)
    sa(ax,1.5,8,1.5,6.7,NAVY); sa(ax,8.5,8,8.5,6.7,TEAL)
    sa(ax,2.8,5.5,3.5,4.2,GOLD); sa(ax,7.2,5.5,6.5,4.2,VIOLET)

    # ── LightGCN ──
    ax = axes[0,2]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    ax.set_title("LightGCN  (Single-Domain)", fontsize=11, fontweight='bold', color=NAVY)
    for i,y in enumerate([8.5,7,5.5]):
        c=plt.Circle((2,y),.4,color=NAVY,zorder=3); ax.add_patch(c)
        ax.text(2,y,f"u{i+1}",ha='center',va='center',fontsize=8,color=WHITE,fontweight='bold',zorder=4)
    for i,y in enumerate([8.5,7,5.5,4]):
        c=plt.Circle((5,y),.4,color=TEAL,zorder=3); ax.add_patch(c)
        ax.text(5,y,f"g{i+1}",ha='center',va='center',fontsize=8,color=WHITE,fontweight='bold',zorder=4)
    for uy,iy in [(8.5,8.5),(8.5,7),(7,7),(7,5.5),(5.5,5.5),(5.5,4)]:
        ax.plot([2.4,4.6],[uy,iy],'-',color='#CBD5E1',lw=1.2,zorder=1)
    sb(ax,1,1,8,2,"Graph Conv → Mean Pool\ne = (1/K)Σ e^(k)",bg=GREEN)
    sa(ax,4,4,5,3,LGRAY)

    # ── CMF ──
    ax = axes[1,0]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    ax.set_title("CMF  (Cross-Domain)", fontsize=11, fontweight='bold', color=TEAL)
    sb(ax,.3,7,3,1.5,"Movie Items",bg=NAVY); sb(ax,6.7,7,3,1.5,"Game Items",bg=TEAL)
    sb(ax,3,4.5,4,2,"Shared User\nEmbedding U",bg=GOLD)
    sb(ax,1,1,8,1.5,"Joint Loss: α·L_movie + (1−α)·L_game",bg=GREEN)
    sa(ax,1.8,7,4,6.5,NAVY); sa(ax,8.2,7,6,6.5,TEAL); sa(ax,5,4.5,5,2.5,GOLD)

    # ── EMCDR ──
    ax = axes[1,1]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    ax.set_title("EMCDR  (Cross-Domain)", fontsize=11, fontweight='bold', color=TEAL)
    sb(ax,.2,7,2.5,1.5,"Movie MF",bg=NAVY); sb(ax,7.3,7,2.5,1.5,"Game MF",bg=TEAL)
    sb(ax,3,4.5,4,1.5,"Global MLP\nmovie→game",bg=GREEN)
    sb(ax,2,1.5,6,1.5,"Inference:\nê_game=MLP(e_movie)",bg=VIOLET)
    sa(ax,1.5,7,4,6,NAVY,lw=2); sa(ax,8.5,7,6,6,TEAL)
    sa(ax,5,4.5,5,3,GREEN)

    # ── PTUPCDR ──
    ax = axes[1,2]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    ax.set_title("PTUPCDR  (Cross-Domain)", fontsize=11, fontweight='bold', color=TEAL)
    sb(ax,.2,7,2.5,1.5,"Movie MF",bg=NAVY); sb(ax,7.3,7,2.5,1.5,"Game MF",bg=TEAL)
    sb(ax,2.5,4,5,2.2,"MoE Hypernetwork\nK experts × gate",bg=VIOLET)
    sb(ax,1.5,1,7,1.5,"Blend: w·MoE + (1−w)·e_game",bg=GOLD)
    sa(ax,1.5,7,4,6.2,NAVY,lw=2); sa(ax,8.5,7,6,6.2,TEAL); sa(ax,5,4,5,2.5,VIOLET)

    fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.95)
    fig.savefig(OUT/"model_architectures.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ model_architectures")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Generating all 37 modern diagrams...")
    fig_cdr_concept()
    fig_mfbpr_training()
    fig_mfbpr_inference()
    fig_ncf_gmf()
    fig_ncf_mlp()
    fig_ncf_combined()
    fig_lightgcn_graph()
    fig_lightgcn_training()
    fig_model_cmf()
    fig_emcdr_phases()
    fig_model_emcdr()
    fig_ptupcdr_moe()
    fig_model_ptupcdr()
    fig_sbert_cdr()
    fig_system_architecture()
    fig_recommendation_flow()
    fig_data_pipeline()
    fig_lesson_flow()
    fig_routing_rule()
    fig_cooc_mechanism()
    fig_model_cooc()
    fig_eval_protocol()
    fig_sbert_concept()
    fig_sbert_encoding()
    fig_model_sbert()
    fig_model_mfbpr()
    fig_model_ncf()
    fig_model_lightgcn()
    fig_cmf_mechanism()
    fig_cooc_matrix()
    fig_cooc_reranking()
    fig_frontend_architecture()
    fig_model_family_strengths()
    fig_results_summary()
    fig_overlap_impact()
    fig_coldstart_comparison()
    fig_model_architectures()
    print(f"\nDone — {len(list(OUT.glob('*.png')))} PNGs in {OUT}/")
