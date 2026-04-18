"""Fix overlapping diagrams — regenerate only the problematic ones.

Issues fixed:
1. ncf_mlp.png — MLP Out box overlaps Layer 2
2. model_cooc.png — '+' symbol and 'Boosted Scores' overlap
3. ptupcdr_moe.png — scorer weights text overlaps Gate arrows
4. model_architectures.png — text too small, boxes crowded
5. system_architecture.png — embedding row cramped
6. recommendation_flow.png — pipeline labels tiny
7. cooc_mechanism.png — small overall
8. routing_rule.png — bottom section cramped
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)

# ── Consistent palette matching existing detailed diagrams ──
NAVY = '#1B2A4A'
TEAL = '#2E8B9A'
GOLD = '#C9942E'
DARK_TEAL = '#1A6B7A'
LIGHT_TEAL = '#E8F4F6'
LIGHT_GOLD = '#FFF8E7'
CORAL = '#E07A5F'
LIGHT_CORAL = '#FCE8E3'
GREEN_OK = '#4CAF50'
LIGHT_GREEN = '#E8F5E9'
SLATE = '#3D5A80'
PURPLE = '#7B68EE'
LIGHT_PURPLE = '#EDE7F6'
WHITE = '#FFFFFF'
LIGHT_BG = '#F7F9FB'
MID_GRAY = '#94A3B8'

# ── Matplotlib colors (hex-safe) matching existing style ──
BLUE_M = '#4472C4'
ORANGE_M = '#ED7D31'
GREEN_M = '#70AD47'
RED_M = '#FF6B6B'
GRAY_M = '#A5A5A5'
PURPLE_M = '#7B68EE'
TEAL_M = '#2EC4B6'
LIGHT_BLUE_M = '#D6E4F0'
LIGHT_ORANGE_M = '#FCE4D6'
LIGHT_GREEN_M = '#E2EFDA'
LIGHT_PURPLE_M = '#E8E0F0'


def styled_box(ax, x, y, w, h, text, bg_color=NAVY, text_color=WHITE,
               fontsize=11, bold=True, alpha=1.0, edge_color=None, lw=2,
               va='center', ha='center', pad=0.3, subtext=None, subsize=9,
               subcolor=None):
    """Draw a rounded box with optional subtitle, matching the detailed diagram style."""
    ec = edge_color or bg_color
    rect = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                           facecolor=bg_color, edgecolor=ec, linewidth=lw,
                           alpha=alpha)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ty = y + h/2 + (0.08*h if subtext else 0)
    ax.text(x + w/2, ty, text, ha=ha, va=va,
            fontsize=fontsize, fontweight=weight, color=text_color,
            zorder=10)
    if subtext:
        sc = subcolor or (text_color + '99' if len(text_color) == 7 else MID_GRAY)
        ax.text(x + w/2, y + h*0.3, subtext, ha='center', va='center',
                fontsize=subsize, color=sc, zorder=10)


def styled_arrow(ax, x1, y1, x2, y2, color='#94A3B8', lw=2, style='->',
                 connectionstyle=None):
    """Draw an arrow between two points."""
    props = dict(arrowstyle=style, color=color, lw=lw)
    if connectionstyle:
        props['connectionstyle'] = connectionstyle
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=props)


def region_box(ax, x, y, w, h, bg_color=LIGHT_BG, edge_color=None, lw=1.5, alpha=0.5):
    """Draw a background region rectangle."""
    ec = edge_color or (bg_color if bg_color != LIGHT_BG else '#E2E8F0')
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                           facecolor=bg_color, edgecolor=ec, linewidth=lw,
                           alpha=alpha, zorder=0)
    ax.add_patch(rect)


# =========================================================================
# Simple box helper for generate_diagrams.py style diagrams
# =========================================================================
def box(ax, x, y, w, h, text, color=BLUE_M, fc=None, fontsize=10, bold=False):
    """Draw a rounded box (generate_diagrams.py style)."""
    fc = fc or (color + '22')
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=fc, edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='#333333')


def arrow(ax, x1, y1, x2, y2, color='#666666', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# =========================================================================
# 1. FIX: NCF MLP — MLP Out overlaps Layer 2
# =========================================================================
def fix_ncf_mlp():
    """Widen figure, add gap between Layer 2 and MLP Out."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off')

    ax.text(6, 4.7, "NCF — Branch 2: MLP (Multi-Layer Perceptron)",
            ha='center', fontsize=16, fontweight='bold', color=NAVY)
    ax.text(6, 4.35, "Neural network layers learn non-linear interaction patterns",
            ha='center', fontsize=11, color=MID_GRAY)

    # Background region for the MLP pipeline
    region_box(ax, 1.8, 1.1, 9.5, 2.8, LIGHT_BG, '#E2E8F0')

    # User Emb
    styled_box(ax, 0.2, 2.6, 1.6, 1.2, "User\nEmb.", NAVY, WHITE, 12)
    ax.text(1.0, 2.35, "64-dim", ha='center', fontsize=9, color=SLATE)

    # Item Emb
    styled_box(ax, 0.2, 1.1, 1.6, 1.2, "Item\nEmb.", SLATE, WHITE, 12)
    ax.text(1.0, 0.85, "64-dim", ha='center', fontsize=9, color=SLATE)

    # Concat
    styled_box(ax, 2.3, 1.7, 1.8, 1.6, "Concat", TEAL, WHITE, 12)
    ax.text(3.2, 1.5, "128-dim", ha='center', fontsize=9, color=TEAL)

    # Layer 1
    styled_box(ax, 4.5, 1.7, 1.8, 1.6, "Layer 1", NAVY, WHITE, 12)
    ax.text(5.4, 1.5, "128 neurons", ha='center', fontsize=9, color=SLATE)

    # Layer 2
    styled_box(ax, 6.7, 1.7, 1.8, 1.6, "Layer 2", NAVY, WHITE, 12)
    ax.text(7.6, 1.5, "64 neurons", ha='center', fontsize=9, color=SLATE)

    # MLP Out — MOVED further right with clear gap
    styled_box(ax, 9.1, 1.7, 1.8, 1.6, "MLP\nOut", GREEN_OK, WHITE, 12)
    ax.text(10.0, 1.5, "32-dim", ha='center', fontsize=9, color=GREEN_OK)

    # Arrows
    styled_arrow(ax, 1.8, 3.2, 2.3, 2.8, NAVY)
    styled_arrow(ax, 1.8, 1.7, 2.3, 2.2, SLATE)
    styled_arrow(ax, 4.1, 2.5, 4.5, 2.5, TEAL)
    styled_arrow(ax, 6.3, 2.5, 6.7, 2.5, NAVY)
    styled_arrow(ax, 8.5, 2.5, 9.1, 2.5, NAVY)

    # Formula
    region_box(ax, 1.5, 0.05, 9.0, 0.55, LIGHT_BG, '#E2E8F0')
    ax.text(6.0, 0.32, "Each layer applies:  output = ReLU( W * input + b )  —  learns complex patterns",
            ha='center', fontsize=11, color=NAVY, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT / "ncf_mlp.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ ncf_mlp.png")


# =========================================================================
# 2. FIX: Model Cooc — '+' overlaps Boosted Scores
# =========================================================================
def fix_model_cooc():
    """Widen layout, separate '+' from Boosted Scores."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')

    ax.text(6, 5.7, "Co-occurrence Reranking",
            ha='center', fontsize=16, fontweight='bold', color=NAVY)
    ax.text(6, 5.3, "Boost game scores based on what movie fans also played",
            ha='center', fontsize=11, color=MID_GRAY)

    # User's Movies
    styled_box(ax, 0.3, 2.8, 2.0, 2.0, "User's\nMovies", NAVY, WHITE, 13)
    ax.text(1.3, 2.55, "(watched list)", ha='center', fontsize=9, color=MID_GRAY)

    # Co-occurrence Matrix — table-like
    region_box(ax, 2.8, 2.2, 3.8, 3.0, LIGHT_TEAL, TEAL, alpha=0.3)
    ax.text(4.7, 4.95, "Co-occurrence Matrix", ha='center', fontsize=12,
            fontweight='bold', color=TEAL)
    # Column headers
    for j, game in enumerate(["Game 1", "Game 2", "Game 3"]):
        ax.text(3.5 + j*1.1, 4.5, game, ha='center', fontsize=9, color=TEAL, fontweight='bold')
    # Rows
    for i, movie in enumerate(["Movie A", "Movie B"]):
        y_row = 3.8 - i * 0.9
        ax.text(3.0, y_row, movie, ha='center', fontsize=9, color=NAVY, fontweight='bold')
        for j, val in enumerate([(0.8, GOLD), (0.2, LIGHT_GREEN), (0.5, TEAL)][::1] if i == 0 else [(0.1, LIGHT_GREEN), (0.7, GOLD), (0.3, TEAL)]):
            v, c = val
            styled_box(ax, 3.2 + j*1.1, y_row - 0.25, 0.7, 0.5, f"{v}", c, WHITE, 10, pad=0.1, lw=1)

    # Arrow from movies to matrix
    styled_arrow(ax, 2.3, 3.8, 2.8, 3.8, NAVY, lw=2.5)

    # Base Model Scores
    styled_box(ax, 7.2, 3.5, 2.0, 1.5, "Base Model\nScores", SLATE, WHITE, 12)
    ax.text(8.2, 3.25, "(any model)", ha='center', fontsize=9, color=MID_GRAY)

    # Plus sign — clearly separated
    ax.text(9.6, 3.3, "+", fontsize=28, fontweight='bold', color=GOLD, ha='center', va='center')

    # Arrow from matrix to plus
    styled_arrow(ax, 6.6, 3.5, 7.2, 3.8, TEAL, lw=2)

    # Boosted Scores — moved further right
    styled_box(ax, 7.2, 1.0, 2.2, 1.5, "Boosted\nScores", TEAL, WHITE, 13)
    ax.text(8.3, 0.75, "reranked", ha='center', fontsize=9, color=TEAL)

    # Arrows
    styled_arrow(ax, 8.2, 3.5, 8.3, 2.5, GOLD, lw=2)
    styled_arrow(ax, 9.6, 3.0, 9.0, 2.5, GOLD, lw=2, connectionstyle='arc3,rad=0.2')

    # Final Ranking
    styled_box(ax, 10.0, 1.0, 1.6, 1.5, "Final\nRanking", GREEN_OK, WHITE, 12)
    ax.text(10.8, 0.75, "Top-K items", ha='center', fontsize=9, color=GREEN_OK)
    styled_arrow(ax, 9.4, 1.75, 10.0, 1.75, TEAL, lw=2)

    # Formula
    region_box(ax, 1.0, -0.2, 10.0, 0.8, LIGHT_BG, '#E2E8F0')
    ax.text(6.0, 0.35, "final_score(game) = model_score(game) + lambda * SUM cooc(movie_j, game)",
            ha='center', fontsize=11, color=NAVY, fontweight='bold', family='monospace')
    ax.text(6.0, -0.0, "Training-free  |  Model-agnostic  |  Applied as post-processing",
            ha='center', fontsize=9, color=MID_GRAY, fontstyle='italic')

    fig.tight_layout()
    fig.savefig(OUT / "model_cooc.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ model_cooc.png")


# =========================================================================
# 3. FIX: PTUPCDR MoE — scorer weights overlaps Gate arrows
# =========================================================================
def fix_ptupcdr_moe():
    """Increase figure height, move text labels out of arrow paths."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis('off')

    ax.text(6.5, 6.7, "PTUPCDR: Personalized Transfer via Mixture of Experts",
            ha='center', fontsize=16, fontweight='bold', color=NAVY)
    ax.text(6.5, 6.3, "Each user gets a custom translator built by blending 8 expert networks",
            ha='center', fontsize=11, color=MID_GRAY)

    # Expert networks region
    region_box(ax, 2.5, 2.0, 3.5, 3.8, LIGHT_BG, SLATE, alpha=0.3)
    ax.text(4.25, 5.6, "8 Expert Networks", ha='center', fontsize=13,
            fontweight='bold', color=SLATE)
    ax.text(4.25, 5.25, "Each expert is a small MLP that maps movie emb to game space",
            ha='center', fontsize=8, color=MID_GRAY)

    # User's Movie Emb
    styled_box(ax, 0.2, 3.2, 2.0, 1.8, "User's\nMovie Emb.", NAVY, WHITE, 12)
    ax.text(1.2, 2.95, "from Phase 1", ha='center', fontsize=9, color=MID_GRAY)

    # Expert boxes
    experts_y = [4.6, 3.8, 3.0]
    for i, y in enumerate(experts_y):
        styled_box(ax, 3.0, y, 2.5, 0.6, f"Expert {i+1}", PURPLE, WHITE, 10, pad=0.15)
        styled_arrow(ax, 2.2, 4.1, 3.0, y + 0.3, NAVY, lw=1.2)
    ax.text(4.25, 2.55, "...", fontsize=18, ha='center', color=SLATE, fontweight='bold')
    styled_box(ax, 3.0, 2.0, 2.5, 0.6, "Expert 8", PURPLE, WHITE, 10, pad=0.15)
    styled_arrow(ax, 2.2, 4.1, 3.0, 2.3, NAVY, lw=1.2)

    # Gate Network — moved right with clear space
    styled_box(ax, 6.5, 3.5, 2.0, 1.6, "Gate\nNetwork", TEAL, WHITE, 13)
    ax.text(7.5, 3.25, "user-specific", ha='center', fontsize=9, color=TEAL)

    # Arrows from experts to gate — moved label below
    styled_arrow(ax, 5.5, 4.0, 6.5, 4.3, PURPLE, lw=1.5)
    styled_arrow(ax, 5.5, 3.3, 6.5, 3.8, PURPLE, lw=1.5)
    ax.text(6.2, 4.7, "Scorer weights\nfor each expert", ha='center', fontsize=9,
            color=PURPLE, fontstyle='italic')

    # Mapped Game Emb
    styled_box(ax, 9.0, 3.5, 2.0, 1.6, "Mapped\nGame Emb.", GOLD, WHITE, 12)
    styled_arrow(ax, 8.5, 4.3, 9.0, 4.3, TEAL, lw=2.5)

    # Actual Game Emb
    styled_box(ax, 6.5, 1.0, 2.0, 1.6, "Actual Game\nEmb.", TEAL, WHITE, 12)
    ax.text(7.5, 0.75, "(if user has games)", ha='center', fontsize=9, color=MID_GRAY)

    # Blend box
    styled_box(ax, 9.5, 1.0, 2.2, 1.6, "Blend", ORANGE_M, WHITE, 14)
    ax.text(10.6, 0.75, "w*mapped\n+(1-w)*game", ha='center', fontsize=9, color=ORANGE_M)

    styled_arrow(ax, 10.0, 3.5, 10.6, 2.6, GOLD, lw=2)
    styled_arrow(ax, 8.5, 1.8, 9.5, 1.8, TEAL, lw=2)

    # Formula region
    region_box(ax, 1.0, -0.5, 11.0, 1.1, LIGHT_BG, '#E2E8F0')
    ax.text(6.5, 0.2, "bridge(u) = sum_k  gate_k(u_movie) * Expert_k(u_movie)",
            ha='center', fontsize=11, color=NAVY, fontweight='bold', family='monospace')
    ax.text(6.5, -0.15, "final = w * bridge(u_movie) + (1-w) * u_game    (w is learned per user)",
            ha='center', fontsize=10, color=SLATE, family='monospace')

    fig.tight_layout()
    fig.savefig(OUT / "ptupcdr_moe.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ ptupcdr_moe.png")


# =========================================================================
# 4. FIX: Model Architectures — larger figure, bigger text
# =========================================================================
def fix_model_architectures():
    """Increase to 16x11, bump all fontsize, widen subplot spacing."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle("Model Architectures Overview", fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.35, hspace=0.45)

    # --- MF-BPR ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("MF-BPR\n(Single-Domain)", fontsize=13, fontweight='bold', color=GRAY_M)
    box(ax, 0.5, 7, 3.5, 1.5, "User\nEmbedding", BLUE_M, LIGHT_BLUE_M, 11)
    box(ax, 6, 7, 3.5, 1.5, "Item\nEmbedding", ORANGE_M, LIGHT_ORANGE_M, 11)
    box(ax, 2.5, 4, 5, 1.5, "Dot Product\nr̂ = uᵀ · i", GREEN_M, LIGHT_GREEN_M, 11)
    box(ax, 2, 1, 6, 1.5, "BPR Loss\n−log σ(r̂⁺ − r̂⁻)", RED_M, '#FFE8E8', 11)
    arrow(ax, 2.5, 7, 4, 5.5)
    arrow(ax, 7.5, 7, 6.5, 5.5)
    arrow(ax, 5, 4, 5, 2.5)

    # --- NCF ---
    ax = axes[0, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("NCF (NeuMF)\n(Single-Domain)", fontsize=13, fontweight='bold', color=GRAY_M)
    box(ax, 0.3, 7.5, 2.5, 1.2, "User", BLUE_M, LIGHT_BLUE_M, 11)
    box(ax, 7.2, 7.5, 2.5, 1.2, "Item", ORANGE_M, LIGHT_ORANGE_M, 11)
    box(ax, 0, 5, 3, 1.2, "GMF\nu ⊙ i", BLUE_M, LIGHT_BLUE_M, 10)
    box(ax, 7, 5, 3, 1.2, "MLP\n[u;i]→h", PURPLE_M, LIGHT_PURPLE_M, 10)
    box(ax, 2.5, 2, 5, 1.5, "Concat + Predict\nσ(hᵀ[GMF⊕MLP])", GREEN_M, LIGHT_GREEN_M, 10)
    arrow(ax, 1.5, 7.5, 1.5, 6.2); arrow(ax, 8.5, 7.5, 8.5, 6.2)
    arrow(ax, 3.0, 5.5, 4.0, 3.5); arrow(ax, 7.0, 5.5, 6.5, 3.5)

    # --- LightGCN ---
    ax = axes[0, 2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("LightGCN\n(Single-Domain)", fontsize=13, fontweight='bold', color=GRAY_M)
    np.random.seed(42)
    for i, y in enumerate([8.5, 7, 5.5]):
        ax.plot(2, y, 'o', color=BLUE_M, markersize=18)
        ax.text(0.5, y, f"u{i+1}", ha='center', va='center', fontsize=10, fontweight='bold')
    for i, y in enumerate([8.5, 7, 5.5, 4]):
        ax.plot(5.5, y, 's', color=ORANGE_M, markersize=16)
        ax.text(7.0, y, f"g{i+1}", ha='center', va='center', fontsize=10, fontweight='bold')
    for uy in [8.5, 7, 5.5]:
        for iy in [8.5, 7, 5.5, 4]:
            if np.random.random() > 0.5:
                ax.plot([2, 5.5], [uy, iy], '-', color='#CCCCCC', lw=1.0)
    box(ax, 0.5, 1, 9, 2, "Graph Conv → Mean Pool\ne_u = (1/K) Σ e_u^(k)", GREEN_M, LIGHT_GREEN_M, 11)
    arrow(ax, 3.5, 4, 5, 3)

    # --- CMF ---
    ax = axes[1, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("CMF\n(Cross-Domain)", fontsize=13, fontweight='bold', color=TEAL_M)
    box(ax, 0, 7, 3.5, 1.5, "Movie Items\nV_movie", BLUE_M, LIGHT_BLUE_M, 11)
    box(ax, 6.5, 7, 3.5, 1.5, "Game Items\nV_game", ORANGE_M, LIGHT_ORANGE_M, 11)
    box(ax, 2.5, 4, 5, 2.2, "Shared User\nEmbedding U\n(one vector for\nboth domains)", PURPLE_M, LIGHT_PURPLE_M, 10, True)
    box(ax, 0.5, 1, 9, 1.5, "Joint Loss: α·L_movie + (1−α)·L_game", GREEN_M, LIGHT_GREEN_M, 11)
    arrow(ax, 1.8, 7, 4, 6.2, BLUE_M); arrow(ax, 8.2, 7, 6.5, 6.2, ORANGE_M)
    arrow(ax, 5, 4, 5, 2.5)

    # --- EMCDR ---
    ax = axes[1, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("EMCDR\n(Cross-Domain)", fontsize=13, fontweight='bold', color=TEAL_M)
    box(ax, 0, 7, 3, 1.5, "Movie MF\ne_movie(u)", BLUE_M, LIGHT_BLUE_M, 11)
    box(ax, 7, 7, 3, 1.5, "Game MF\ne_game(u)", ORANGE_M, LIGHT_ORANGE_M, 11)
    box(ax, 2.5, 4, 5, 1.8, "Global MLP\nf: movie → game", GREEN_M, LIGHT_GREEN_M, 12, True)
    box(ax, 1.5, 1, 7, 1.5, "Predict: ê_game(u) = MLP(e_movie(u))", PURPLE_M, LIGHT_PURPLE_M, 10)
    arrow(ax, 1.5, 7, 4, 5.8, BLUE_M, lw=2)
    arrow(ax, 8.5, 7, 6.5, 5.8, ORANGE_M)
    arrow(ax, 5, 4, 5, 2.5)
    ax.text(1.2, 9.3, "① Train", fontsize=10, color=BLUE_M, fontweight='bold')
    ax.text(8.0, 9.3, "② Train", fontsize=10, color=ORANGE_M, fontweight='bold')
    ax.text(4.5, 6.1, "③ Map", fontsize=10, color=GREEN_M, fontweight='bold')

    # --- PTUPCDR ---
    ax = axes[1, 2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("PTUPCDR\n(Cross-Domain)", fontsize=13, fontweight='bold', color=TEAL_M)
    box(ax, 0, 7, 3, 1.5, "Movie MF\ne_movie(u)", BLUE_M, LIGHT_BLUE_M, 11)
    box(ax, 7, 7, 3, 1.5, "Game MF\ne_game(u)", ORANGE_M, LIGHT_ORANGE_M, 11)
    box(ax, 2, 3.5, 6, 2.5, "MoE Hypernetwork\nExpert₁  Expert₂  ...  Expert_k\n\nPersonalized per user", GREEN_M, LIGHT_GREEN_M, 10, True)
    box(ax, 1, 0.5, 8, 1.8, "Blend: w·MoE(e_movie) + (1−w)·e_game\nw = 1/(1+k), k = #game interactions", PURPLE_M, LIGHT_PURPLE_M, 9)
    arrow(ax, 1.5, 7, 4, 6, BLUE_M, lw=2)
    arrow(ax, 8.5, 7, 7, 6, ORANGE_M)
    arrow(ax, 5, 3.5, 5, 2.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "model_architectures.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ model_architectures.png")


# =========================================================================
# 5. FIX: System Architecture — larger, less cramped
# =========================================================================
def fix_system_architecture():
    """Increase figure size, spread out embedding row."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')

    ax.text(7, 9.7, "System Architecture",
            ha='center', fontsize=18, fontweight='bold', color=NAVY)
    ax.text(7, 9.3, "Three-tier design: Next.js frontend, FastAPI backend, offline ML pipeline",
            ha='center', fontsize=11, color=MID_GRAY)

    # ── FRONTEND ──
    region_box(ax, 0.5, 7.5, 13.0, 1.7, '#E8F0FE', NAVY, alpha=0.3)
    ax.text(1.0, 9.0, "FRONTEND  (Next.js 14)", fontsize=10, fontweight='bold', color=NAVY)

    styled_box(ax, 1.0, 7.7, 3.5, 1.2, "Home Page", NAVY, WHITE, 12)
    ax.text(2.75, 7.5, "User picker + groups", ha='center', fontsize=9, color=MID_GRAY)

    styled_box(ax, 5.0, 7.7, 3.5, 1.2, "Recommendations", TEAL, WHITE, 12)
    ax.text(6.75, 7.5, "9 model rows (5G + 4M)", ha='center', fontsize=9, color=MID_GRAY)

    styled_box(ax, 9.5, 7.7, 3.5, 1.2, "Item Detail", SLATE, WHITE, 12)
    ax.text(11.25, 7.5, "Similar items + rating", ha='center', fontsize=9, color=MID_GRAY)

    # ── BACKEND ──
    region_box(ax, 0.5, 5.2, 13.0, 2.1, LIGHT_TEAL, TEAL, alpha=0.2)
    ax.text(1.0, 7.1, "BACKEND  (FastAPI, port 8000)", fontsize=10, fontweight='bold', color=TEAL)

    # REST API label
    ax.text(7, 6.95, "REST API ENDPOINTS", ha='center', fontsize=9, fontweight='bold', color=TEAL)

    styled_box(ax, 1.0, 5.9, 2.8, 0.9, "User API", TEAL, WHITE, 11)
    ax.text(2.4, 5.7, "/api/users", ha='center', fontsize=8, color=MID_GRAY)

    styled_box(ax, 4.2, 5.9, 2.8, 0.9, "Rec Engine", DARK_TEAL, WHITE, 11)
    ax.text(5.6, 5.7, "/api/recommendations", ha='center', fontsize=8, color=MID_GRAY)

    styled_box(ax, 7.4, 5.9, 2.8, 0.9, "Item API", TEAL, WHITE, 11)
    ax.text(8.8, 5.7, "/api/items/{id}", ha='center', fontsize=8, color=MID_GRAY)

    styled_box(ax, 10.6, 5.9, 2.5, 0.9, "Retrain", GOLD, WHITE, 11)
    ax.text(11.85, 5.7, "Scheduler", ha='center', fontsize=8, color=MID_GRAY)

    # ── IN-MEMORY EMBEDDING STORE ──
    ax.text(7, 5.35, "IN-MEMORY EMBEDDING STORE", ha='center', fontsize=10,
            fontweight='bold', color=SLATE)

    # 7 model boxes — spread across full width
    models = ["LightGCN\nGames", "LightGCN\nMovies", "EMCDR", "PTUPCDR", "NCF", "SBERT", "Co-occ\nMatrix"]
    model_colors = [NAVY, NAVY, TEAL, TEAL, SLATE, PURPLE, GOLD]
    model_x = 0.8
    model_w = 1.65
    model_gap = 0.15
    for i, (name, color) in enumerate(zip(models, model_colors)):
        x = model_x + i * (model_w + model_gap)
        styled_box(ax, x, 4.4, model_w, 0.8, name, color, WHITE, 9, pad=0.15)

    # Arrows frontend → backend
    styled_arrow(ax, 6.75, 7.7, 5.6, 6.8, NAVY, lw=1.5)

    # ── STORAGE ──
    region_box(ax, 0.5, 1.5, 5.5, 2.5, LIGHT_GOLD, GOLD, alpha=0.2)
    ax.text(1.0, 3.8, "STORAGE", fontsize=10, fontweight='bold', color=GOLD)

    styled_box(ax, 1.0, 2.2, 2.2, 1.3, "SQLite DB", GOLD, WHITE, 11)
    ax.text(2.1, 2.0, "Users + Ratings", ha='center', fontsize=9, color=MID_GRAY)

    styled_box(ax, 3.5, 2.2, 2.2, 1.3, "Artifacts", GOLD, WHITE, 11)
    ax.text(4.6, 2.0, "*.npy files", ha='center', fontsize=9, color=MID_GRAY)

    # ── ML PIPELINE ──
    region_box(ax, 7.0, 1.5, 6.5, 2.5, LIGHT_TEAL, TEAL, alpha=0.2)
    ax.text(7.5, 3.8, "ML PIPELINE  (Offline)", fontsize=10, fontweight='bold', color=TEAL)

    pipe_steps = ["Raw Data", "Process", "Train", "Export"]
    pipe_sub = ["JSONL.gz", "Parquet", "Models", "Embeddings"]
    for i, (step, sub) in enumerate(zip(pipe_steps, pipe_sub)):
        x = 7.5 + i * 1.55
        styled_box(ax, x, 2.2, 1.3, 1.0, step, TEAL, WHITE, 10, pad=0.15)
        ax.text(x + 0.65, 2.0, sub, ha='center', fontsize=8, color=MID_GRAY)
        if i < len(pipe_steps) - 1:
            styled_arrow(ax, x + 1.3, 2.7, x + 1.55, 2.7, TEAL, lw=1.2)

    # Arrow from pipeline to storage
    styled_arrow(ax, 12.5, 2.7, 13.2, 4.8, TEAL, lw=1.5, connectionstyle='arc3,rad=0.3')

    # ── Refresh Tiers Legend ──
    ax.text(7, 1.1, "Refresh Tiers", ha='center', fontsize=11, fontweight='bold', color=NAVY)
    for i, (label, color, desc) in enumerate([
        ("Instant (~1 ms)", GREEN_OK, "Co-occurrence, SBERT: Precomputed"),
        ("Fast (~50 ms)", TEAL, "In-memory dot products/recommendations"),
        ("Batch (~40 s)", GOLD, "LightGCN retrain: hourly scheduler"),
    ]):
        x = 1.5 + i * 4.2
        ax.plot(x, 0.6, 'o', color=color, markersize=10)
        ax.text(x + 0.3, 0.6, label, fontsize=10, fontweight='bold', color=color, va='center')
        ax.text(x + 0.3, 0.25, desc, fontsize=8, color=MID_GRAY, va='center')

    fig.tight_layout()
    fig.savefig(OUT / "system_architecture.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ system_architecture.png")


# =========================================================================
# 6. FIX: Recommendation Flow — larger pipeline labels
# =========================================================================
def fix_recommendation_flow():
    """Increase figure size, make pipeline text readable."""
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13); ax.set_ylim(0, 9); ax.axis('off')

    ax.text(6.5, 8.7, "Recommendation Flow",
            ha='center', fontsize=18, fontweight='bold', color=NAVY)
    ax.text(6.5, 8.3, "Per-user request pipeline: 9 model rows with domain-specific routing",
            ha='center', fontsize=11, color=MID_GRAY)

    # API endpoint
    styled_box(ax, 2.5, 7.2, 8.0, 0.8, "GET  /api/recommendations/{user_id}", TEAL, WHITE, 13,
               pad=0.2)

    # Resolve step
    styled_arrow(ax, 6.5, 7.2, 6.5, 6.8, TEAL, lw=2)
    region_box(ax, 2.5, 6.1, 8.0, 0.65, LIGHT_BG, '#E2E8F0')
    ax.text(6.5, 6.42, "Resolve user → external_id → model indices + rated items",
            ha='center', fontsize=11, color=NAVY)

    # Generate rows label
    ax.text(6.5, 5.75, "Generate 9 recommendation rows in parallel",
            ha='center', fontsize=12, fontweight='bold', color=NAVY)

    # Game Rows (left)
    region_box(ax, 0.5, 3.3, 5.5, 2.3, '#E8F0FE', NAVY, alpha=0.3)
    ax.text(3.25, 5.35, "Game Rows (5)", ha='center', fontsize=12, fontweight='bold', color=TEAL)

    game_rows = [
        ("Top Picks for You", "LightGCN + Co-occ"),
        ("Based on Movie Taste", "EMCDR / PTUPCDR"),
        ("Players Also Played", "Co-occurrence"),
        ("Similar Theme", "SBERT content"),
        ("Trending Games", "Popularity"),
    ]
    for i, (name, model) in enumerate(game_rows):
        y = 4.9 - i * 0.38
        styled_box(ax, 0.7, y, 3.0, 0.32, name, NAVY, WHITE, 9, pad=0.08)
        ax.text(5.6, y + 0.16, model, ha='right', fontsize=9, color=SLATE, fontstyle='italic')

    # Movie Rows (right)
    region_box(ax, 7.0, 3.3, 5.5, 2.3, LIGHT_TEAL, TEAL, alpha=0.2)
    ax.text(9.75, 5.35, "Movie Rows (4)", ha='center', fontsize=12, fontweight='bold', color=TEAL)

    movie_rows = [
        ("Top Movie Picks", "LightGCN-Movies + Rev. Co-occ"),
        ("Fans Also Watched", "Reverse co-occurrence"),
        ("Movies You'll Enjoy", "SBERT content"),
        ("Trending Movies", "Popularity"),
    ]
    for i, (name, model) in enumerate(movie_rows):
        y = 4.9 - i * 0.38
        styled_box(ax, 7.2, y, 3.0, 0.32, name, TEAL, WHITE, 9, pad=0.08)
        ax.text(12.2, y + 0.16, model, ha='right', fontsize=9, color=SLATE, fontstyle='italic')

    # Pipeline arrows
    styled_arrow(ax, 3.25, 3.3, 6.5, 2.8, NAVY, lw=2)
    styled_arrow(ax, 9.75, 3.3, 6.5, 2.8, TEAL, lw=2)

    # Per-Row Processing Pipeline
    region_box(ax, 0.5, 0.3, 12.0, 2.3, LIGHT_BG, '#E2E8F0', alpha=0.5)
    ax.text(6.5, 2.35, "Per-Row Processing Pipeline", ha='center', fontsize=13,
            fontweight='bold', color=NAVY, fontstyle='italic')

    # Top row
    pipe_top = ["Score items", "Add co-occ bonus", "Domain mask", "Normalize [0,1]"]
    for i, step in enumerate(pipe_top):
        x = 1.0 + i * 3.0
        styled_box(ax, x, 1.5, 2.5, 0.6, step, TEAL, WHITE, 10, pad=0.12)
        if i < len(pipe_top) - 1:
            styled_arrow(ax, x + 2.5, 1.8, x + 3.0, 1.8, MID_GRAY, lw=1.5)

    # Bottom row
    pipe_bot = ["Exclude rated", "Top-K (25)", "Cross-row dedup", "Enrich metadata"]
    for i, step in enumerate(pipe_bot):
        x = 1.0 + i * 3.0
        styled_box(ax, x, 0.5, 2.5, 0.6, step, DARK_TEAL, WHITE, 10, pad=0.12)
        if i < len(pipe_bot) - 1:
            styled_arrow(ax, x + 2.5, 0.8, x + 3.0, 0.8, MID_GRAY, lw=1.5)

    # Connect top row to bottom row
    styled_arrow(ax, 11.5, 1.5, 11.5, 1.1, MID_GRAY, lw=1.5)
    styled_arrow(ax, 11.5, 1.3, 10.5, 1.1, MID_GRAY, lw=1.5, connectionstyle='arc3,rad=0.3')

    fig.tight_layout()
    fig.savefig(OUT / "recommendation_flow.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ recommendation_flow.png")


# =========================================================================
# 7. FIX: Cooc Mechanism (from generate_diagrams.py) — larger
# =========================================================================
def fix_cooc_mechanism():
    """Increase figure size and font sizes."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(6.5, 5.7, "Cross-Domain Item Co-occurrence Reranking",
            fontsize=15, fontweight='bold', ha='center', color=NAVY)

    # User's movie history
    styled_box(ax, 0.3, 2.8, 2.8, 2.2, "User's Movies", NAVY, WHITE, 13)
    for i, m in enumerate(["Action Movie A", "Sci-Fi Movie B", "RPG Movie C"]):
        ax.text(1.7, 4.2 - i*0.5, m, ha='center', fontsize=9, color='#CCDDEE')

    # Cooc matrix
    region_box(ax, 3.8, 2.2, 3.8, 3.0, LIGHT_TEAL, TEAL, alpha=0.3)
    ax.text(5.7, 5.0, "Co-occurrence Matrix", ha='center', fontsize=13,
            fontweight='bold', color=TEAL)
    ax.text(5.7, 4.55, "cooc[movie][game]", ha='center', fontsize=10, color=TEAL,
            family='monospace')
    ax.text(5.7, 4.15, "= log(1 + count of users\nwho liked both)", ha='center',
            fontsize=10, color=SLATE)

    # Bonus scores
    styled_box(ax, 8.3, 3.0, 4.0, 2.2, "Game Score Bonus", GOLD, WHITE, 13)
    for i, (g, v) in enumerate([("Action Game:", "+0.8"), ("RPG Game:", "+0.5"), ("Puzzle Game:", "+0.1")]):
        ax.text(10.3, 4.35 - i*0.5, f"{g}  {v}", ha='center', fontsize=10, color='#FFF8E7')

    # Base model
    styled_box(ax, 8.3, 0.3, 4.0, 2.0, "Base Model Score", SLATE, WHITE, 13)
    ax.text(10.3, 1.0, "(LightGCN, EMCDR, ...)\n+ λ × bonus", ha='center',
            fontsize=10, color='#CCDDEE')

    # Arrows
    styled_arrow(ax, 3.1, 3.9, 3.8, 3.9, NAVY, lw=2.5)
    styled_arrow(ax, 7.6, 3.9, 8.3, 3.9, TEAL, lw=2.5)
    styled_arrow(ax, 10.3, 3.0, 10.3, 2.3, GOLD, lw=2.5)

    # Labels
    ax.text(3.2, 5.3, "① Look up", fontsize=11, color=NAVY, fontweight='bold')
    ax.text(7.6, 5.3, "② Sum bonuses", fontsize=11, color=GOLD, fontweight='bold')
    ax.text(11.8, 2.5, "③ Add to\nbase scores", fontsize=10, color=SLATE, fontweight='bold')

    # Footer
    region_box(ax, 1.0, -0.3, 11.0, 0.6, LIGHT_BG, '#E2E8F0')
    ax.text(6.5, 0.0, "Training-free  •  Model-agnostic  •  Test-time only",
            ha='center', fontsize=11, fontstyle='italic', color=MID_GRAY)

    fig.tight_layout()
    fig.savefig(OUT / "cooc_mechanism.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ cooc_mechanism.png")


# =========================================================================
# 8. FIX: Routing Rule — less cramped bottom section
# =========================================================================
def fix_routing_rule():
    """Increase figure height, spread bottom section."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
    ax.text(6, 7.7, "Model Routing Decision Rule",
            fontsize=16, fontweight='bold', ha='center', color=NAVY)

    # Root
    styled_box(ax, 3.5, 6.3, 5, 1.0, "New user arrives", SLATE, WHITE, 13, pad=0.2)

    # Decision 1
    styled_box(ax, 3.5, 4.7, 5, 1.0, "Has game history?", PURPLE, WHITE, 13, pad=0.2)
    styled_arrow(ax, 6.0, 6.3, 6.0, 5.7, SLATE, lw=2)

    # No games branch (left)
    ax.text(2.0, 5.2, "No", fontsize=13, fontweight='bold', color=CORAL)
    styled_box(ax, 0.3, 3.0, 4.0, 1.0, "EMCDR + cooc\n(cold-start CDR)", CORAL, WHITE, 12, pad=0.2)
    styled_arrow(ax, 3.5, 4.8, 2.5, 4.0, CORAL, lw=2)

    # Has games branch (right)
    ax.text(9.5, 5.2, "Yes", fontsize=13, fontweight='bold', color=GREEN_OK)
    styled_box(ax, 7.5, 3.0, 4.0, 1.0, "How many games?", GREEN_OK, WHITE, 12, pad=0.2)
    styled_arrow(ax, 8.5, 4.7, 9.5, 4.0, GREEN_OK, lw=2)

    # Sub-branches
    ax.text(5.8, 2.8, "1-2", fontsize=12, color=GOLD, fontweight='bold')
    styled_box(ax, 4.5, 1.2, 3.5, 1.0, "PTUPCDR + cooc\n(few-shot blend)", GOLD, WHITE, 11, pad=0.2)
    styled_arrow(ax, 8.5, 3.0, 7.0, 2.2, GOLD, lw=2)

    ax.text(10.5, 2.8, "3+", fontsize=12, color=NAVY, fontweight='bold')
    styled_box(ax, 8.5, 1.2, 3.2, 1.0, "LightGCN + cooc\n(graph CF)", NAVY, WHITE, 11, pad=0.2)
    styled_arrow(ax, 10.0, 3.0, 10.1, 2.2, NAVY, lw=2)

    # Niche fallback — more space
    region_box(ax, 0.3, -0.2, 5.5, 0.9, LIGHT_PURPLE, PURPLE, alpha=0.3)
    ax.text(0.5, 0.6, "Always check:", fontsize=10, color=SLATE, fontweight='bold')
    ax.text(3.05, 0.2, "If niche item → SBERT-CDR + cooc", ha='center',
            fontsize=11, color=PURPLE, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT / "routing_rule.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ routing_rule.png")


# =========================================================================
if __name__ == "__main__":
    print("Fixing overlapping diagrams...")
    fix_ncf_mlp()
    fix_model_cooc()
    fix_ptupcdr_moe()
    fix_model_architectures()
    fix_system_architecture()
    fix_recommendation_flow()
    fix_cooc_mechanism()
    fix_routing_rule()
    print(f"\nDone! Fixed 8 diagrams in {OUT}/")
