"""Generate all diagrams for the project report."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)

# Consistent style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.facecolor': 'white',
})

BLUE = '#4472C4'
ORANGE = '#ED7D31'
GREEN = '#70AD47'
RED = '#FF6B6B'
GRAY = '#A5A5A5'
PURPLE = '#7B68EE'
TEAL = '#2EC4B6'
LIGHT_BLUE = '#D6E4F0'
LIGHT_ORANGE = '#FCE4D6'
LIGHT_GREEN = '#E2EFDA'
LIGHT_PURPLE = '#E8E0F0'


def box(ax, x, y, w, h, text, color=BLUE, fc=None, fontsize=10, bold=False):
    """Draw a rounded box with centered text."""
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
# 1. Cross-Domain Concept Diagram
# =========================================================================
def fig_cdr_concept():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title("Cross-Domain Recommendation: Movies → Games", fontsize=14, fontweight='bold', pad=15)

    # Movie domain
    box(ax, 0.3, 3.2, 2.5, 1.2, "Movie Domain\n(Source)", BLUE, LIGHT_BLUE, 12, True)
    for i, m in enumerate(["Action Movies", "Sci-Fi Movies", "RPG Movies"]):
        box(ax, 0.5, 2.3 - i*0.7, 2.1, 0.5, m, BLUE, '#EEF3FA', 8)

    # User in middle
    box(ax, 3.8, 2.5, 2.4, 1.5, "Overlap User\n\nWatched 50 movies\nPlayed 2 games", PURPLE, LIGHT_PURPLE, 9, True)

    # Game domain
    box(ax, 7.2, 3.2, 2.5, 1.2, "Game Domain\n(Target)", ORANGE, LIGHT_ORANGE, 12, True)
    for i, g in enumerate(["Action Games", "Sci-Fi Games", "RPG Games"]):
        box(ax, 7.4, 2.3 - i*0.7, 2.1, 0.5, g, ORANGE, '#FFF2E8', 8)

    # Arrows
    arrow(ax, 2.8, 3.8, 3.8, 3.5, BLUE, '->')
    arrow(ax, 6.2, 3.5, 7.2, 3.8, ORANGE, '->')

    # CDR transfer arrow
    ax.annotate('', xy=(7.0, 1.5), xytext=(3.2, 1.5),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=3,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(5.0, 0.6, "CDR Transfer\n\"Likes action movies → likely likes action games\"",
            ha='center', fontsize=10, fontstyle='italic', color=GREEN)

    fig.tight_layout()
    fig.savefig(OUT / "cdr_concept.png", dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 2. Model Architecture Comparison
# =========================================================================
def fig_model_architectures():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Model Architectures Overview", fontsize=14, fontweight='bold', y=0.98)

    # --- MF-BPR ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("MF-BPR\n(Single-Domain)", fontsize=11, fontweight='bold', color=GRAY)
    box(ax, 1, 7, 3, 1.5, "User\nEmbedding", BLUE, LIGHT_BLUE, 9)
    box(ax, 6, 7, 3, 1.5, "Item\nEmbedding", ORANGE, LIGHT_ORANGE, 9)
    box(ax, 3, 4, 4, 1.5, "Dot Product\nr̂ = uᵀ · i", GREEN, LIGHT_GREEN, 9)
    box(ax, 2.5, 1, 5, 1.5, "BPR Loss\n-log σ(r̂⁺ - r̂⁻)", RED, '#FFE8E8', 9)
    arrow(ax, 2.5, 7, 4, 5.5)
    arrow(ax, 7.5, 7, 6, 5.5)
    arrow(ax, 5, 4, 5, 2.5)

    # --- NCF ---
    ax = axes[0, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("NCF (NeuMF)\n(Single-Domain)", fontsize=11, fontweight='bold', color=GRAY)
    box(ax, 0.5, 7.5, 2, 1.2, "User", BLUE, LIGHT_BLUE, 9)
    box(ax, 7.5, 7.5, 2, 1.2, "Item", ORANGE, LIGHT_ORANGE, 9)
    box(ax, 0.3, 5, 2.5, 1.2, "GMF\nu ⊙ i", BLUE, LIGHT_BLUE, 8)
    box(ax, 7.2, 5, 2.5, 1.2, "MLP\n[u;i]→h", PURPLE, LIGHT_PURPLE, 8)
    box(ax, 3, 2.5, 4, 1.2, "Concat + Predict\nσ(hᵀ[GMF⊕MLP])", GREEN, LIGHT_GREEN, 8)
    arrow(ax, 1.5, 7.5, 1.5, 6.2); arrow(ax, 8.5, 7.5, 8.5, 6.2)
    arrow(ax, 2.8, 5.5, 3.5, 3.7); arrow(ax, 7.2, 5.5, 6.5, 3.7)

    # --- LightGCN ---
    ax = axes[0, 2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("LightGCN\n(Single-Domain)", fontsize=11, fontweight='bold', color=GRAY)
    # Bipartite graph
    for i, y in enumerate([8.5, 7, 5.5]):
        ax.plot(2, y, 'o', color=BLUE, markersize=15)
        ax.text(0.8, y, f"u{i+1}", ha='center', va='center', fontsize=8)
    for i, y in enumerate([8.5, 7, 5.5, 4]):
        ax.plot(5, y, 's', color=ORANGE, markersize=13)
        ax.text(6.2, y, f"g{i+1}", ha='center', va='center', fontsize=8)
    # Edges
    for uy in [8.5, 7, 5.5]:
        for iy in [8.5, 7, 5.5, 4]:
            if np.random.random() > 0.5:
                ax.plot([2, 5], [uy, iy], '-', color='#CCCCCC', lw=0.8)
    box(ax, 1, 1, 8, 2, "Graph Conv → Mean Pool\ne_u = (1/K) Σ e_u^(k)", GREEN, LIGHT_GREEN, 9)
    arrow(ax, 3.5, 4, 5, 3)

    # --- CMF ---
    ax = axes[1, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("CMF\n(Cross-Domain)", fontsize=11, fontweight='bold', color=TEAL)
    box(ax, 0.3, 7, 3, 1.5, "Movie Items\nV_movie", BLUE, LIGHT_BLUE, 9)
    box(ax, 6.7, 7, 3, 1.5, "Game Items\nV_game", ORANGE, LIGHT_ORANGE, 9)
    box(ax, 3, 4.5, 4, 2, "Shared User\nEmbedding U\n(one vector for\nboth domains)", PURPLE, LIGHT_PURPLE, 9, True)
    box(ax, 1, 1, 8, 1.5, "Joint Loss: α·L_movie + (1-α)·L_game", GREEN, LIGHT_GREEN, 9)
    arrow(ax, 1.8, 7, 4, 6.5, BLUE); arrow(ax, 8.2, 7, 6, 6.5, ORANGE)
    arrow(ax, 5, 4.5, 5, 2.5)

    # --- EMCDR ---
    ax = axes[1, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("EMCDR\n(Cross-Domain)", fontsize=11, fontweight='bold', color=TEAL)
    box(ax, 0.2, 7, 2.5, 1.5, "Movie MF\ne_movie(u)", BLUE, LIGHT_BLUE, 9)
    box(ax, 7.3, 7, 2.5, 1.5, "Game MF\ne_game(u)", ORANGE, LIGHT_ORANGE, 9)
    box(ax, 3, 4.5, 4, 1.5, "Global MLP\nf: movie → game", GREEN, LIGHT_GREEN, 10, True)
    box(ax, 2, 1.5, 6, 1.5, "Predict: ê_game(u) = MLP(e_movie(u))", PURPLE, LIGHT_PURPLE, 9)
    arrow(ax, 1.5, 7, 4, 6, BLUE, lw=2)
    arrow(ax, 8.5, 7, 6, 6, ORANGE)
    arrow(ax, 5, 4.5, 5, 3)
    # Phase labels
    ax.text(1.5, 9.3, "① Train", fontsize=8, color=BLUE, fontweight='bold')
    ax.text(8.0, 9.3, "② Train", fontsize=8, color=ORANGE, fontweight='bold')
    ax.text(4.5, 6.3, "③ Map", fontsize=8, color=GREEN, fontweight='bold')

    # --- PTUPCDR ---
    ax = axes[1, 2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("PTUPCDR\n(Cross-Domain)", fontsize=11, fontweight='bold', color=TEAL)
    box(ax, 0.2, 7, 2.5, 1.5, "Movie MF\ne_movie(u)", BLUE, LIGHT_BLUE, 9)
    box(ax, 7.3, 7, 2.5, 1.5, "Game MF\ne_game(u)", ORANGE, LIGHT_ORANGE, 9)
    # MoE
    box(ax, 2.5, 4, 5, 2.2, "MoE Hypernetwork\nExpert₁  Expert₂  ...  Expert_k\n\nPersonalized per user", GREEN, LIGHT_GREEN, 9, True)
    box(ax, 1.5, 1, 7, 1.5, "Blend: w·MoE(e_movie) + (1-w)·e_game\nw = 1/(1+k), k = #game interactions", PURPLE, LIGHT_PURPLE, 8)
    arrow(ax, 1.5, 7, 4, 6.2, BLUE, lw=2)
    arrow(ax, 8.5, 7, 6, 6.2, ORANGE)
    arrow(ax, 5, 4, 5, 2.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "model_architectures.png", dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 3. Data Pipeline Diagram
# =========================================================================
def fig_data_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis('off')
    ax.set_title("Data Processing Pipeline", fontsize=14, fontweight='bold', pad=10)

    steps = [
        ("Raw JSONL\n(Amazon 2023)", GRAY, '#F0F0F0'),
        ("Rating ≥ 4\n→ Implicit", BLUE, LIGHT_BLUE),
        ("Item K-Core\nM≥20, G≥10", ORANGE, LIGHT_ORANGE),
        ("User Filter\n(per lesson)", GREEN, LIGHT_GREEN),
        ("LLO Split\ntrain/val/test", PURPLE, LIGHT_PURPLE),
        ("Evaluate\n@10", RED, '#FFE8E8'),
    ]
    for i, (text, color, fc) in enumerate(steps):
        x = 0.3 + i * 2.0
        box(ax, x, 1, 1.6, 2, text, color, fc, 9, True)
        if i < len(steps) - 1:
            arrow(ax, x + 1.6, 2, x + 2.0, 2, '#888888', '->')

    fig.tight_layout()
    fig.savefig(OUT / "data_pipeline.png", dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 4. Experiment Progression (Lesson Flow)
# =========================================================================
def fig_lesson_flow():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title("Experiment Progression: Each Lesson Isolates One Variable", fontsize=13, fontweight='bold', pad=10)

    lessons = [
        ("L1", "BPR vs\nExplicit", "Loss\nfunction", GRAY),
        ("L2", "Low overlap\n(5.8%)", "Overlap\nratio", BLUE),
        ("L3", "100%\noverlap", "User\nfilter", BLUE),
        ("L4", "Source-rich\n(movies≥10)", "Source\nrichness", GREEN),
        ("L5", "Catalog\nfilter", "Item\ncatalog", ORANGE),
        ("L6", "Cold-start\n(0 games)", "Target\nsparsity", RED),
        ("L7", "SBERT\ncontent", "Signal\ntype", PURPLE),
        ("L8", "Cooc\nrerank", "Post-\nprocessing", TEAL),
    ]

    for i, (num, desc, variable, color) in enumerate(lessons):
        x = 0.3 + i * 1.58
        # Lesson box
        box(ax, x, 2.5, 1.35, 2.5, f"{num}\n\n{desc}", color, color + '20', 9, True)
        # Variable label below
        box(ax, x, 1, 1.35, 1.2, variable, '#666666', '#F8F8F8', 8)
        # Arrow to next
        if i < len(lessons) - 1:
            arrow(ax, x + 1.35, 3.75, x + 1.58, 3.75, '#AAAAAA', '->')

    # Legend
    ax.text(0.3, 5.5, "Variable isolated:", fontsize=10, fontweight='bold')
    ax.text(0.3, 0.3, "→ Each lesson changes exactly one variable from the previous, keeping everything else constant",
            fontsize=9, fontstyle='italic', color='#666666')

    fig.tight_layout()
    fig.savefig(OUT / "lesson_flow.png", dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 5. Routing Decision Diagram
# =========================================================================
def fig_routing():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title("Model Routing Decision Rule", fontsize=14, fontweight='bold', pad=10)

    # Root
    box(ax, 3.5, 5.8, 4, 0.9, "New user arrives", '#333333', '#F5F5F5', 11, True)

    # Decision 1
    box(ax, 3.5, 4.3, 4, 0.9, "Has game history?", PURPLE, LIGHT_PURPLE, 10, True)
    arrow(ax, 5.5, 5.8, 5.5, 5.2)

    # No games branch (left)
    ax.text(2.0, 4.7, "No", fontsize=10, fontweight='bold', color=RED)
    box(ax, 0.3, 2.8, 3.5, 0.9, "EMCDR + cooc\n(cold-start CDR)", RED, '#FFE8E8', 10, True)
    arrow(ax, 3.5, 4.5, 3.0, 3.7, RED)

    # Has games branch (right)
    ax.text(8.5, 4.7, "Yes", fontsize=10, fontweight='bold', color=GREEN)
    box(ax, 6.8, 2.8, 3.5, 0.9, "How many games?", GREEN, LIGHT_GREEN, 10, True)
    arrow(ax, 7.5, 4.3, 8.5, 3.7, GREEN)

    # Sub-branches
    ax.text(5.8, 2.5, "1-2", fontsize=9, color=ORANGE, fontweight='bold')
    box(ax, 4.5, 1.2, 2.8, 0.9, "PTUPCDR + cooc\n(few-shot blend)", ORANGE, LIGHT_ORANGE, 9, True)
    arrow(ax, 7.5, 2.8, 6.5, 2.1, ORANGE)

    ax.text(9.5, 2.5, "3+", fontsize=9, color=BLUE, fontweight='bold')
    box(ax, 7.8, 1.2, 2.8, 0.9, "LightGCN + cooc\n(graph CF)", BLUE, LIGHT_BLUE, 9, True)
    arrow(ax, 9.0, 2.8, 9.2, 2.1, BLUE)

    # Niche fallback
    box(ax, 0.3, 0, 3.5, 0.8, "If niche item → SBERT-CDR + cooc", PURPLE, LIGHT_PURPLE, 8)
    ax.text(0.3, 0.85, "Always check:", fontsize=8, color='#666666')

    fig.tight_layout()
    fig.savefig(OUT / "routing_rule.png", dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 6. Cooc Mechanism Diagram
# =========================================================================
def fig_cooc_mechanism():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis('off')
    ax.set_title("Cross-Domain Item Co-occurrence Reranking", fontsize=13, fontweight='bold', pad=10)

    # User's movie history
    box(ax, 0.2, 3, 2.3, 1.5, "User's Movies\n\nAction Movie A\nSci-Fi Movie B\nRPG Movie C", BLUE, LIGHT_BLUE, 9)

    # Cooc matrix
    box(ax, 3.3, 2.5, 3, 2.3, "Co-occurrence Matrix\n\ncooc[movie][game]\n= log(1 + count of\n  users who liked both)", GREEN, LIGHT_GREEN, 8, True)

    # Bonus scores
    box(ax, 7.0, 3, 3.5, 1.5, "Game Score Bonus\n\nAction Game: +0.8\nRPG Game: +0.5\nPuzzle Game: +0.1", ORANGE, LIGHT_ORANGE, 9)

    # Base model
    box(ax, 7.0, 0.5, 3.5, 1.5, "Base Model Score\n(LightGCN, EMCDR, ...)\n+ λ × bonus", PURPLE, LIGHT_PURPLE, 9, True)

    # Arrows
    arrow(ax, 2.5, 3.8, 3.3, 3.8, BLUE, '->', lw=2)
    arrow(ax, 6.3, 3.8, 7.0, 3.8, ORANGE, '->', lw=2)
    arrow(ax, 8.75, 3.0, 8.75, 2.0, '#666666', '->', lw=2)

    # Labels
    ax.text(2.6, 4.7, "① Look up", fontsize=9, color=BLUE, fontweight='bold')
    ax.text(6.3, 4.7, "② Sum bonuses", fontsize=9, color=ORANGE, fontweight='bold')
    ax.text(9.5, 2.2, "③ Add to\nbase scores", fontsize=8, color=PURPLE, fontweight='bold')

    ax.text(0.2, 0.3, "Training-free • Model-agnostic • Test-time only",
            fontsize=10, fontstyle='italic', color='#888888')

    fig.tight_layout()
    fig.savefig(OUT / "cooc_mechanism.png", dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 7. Key Results Summary Chart
# =========================================================================
def fig_results_summary():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Key Results: Recall@10 Across Regimes", fontsize=13, fontweight='bold')

    # LLO results (L3)
    models_llo = ['LightGCN', 'MF-BPR', 'CMF', 'PTUPCDR', 'NCF', 'EMCDR']
    base_llo = [0.0595, 0.0445, 0.0395, 0.0315, 0.0255, 0.0225]
    cooc_llo = [0.0625, 0.0520, 0.0380, 0.0355, 0.0370, 0.0335]

    x = np.arange(len(models_llo))
    w = 0.35
    ax1.bar(x - w/2, base_llo, w, label='Base', color=BLUE, alpha=0.8)
    ax1.bar(x + w/2, cooc_llo, w, label='+Cooc', color=ORANGE, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_llo, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('Recall@10')
    ax1.set_title('Standard Evaluation (LLO, L3)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 0.08)

    # Cold-start results (L6)
    models_cs = ['Popularity', 'EMCDR', 'PTUPCDR', 'LightGCN', 'CMF', 'MF-BPR']
    base_cs = [0.0381, 0.0334, 0.0301, 0.0067, 0.0020, 0.0007]
    cooc_cs = [0.0387, 0.0341, 0.0301, 0.0261, 0.0321, 0.0007]

    x = np.arange(len(models_cs))
    ax2.bar(x - w/2, base_cs, w, label='Base', color=BLUE, alpha=0.8)
    ax2.bar(x + w/2, cooc_cs, w, label='+Cooc', color=ORANGE, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_cs, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Recall@10')
    ax2.set_title('Cold-Start (L6, zero game history)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 0.05)

    fig.tight_layout()
    fig.savefig(OUT / "results_summary.png", dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 8. SBERT Concept Diagram
# =========================================================================
def fig_sbert_concept():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')
    ax.set_title("SBERT-CDR: Content-Based Cross-Domain Transfer", fontsize=13, fontweight='bold', pad=10)

    # Movie items → SBERT
    box(ax, 0.2, 2.2, 2, 1.3, "Movie Titles\n& Descriptions", BLUE, LIGHT_BLUE, 9)
    box(ax, 2.8, 2.2, 2.2, 1.3, "SBERT\nEncoder\n(384-dim)", GREEN, LIGHT_GREEN, 9, True)
    box(ax, 5.6, 2.2, 2, 1.3, "User Profile\n= mean of\nmovie vectors", PURPLE, LIGHT_PURPLE, 9)
    box(ax, 8.0, 2.2, 1.8, 1.3, "Rank Games\nby cosine\nsimilarity", ORANGE, LIGHT_ORANGE, 9)

    arrow(ax, 2.2, 2.9, 2.8, 2.9, BLUE, '->', lw=2)
    arrow(ax, 5.0, 2.9, 5.6, 2.9, GREEN, '->', lw=2)
    arrow(ax, 7.6, 2.9, 8.0, 2.9, PURPLE, '->', lw=2)

    # Game items also encoded
    box(ax, 2.8, 0.3, 2.2, 1.3, "Game Titles\n& Descriptions", ORANGE, LIGHT_ORANGE, 9)
    arrow(ax, 5.0, 0.9, 8.5, 2.2, ORANGE, '->')
    ax.text(6.5, 1.2, "Same SBERT\nspace", fontsize=8, fontstyle='italic', color='#666')

    fig.tight_layout()
    fig.savefig(OUT / "sbert_concept.png", dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 9. Overlap Impact Chart
# =========================================================================
def fig_overlap_impact():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = ['LightGCN', 'EMCDR', 'PTUPCDR', 'CMF']
    l2 = [0.0290, 0.0160, 0.0085, 0.0050]  # 5.8% overlap
    l3 = [0.0595, 0.0225, 0.0315, 0.0395]  # 100% overlap

    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, l2, w, label='L2: 5.8% overlap', color=RED, alpha=0.7)
    ax.bar(x + w/2, l3, w, label='L3: 100% overlap', color=GREEN, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Recall@10', fontsize=11)
    ax.set_title('Impact of User Overlap on Model Performance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Annotate CDR improvements
    for i in [1, 2, 3]:
        pct = (l3[i] - l2[i]) / l2[i] * 100
        ax.annotate(f'+{pct:.0f}%', xy=(i + w/2, l3[i]), xytext=(i + w/2, l3[i] + 0.003),
                    ha='center', fontsize=9, fontweight='bold', color=GREEN)

    fig.tight_layout()
    fig.savefig(OUT / "overlap_impact.png", dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# 10. Cold-Start Comparison
# =========================================================================
def fig_coldstart_comparison():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    models = ['Popularity', 'EMCDR', 'PTUPCDR', 'LightGCN', 'CMF', 'NCF', 'MF-BPR']
    recall = [0.0381, 0.0334, 0.0301, 0.0067, 0.0020, 0.0020, 0.0007]
    colors = [GRAY, TEAL, TEAL, BLUE, TEAL, BLUE, BLUE]

    bars = ax.barh(models[::-1], recall[::-1], color=colors[::-1], alpha=0.8, height=0.6)
    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_title('Cold-Start Performance (Zero Game History)', fontsize=12, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, recall[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    # Legend
    ax.text(0.025, -0.6, '■ CDR    ■ Single-Domain    ■ Baseline',
            fontsize=9, color='#666')

    fig.tight_layout()
    fig.savefig(OUT / "coldstart_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    fig_cdr_concept()
    fig_model_architectures()
    fig_data_pipeline()
    fig_lesson_flow()
    fig_routing()
    fig_cooc_mechanism()
    fig_results_summary()
    fig_sbert_concept()
    fig_overlap_impact()
    fig_coldstart_comparison()
    print(f"Generated {len(list(OUT.glob('*.png')))} figures in {OUT}/")
