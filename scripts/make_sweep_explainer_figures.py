"""Explainer schematics for §5.9 hyperparameter sweeps.

For each model, one PNG illustrates *why* the winning hyperparameter wins —
not the sweep curve (which already lives in fig_hp_<model>.png), but the
mechanism the sweep exposes:

  fig_hp_cmf_explainer.png       — α re-weights the loss against a skewed data mix
  fig_hp_emcdr_explainer.png     — three λ regimes: weak / corrective / overridden
  fig_hp_lightgcn_explainer.png  — K-hop neighborhood reach on a bipartite graph
  fig_hp_ptupcdr_explainer.png   — users-per-expert shrinks as n_experts grows
  fig_hp_sbert_explainer.png     — profile blend pulled toward movies or games
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report_figures_v3"
OUT.mkdir(exist_ok=True)

BLUE    = "#4472C4"
TEAL    = "#2EC4B6"
GREEN   = "#10B981"
ORANGE  = "#ED7D31"
RED     = "#EF4444"
PINK    = "#EC4899"
PURPLE  = "#7B68EE"
GREY    = "#A5A5A5"
INK     = "#0F172A"
MUTED   = "#94A3B8"
MOVIE   = "#3B82F6"
GAME    = "#EF4444"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _setup(ax, xlim, ylim, equal=True):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _chip(ax, x, y, w, h, text, color, fc, fontsize=10, tc=INK, fw="bold"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=color, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, fontweight=fw)


def _arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.3, style="->",
           cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, color=color,
        lw=lw, mutation_scale=12, connectionstyle=cs))


# ────────────────────────────────────────────────────────────────────────
# CMF — α corrects for data skew
# ────────────────────────────────────────────────────────────────────────

def fig_cmf_explainer():
    """Show that interactions are ~80% movies vs 20% games, so α ≠ 0.5."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)

    # Left: interaction-volume bar
    _setup(ax1, (0, 10), (0, 6), equal=False)
    ax1.set_title("1.   Data is skewed toward movies",
                  fontsize=12, fontweight="bold", color=INK, loc="left")
    ax1.add_patch(Rectangle((0.5, 3.6), 8.0, 0.9,
                            facecolor=MOVIE, edgecolor="none"))
    ax1.add_patch(Rectangle((0.5 + 8.0, 3.6), 1.2, 0.9,
                            facecolor=GAME, edgecolor="none"))
    ax1.text(4.5, 4.05, "~80% movie interactions", ha="center", va="center",
             color="white", fontsize=11, fontweight="bold")
    ax1.text(9.1, 4.05, "~20%", ha="center", va="center",
             color="white", fontsize=10, fontweight="bold")
    ax1.text(0.5, 3.2, "every SGD batch draws from this mix",
             fontsize=10, color=MUTED, style="italic")

    # Effective gradient bars under 3 α values
    labels = ["α = 0.5  (equal weight)",
              "α = 0.2  (rebalanced — winner)",
              "α = 0.01 (target-only)"]
    # contribution = (movie_frac * α)  vs  (game_frac * (1-α))
    splits = [(0.80 * 0.5, 0.20 * 0.5),
              (0.80 * 0.2, 0.20 * 0.8),
              (0.80 * 0.01, 0.20 * 0.99)]
    y0 = 2.3
    for i, (lab, (m, g)) in enumerate(zip(labels, splits)):
        y = y0 - 0.7 * i
        total = m + g
        mw = 7.0 * m / total
        gw = 7.0 * g / total
        ax1.add_patch(Rectangle((1.5, y), mw, 0.35,
                                facecolor=MOVIE, edgecolor="none"))
        ax1.add_patch(Rectangle((1.5 + mw, y), gw, 0.35,
                                facecolor=GAME, edgecolor="none"))
        ax1.text(1.4, y + 0.17, lab, ha="right", va="center",
                 fontsize=9.5, color=INK)
        ax1.text(1.55 + mw / 2, y + 0.17, f"{m/total:.0%}",
                 ha="center", va="center", color="white",
                 fontsize=8.5, fontweight="bold")
        ax1.text(1.55 + mw + gw / 2, y + 0.17, f"{g/total:.0%}",
                 ha="center", va="center", color="white",
                 fontsize=8.5, fontweight="bold")
    ax1.text(1.5, 0.4, "effective gradient split movies : games",
             fontsize=9, color=MUTED, style="italic")

    # Right: verdict panel
    _setup(ax2, (0, 10), (0, 6), equal=False)
    ax2.set_title("2.   α = 0.2 = Recall@10 = 0.041",
                  fontsize=12, fontweight="bold", color=INK, loc="left")
    # Three small outcome chips
    outcomes = [
        ("α = 0.5",   "movies dominate the gradient,\ngame predictions compromised",
         GREY,   "#F1F5F9", "0.031"),
        ("α = 0.2",   "games get the majority of loss\nweight despite 20% of data",
         GREEN,  "#ECFDF5", "0.041"),
        ("α = 0.01",  "source signal ignored — no\ncross-domain lift at all",
         ORANGE, "#FFF7ED", "0.037"),
    ]
    for i, (tag, body, col, fc, score) in enumerate(outcomes):
        y = 4.4 - 1.55 * i
        _chip(ax2, 0.3, y, 1.6, 1.1, tag, col, fc, fontsize=11)
        ax2.text(2.1, y + 0.55, body, va="center", ha="left",
                 fontsize=9.5, color=INK)
        ax2.text(8.9, y + 0.55, f"Recall@10\n{score}", va="center",
                 ha="center", fontsize=10, color=col, fontweight="bold")

    fig.suptitle(
        "CMF — why α = 0.2 wins: the source-weight corrects a 4:1 data skew",
        fontsize=13, fontweight="bold", y=1.00, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_hp_cmf_explainer.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# EMCDR + cooc — three λ regimes
# ────────────────────────────────────────────────────────────────────────

def fig_emcdr_explainer():
    """Three schematic rankings: λ=0 / λ=0.05 / λ=0.3 — what each regime does."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), dpi=180)

    panels = [
        ("λ = 0.0", "only EMCDR signal",
         [("Indie RPG A",    0.91, True),
          ("Puzzle B",       0.88, False),
          ("Strategy C",     0.85, True),
          ("Racing D",       0.80, False),
          ("Platformer E",   0.77, False)],
         GREY, "R@10 = 0.026"),
        ("λ = 0.05", "cooc nudges popular\ntransfer matches up",
         [("Strategy C",     0.92, True),   # moved up — cooc-supported winner
          ("Indie RPG A",    0.91, True),
          ("Popular FPS F",  0.86, True),   # new entry from cooc
          ("Puzzle B",       0.83, False),
          ("Racing D",       0.78, False)],
         GREEN, "R@10 = 0.034"),
        ("λ = 0.30", "cooc dominates, ranking\nignores user's learned profile",
         [("Popular FPS F",  1.20, True),
          ("AAA Shooter G",  1.10, False),
          ("MMO Classic H",  1.05, False),
          ("Strategy C",     0.98, True),
          ("Indie RPG A",    0.90, True)],
         RED, "R@10 = 0.029"),
    ]

    for ax, (tag, sub, items, color, score) in zip(axes, panels):
        _setup(ax, (0, 10), (0, 7.5), equal=False)
        _chip(ax, 0.4, 6.4, 2.6, 0.8, tag, color,
              ("#ECFDF5" if color == GREEN
               else "#FEF2F2" if color == RED else "#F1F5F9"),
              fontsize=12)
        ax.text(3.2, 6.8, sub, ha="left", va="center",
                fontsize=10, color=MUTED, style="italic")

        # Ranking list
        for i, (name, score_val, relevant) in enumerate(items):
            y = 5.3 - i * 0.95
            row_color = GREEN if relevant else GREY
            row_fc = "#ECFDF5" if relevant else "#F8FAFC"
            _chip(ax, 0.4, y, 7.3, 0.7, f"{i+1}.  {name}",
                  row_color, row_fc, fontsize=10,
                  tc=INK, fw="normal")
            ax.text(8.2, y + 0.35, "✓" if relevant else "·",
                    ha="center", va="center",
                    fontsize=14 if relevant else 16,
                    color=GREEN if relevant else MUTED,
                    fontweight="bold")

        ax.text(5.0, 0.25, score, ha="center", fontsize=11,
                fontweight="bold", color=color)

    fig.suptitle(
        "EMCDR + cooc — why λ = 0.05 wins: a nudge lifts cooc-supported items, a shove overrides personalisation",
        fontsize=12, fontweight="bold", y=1.03, color=INK)
    fig.text(0.5, 0.03,
             "✓ = relevant (in held-out test); rankings are schematic, scores are from the sweep",
             ha="center", fontsize=9, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUT / "fig_hp_emcdr_explainer.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# LightGCN — K-hop neighbourhood reach
# ────────────────────────────────────────────────────────────────────────

def fig_lightgcn_explainer():
    """Bipartite graph with progressively larger receptive fields for K=1,2,4,5."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), dpi=180)
    axes = axes.flatten()

    # Shared graph: 4 users, 5 items
    user_pos  = [(1.0, y) for y in [4.0, 3.0, 2.0, 1.0]]
    item_pos  = [(4.5, y) for y in [4.2, 3.3, 2.4, 1.5, 0.6]]
    edges = [(0, 0), (0, 1),             # u0
             (1, 1), (1, 2),             # u1
             (2, 2), (2, 3), (2, 4),     # u2
             (3, 0), (3, 3)]             # u3

    # Each K: which users/items are "reached" from u0
    # BFS on bipartite graph
    from collections import defaultdict, deque
    adj = defaultdict(set)
    for u, it in edges:
        adj[("u", u)].add(("i", it))
        adj[("i", it)].add(("u", u))

    def reach(start, k):
        frontier = {start}
        seen = {start: 0}
        for step in range(1, k + 1):
            new = set()
            for node in frontier:
                for nb in adj[node]:
                    if nb not in seen:
                        seen[nb] = step
                        new.add(nb)
            frontier = new
        return seen

    titles = [
        ("K = 1", "direct neighbours only", BLUE),
        ("K = 2", "2-hop: users who share items", TEAL),
        ("K = 4", "4-hop — plateau, winner", GREEN),
        ("K = 5+", "all embeddings collapse", RED),
    ]
    Ks = [1, 2, 4, 5]

    for ax, (tag, sub, color), K in zip(axes, titles, Ks):
        _setup(ax, (0, 6), (0, 5), equal=False)
        ax.set_title(f"{tag}   ·   {sub}",
                     fontsize=11, fontweight="bold", color=color, loc="left")

        seen = reach(("u", 0), K)
        over_smooth = (K >= 5)

        # Draw edges
        for u, it in edges:
            up, ip = user_pos[u], item_pos[it]
            # both endpoints need to be reached to color the edge
            both_in = ("u", u) in seen and ("i", it) in seen
            col = color if both_in and not over_smooth else "#E5E7EB"
            lw = 1.7 if both_in and not over_smooth else 0.8
            ax.plot([up[0], ip[0]], [up[1], ip[1]], color=col, lw=lw,
                    zorder=1, alpha=0.95 if both_in else 0.6)

        # Draw users
        for u, pos in enumerate(user_pos):
            reached = ("u", u) in seen
            fc = color if (reached and not over_smooth) else ("#9CA3AF" if over_smooth and reached else "white")
            ec = color if reached else MUTED
            if u == 0:  # highlight start
                ax.add_patch(Circle(pos, 0.22, facecolor=color, edgecolor=INK, lw=2, zorder=3))
                ax.text(*pos, "u₀", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold", zorder=4)
            else:
                ax.add_patch(Circle(pos, 0.19, facecolor=fc, edgecolor=ec, lw=1.6, zorder=3))
                ax.text(*pos, f"u{u}", ha="center", va="center",
                        color="white" if reached and not over_smooth else MUTED,
                        fontsize=9, zorder=4)

        # Draw items
        for it, pos in enumerate(item_pos):
            reached = ("i", it) in seen
            fc = color if (reached and not over_smooth) else ("#9CA3AF" if over_smooth and reached else "white")
            ec = color if reached else MUTED
            ax.add_patch(Rectangle((pos[0] - 0.18, pos[1] - 0.18), 0.36, 0.36,
                                    facecolor=fc, edgecolor=ec, lw=1.6, zorder=3))
            ax.text(*pos, f"g{it}", ha="center", va="center",
                    color="white" if reached and not over_smooth else MUTED,
                    fontsize=9, zorder=4)

        # Legend blurb
        n_reach_items = sum(1 for k in seen if k[0] == "i")
        n_reach_users = sum(1 for k in seen if k[0] == "u")
        ax.text(3.0, 0.15,
                f"{n_reach_users}/4 users · {n_reach_items}/5 items in receptive field",
                ha="center", fontsize=9.5, color=MUTED, style="italic")

    fig.suptitle(
        "LightGCN — why K = 4 wins: propagation reaches enough neighbours without collapsing all embeddings",
        fontsize=13, fontweight="bold", y=1.00, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig_hp_lightgcn_explainer.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# PTUPCDR — users per expert
# ────────────────────────────────────────────────────────────────────────

def fig_ptupcdr_explainer():
    """As n_experts grows, each expert sees fewer users → overfitting."""
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=180)
    _setup(ax, (0, 13), (0, 9), equal=False)
    ax.set_title(
        "PTUPCDR — why n = 2 wins on 100% overlap: more experts means fewer users each",
        fontsize=12, fontweight="bold", color=INK, loc="left")

    # Fixed user pool size (20k users on Lesson 3)
    N = 20000
    configs = [
        (2,  GREEN,  "#ECFDF5"),
        (4,  BLUE,   "#EFF6FF"),
        (8,  ORANGE, "#FFF7ED"),
        (16, RED,    "#FEF2F2"),
    ]

    col_x = [1.8, 4.4, 7.0, 9.6]

    # Pool header, at the very top
    ax.text(5.7, 8.2, f"fixed pool: {N:,} overlapping users",
            ha="center", fontsize=11, color=INK, fontweight="bold")

    # draw each config as a column of mini expert boxes
    for (n, col, fc), cx in zip(configs, col_x):
        users_per = N // n
        # Big header chip at top of column
        _chip(ax, cx - 0.9, 7.1, 1.8, 0.7,
              f"n = {n}", col, fc, fontsize=11)
        # Recall score right under the header chip
        scores = {2: "0.033", 4: "0.028", 8: "0.030", 16: "0.029"}
        ax.text(cx, 6.7,
                f"Recall@10 = {scores[n]}",
                ha="center", fontsize=9.5, color=col, fontweight="bold")

        # Expert boxes — one per expert, sized to reflect data each gets
        per_box_w = 1.5
        # We show at most 4 boxes stacked; if n>4 we show 4 with "…" between
        shown_n = min(n, 4)
        total_h = 4.6
        box_h = total_h / shown_n * 0.72
        gap = (total_h - shown_n * box_h) / max(1, shown_n - 1) if shown_n > 1 else 0
        for k in range(shown_n):
            y = 1.3 + k * (box_h + gap)
            _chip(ax, cx - per_box_w/2, y, per_box_w, box_h,
                  f"expert {k+1}", INK, "#F8FAFC",
                  fontsize=8.5, tc=INK, fw="normal")
        if n > 4:
            ax.text(cx, 0.95, "⋮", ha="center", va="center",
                    fontsize=16, color=MUTED)

        # Users-per-expert chip under
        color_for_up = (GREEN if users_per >= 5000 else
                        BLUE if users_per >= 2500 else
                        ORANGE if users_per >= 1500 else RED)
        ax.text(cx, 0.4,
                f"≈ {users_per:,}\nusers / expert",
                ha="center", va="center", fontsize=9.5,
                color=color_for_up, fontweight="bold")

    # Inline captions above the column header chips
    ax.text(col_x[0], 7.95, "sweet spot — enough data each",
            ha="center", fontsize=9, color=GREEN, style="italic")
    ax.text(col_x[-1], 7.95, "too few users per expert — router overfits",
            ha="center", fontsize=9, color=RED, style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "fig_hp_ptupcdr_explainer.png",
                bbox_inches="tight", dpi=180)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# SBERT-CDR — profile blend on high-overlap cohort
# ────────────────────────────────────────────────────────────────────────

def fig_sbert_explainer():
    """User's game-games cluster vs movie cluster; source_weight pulls the profile."""
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=180)
    _setup(ax, (-1, 11), (-1, 7), equal=False)
    ax.set_title(
        "SBERT-CDR — why source_weight = 0.1 wins on 100% overlap: target items live near game history",
        fontsize=12, fontweight="bold", color=INK, loc="left")

    # Movie cluster (left), Game cluster (right), target items (near game cluster)
    rng = np.random.default_rng(0)
    movie_centre = np.array([1.8, 3.2])
    game_centre  = np.array([7.6, 3.0])
    target_centre = np.array([8.2, 2.6])

    # Plot movie rated items
    pts = movie_centre + rng.normal(0, 0.45, (12, 2))
    ax.scatter(pts[:, 0], pts[:, 1], color=MOVIE, s=38, alpha=0.75,
               edgecolor="white", linewidth=0.8, label="rated movies", zorder=2)
    # Plot game rated items
    pts = game_centre + rng.normal(0, 0.4, (10, 2))
    ax.scatter(pts[:, 0], pts[:, 1], color=GAME, s=38, alpha=0.75,
               edgecolor="white", linewidth=0.8, label="rated games", zorder=2)
    # Candidate game items (the ranking target)
    pts = target_centre + rng.normal(0, 0.65, (30, 2))
    ax.scatter(pts[:, 0], pts[:, 1], color=GAME, s=22, alpha=0.30,
               marker="s", edgecolor="none", label="candidate games", zorder=1)

    # Profile positions for two source_weights
    sw01_profile = 0.1 * movie_centre + 0.9 * game_centre
    sw09_profile = 0.9 * movie_centre + 0.1 * game_centre

    ax.scatter(*sw01_profile, color=GREEN, s=260, marker="*",
               edgecolor=INK, linewidth=1.4, zorder=4,
               label="profile @ sw=0.1 (winner)")
    ax.annotate("sw = 0.1\nprofile sits\namong games",
                xy=sw01_profile, xytext=(sw01_profile[0] - 1.0, sw01_profile[1] + 1.3),
                fontsize=9.5, color=GREEN, fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))

    ax.scatter(*sw09_profile, color=RED, s=260, marker="*",
               edgecolor=INK, linewidth=1.4, zorder=4,
               label="profile @ sw=0.9 (loses)")
    ax.annotate("sw = 0.9\nprofile pulled\naway from candidates",
                xy=sw09_profile, xytext=(sw09_profile[0] - 1.4, sw09_profile[1] - 1.6),
                fontsize=9.5, color=RED, fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))

    # Centre lines for clusters
    for centre, name, col in [(movie_centre, "movie cluster", MOVIE),
                              (game_centre,  "game cluster",  GAME)]:
        ax.add_patch(plt.Circle(centre, 1.3, fill=False,
                                edgecolor=col, lw=1.5, linestyle="--",
                                alpha=0.6))
        ax.text(centre[0], centre[1] + 1.6, name, ha="center",
                fontsize=9, color=col, style="italic")

    # Candidate region label
    ax.text(target_centre[0], target_centre[1] - 1.8,
            "ranking target: game items",
            ha="center", fontsize=9, color=GAME, style="italic")

    # Verdict chips
    _chip(ax, 0.2, 5.7, 3.2, 0.7,
          "Recall@10 @ sw=0.1 :  0.029",
          GREEN, "#ECFDF5", fontsize=10)
    _chip(ax, 0.2, 4.9, 3.2, 0.7,
          "Recall@10 @ sw=0.9 :  0.004",
          RED, "#FEF2F2", fontsize=10)

    ax.legend(loc="lower left", fontsize=9, frameon=True,
              facecolor="white", edgecolor=MUTED)

    fig.tight_layout()
    fig.savefig(OUT / "fig_hp_sbert_explainer.png",
                bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_cmf_explainer()
    fig_emcdr_explainer()
    fig_lightgcn_explainer()
    fig_ptupcdr_explainer()
    fig_sbert_explainer()
    print("Saved explainer figures to", OUT)
