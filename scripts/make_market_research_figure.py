"""Market-research illustration for §2.1.

Two-panel figure:
  left  — three platforms (Netflix / Amazon / Spotify) with their domain spans
  right — three business outcomes unlocked by CDR (retention / cross-sell /
          happy surprise)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "report_figures_v3" / "fig_market_research.png"
OUT.parent.mkdir(exist_ok=True)

NETFLIX = "#E50914"
AMAZON = "#FF9900"
SPOTIFY = "#1DB954"
INK = "#111111"
PANEL = "#F6F7F9"
TEAL = "#2EC4B6"
BLUE = "#4472C4"
PINK = "#EC4899"


def card(ax, x, y, w, h, color, title, subtitle, bullets):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=0, facecolor=color, alpha=0.12))
    # coloured strip on the left
    ax.add_patch(FancyBboxPatch((x, y), 0.12, h,
                                boxstyle="round,pad=0.0,rounding_size=0.06",
                                linewidth=0, facecolor=color, alpha=0.9))
    ax.text(x + 0.22, y + h - 0.22, title, fontsize=14, fontweight="bold",
            color=color, va="top")
    ax.text(x + 0.22, y + h - 0.48, subtitle, fontsize=10, color=INK,
            va="top")
    for i, b in enumerate(bullets):
        ax.text(x + 0.22, y + h - 0.78 - i * 0.34, "•  " + b, fontsize=9.5,
                color=INK, va="top")


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=180,
                                    gridspec_kw={"width_ratios": [1, 1]})

    # ── left: platforms ───────────────────────────────────────────────
    ax1.set_xlim(0, 5); ax1.set_ylim(0, 6)
    ax1.set_title("Platforms already span multiple domains",
                  fontsize=13, fontweight="bold", loc="left", pad=12)
    ax1.axis("off")
    card(ax1, 0.1, 4.05, 4.8, 1.75, NETFLIX, "Netflix",
         "Video  +  Mobile Games  (since 2021)",
         ["2 domains, 1 account", "Same taste signal, different catalog"])
    card(ax1, 0.1, 2.1,  4.8, 1.75, AMAZON, "Amazon",
         "Cross-category product recommendation",
         ["Full e-commerce catalog", "Cross-category is the native mode"])
    card(ax1, 0.1, 0.15, 4.8, 1.75, SPOTIFY, "Spotify",
         "Music  +  Podcasts  (unified feed)",
         ["2 media types in one recommender",
          "Bridged by shared listening behaviour"])

    # ── right: outcomes ───────────────────────────────────────────────
    ax2.set_xlim(0, 5); ax2.set_ylim(0, 6)
    ax2.set_title("Business outcomes a working CDR unlocks",
                  fontsize=13, fontweight="bold", loc="left", pad=12)
    ax2.axis("off")
    card(ax2, 0.1, 4.05, 4.8, 1.75, BLUE, "Retention ↑",
         "More relevant top-K → longer sessions",
         ["Fewer dead-ends for low-history users",
          "Personalisation at cold-start"])
    card(ax2, 0.1, 2.1,  4.8, 1.75, TEAL, "Cross-sell",
         "Introduce adjacent categories users would adopt",
         ["Converts existing logs, no new data",
          "e.g. action-movie fan → action game"])
    card(ax2, 0.1, 0.15, 4.8, 1.75, PINK, "Happy surprise",
         "Items outside active category that still land",
         ["Niche / long-tail discovery",
          "Content bridging via SBERT embeddings"])

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
