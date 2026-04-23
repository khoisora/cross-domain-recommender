"""Central style definitions for modernized report figures.

Modern, flat, indigo-leaning palette inspired by Tailwind / Stripe / Linear.
Each color has stroke (border / accent), fill (soft tint background), and text variants.
"""

from __future__ import annotations

FONT = "Lato"
TITLE_FONT = "Lato"
MONO_FONT = "Noto Sans Mono"

# Background and neutrals
BG = "#FFFFFF"
PANEL = "#F8FAFC"
TITLE = "#0F172A"
MUTED = "#64748B"
EDGE = "#94A3B8"
BORDER = "#CBD5E1"

# Semantic palette: 10 colors, each with stroke / fill / text
PALETTE = {
    "indigo":  {"stroke": "#4F46E5", "fill": "#E0E7FF", "text": "#1E1B4B"},
    "sky":     {"stroke": "#0284C7", "fill": "#E0F2FE", "text": "#0C4A6E"},
    "teal":    {"stroke": "#0D9488", "fill": "#CCFBF1", "text": "#134E4A"},
    "emerald": {"stroke": "#059669", "fill": "#D1FAE5", "text": "#064E3B"},
    "amber":   {"stroke": "#D97706", "fill": "#FEF3C7", "text": "#78350F"},
    "rose":    {"stroke": "#E11D48", "fill": "#FFE4E6", "text": "#881337"},
    "violet":  {"stroke": "#7C3AED", "fill": "#EDE9FE", "text": "#3B0764"},
    "slate":   {"stroke": "#475569", "fill": "#F1F5F9", "text": "#0F172A"},
    "pink":    {"stroke": "#DB2777", "fill": "#FCE7F3", "text": "#831843"},
    "cyan":    {"stroke": "#0891B2", "fill": "#CFFAFE", "text": "#164E63"},
}

ORDER = ["indigo", "sky", "teal", "emerald", "amber", "rose", "violet", "slate", "pink", "cyan"]


def color(name: str) -> dict:
    return PALETTE[name]


def node_attrs(color_name: str = "indigo", fontsize: int = 12, bold: bool = True) -> str:
    """Return Graphviz node attribute string for a styled box."""
    c = PALETTE[color_name]
    weight = "bold" if bold else "normal"
    return (
        f'shape=box style="rounded,filled" '
        f'fillcolor="{c["fill"]}" color="{c["stroke"]}" '
        f'fontname="{FONT}" fontsize={fontsize} fontcolor="{c["text"]}" '
        f'penwidth=1.6'
    )


def title_label(title: str, subtitle: str | None = None) -> str:
    """Graphviz HTML-like label for a graph-level title strip."""
    inner = f'<FONT POINT-SIZE="18" COLOR="{TITLE}"><B>{title}</B></FONT>'
    if subtitle:
        inner += f'<BR/><FONT POINT-SIZE="11" COLOR="{MUTED}">{subtitle}</FONT>'
    return f'<{inner}>'


def set_mpl_style() -> None:
    """Apply consistent matplotlib styling globally."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": FONT,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlecolor": TITLE,
        "axes.titlepad": 14,
        "axes.labelsize": 11,
        "axes.labelcolor": TITLE,
        "axes.edgecolor": BORDER,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.dpi": 200,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.labelcolor": TITLE,
    })
