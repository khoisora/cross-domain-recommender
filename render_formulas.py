#!/usr/bin/env python3
"""
render_formulas.py
Renders all 8 formula callout-box equations as academic-quality PNG images
using matplotlib mathtext (STIX fonts).  No external LaTeX installation needed.

Output: report_figures/formulas/formula_<name>.png
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})

OUTPUT_DIR = "report_figures/formulas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colours (transparent background — inherits outer Word callout box colour) ──
TITLE_COLOR   = '#1B2A4A'   # navy – matches COLORS["navy"] in generate_report.py
FORMULA_COLOR = '#1B2A4A'   # same navy for formulas
NOTE_COLOR    = '#2D3748'   # body-text grey


# ── Core renderer ────────────────────────────────────────────────────────────
def render_png(name, title, blocks, width_in=9.5):
    """
    blocks : list of dict with keys
        text      – string (use $...$ for inline math)
        mode      – 'formula' | 'note' | 'subhead'
        align     – 'center' | 'left'  (default 'center' for formula, 'left' for note)
        fs        – font size override (optional)
    """
    # measure height: 0.55 in per block + 0.65 in header
    height_in = 0.65 + len(blocks) * 0.62 + 0.25

    # Transparent figure so there is no inner box when embedded in Word
    fig = plt.figure(figsize=(width_in, height_in))
    fig.patch.set_alpha(0)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.patch.set_alpha(0)   # transparent axes background too

    # No accent bar in the PNG — the Word table cell already has one

    total_rows = 1 + len(blocks)   # title row + formula rows
    row_h = 1.0 / (total_rows + 0.5)

    # ── title row ─────────────────────────────────────────────────────────────
    y_title = 1.0 - row_h * 0.55
    ax.text(0.03, y_title, title,
            transform=ax.transAxes,
            fontsize=13, fontweight='bold', color=TITLE_COLOR,
            va='center', ha='left', zorder=4)

    # ── formula / note rows ───────────────────────────────────────────────────
    for idx, blk in enumerate(blocks):
        y = 1.0 - row_h * (1.4 + idx * 1.05)

        mode   = blk.get('mode', 'formula')
        text   = blk['text']
        align  = blk.get('align', 'center' if mode == 'formula' else 'left')
        x_pos  = 0.52 if align == 'center' else 0.035
        ha     = 'center' if align == 'center' else 'left'

        if mode == 'formula':
            fs    = blk.get('fs', 14)
            color = FORMULA_COLOR
            style = 'normal'
            fw    = 'normal'
        elif mode == 'subhead':
            fs    = blk.get('fs', 12)
            color = TITLE_COLOR
            style = 'normal'
            fw    = 'bold'
        else:   # note
            fs    = blk.get('fs', 11)
            color = NOTE_COLOR
            style = 'italic'
            fw    = 'normal'

        ax.text(x_pos, y, text,
                transform=ax.transAxes,
                fontsize=fs, color=color,
                fontweight=fw, fontstyle=style,
                va='center', ha=ha, zorder=4)

    out_path = os.path.join(OUTPUT_DIR, f"formula_{name}.png")
    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                transparent=True)   # transparent PNG — no inner box
    plt.close(fig)
    print(f"  ✓  {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Formula definitions
# ═══════════════════════════════════════════════════════════════════════════════

def formula_bpr():
    render_png(
        name="bpr",
        title="BPR Loss Function",
        blocks=[
            dict(text=r"$\mathcal{L}_{\mathrm{BPR}} = -\log\,\sigma\!\left(\mathbf{u}^{\top}\mathbf{i}^{+} - \mathbf{u}^{\top}\mathbf{i}^{-}\right) + \lambda\!\left(\|\mathbf{u}\|^{2} + \|\mathbf{i}^{+}\|^{2} + \|\mathbf{i}^{-}\|^{2}\right)$",
              mode='formula', fs=13),
            dict(text=(r"$\sigma$ = sigmoid; $\mathbf{u}^{\top}\mathbf{i}^{+}$ = user–positive score; "
                       r"$\mathbf{u}^{\top}\mathbf{i}^{-}$ = user–negative score; $\lambda$ = L2 regularisation weight"),
              mode='note', align='left', fs=11),
        ]
    )


def formula_ncf():
    render_png(
        name="ncf",
        title="NCF Loss Function (BCE)",
        blocks=[
            dict(text=r"$\hat{y} = \sigma\!\left(\mathbf{h}^{\top}\left[\,\mathrm{GMF}(u,i)\;;\;\mathrm{MLP}(u,i)\,\right]\right)$",
              mode='formula', fs=13),
            dict(text=r"$\mathcal{L} = -\sum_{(u,i)}\!\left[\,y\log\hat{y} + (1-y)\log(1-\hat{y})\,\right]$",
              mode='formula', fs=13),
            dict(text=(r"$y=1$ for observed interactions; $y=0$ for sampled negatives.  "
                       r"Adam optimiser, lr $= 0.001$, 150 epochs."),
              mode='note', align='left', fs=11),
        ]
    )


def formula_lightgcn():
    render_png(
        name="lightgcn",
        title="LightGCN Layer Propagation",
        blocks=[
            dict(text=r"$\mathbf{e}_{u}^{(k)} = \sum_{i\in\mathcal{N}_{u}} \frac{1}{\sqrt{|\mathcal{N}_{u}|}\,\sqrt{|\mathcal{N}_{i}|}}\;\mathbf{e}_{i}^{(k-1)}$",
              mode='formula', fs=13),
            dict(text=r"$\mathbf{e}_{u}^{*} = \frac{1}{K+1}\sum_{k=0}^{K}\mathbf{e}_{u}^{(k)}$",
              mode='formula', fs=13),
            dict(text=(r"Symmetric sqrt-degree normalisation prevents popular items from dominating.  "
                       r"$K=3$ hops, 96-dim embeddings."),
              mode='note', align='left', fs=11),
        ]
    )


def formula_cmf():
    render_png(
        name="cmf",
        title="CMF Joint Loss",
        blocks=[
            dict(text=r"$\mathcal{L}_{\mathrm{total}} = \alpha\;\mathcal{L}_{\mathrm{BPR}}^{\mathrm{movie}}(\mathbf{U},\mathbf{V}_{\mathrm{movie}}) + (1-\alpha)\;\mathcal{L}_{\mathrm{BPR}}^{\mathrm{game}}(\mathbf{U},\mathbf{V}_{\mathrm{game}})$",
              mode='formula', fs=13),
            dict(text=(r"$\alpha = 0.05$ (5 % movie, 95 % game).  Shared $\mathbf{U}$ receives gradients from both domains; "
                       r"$\mathbf{V}_{\mathrm{movie}}$ and $\mathbf{V}_{\mathrm{game}}$ update from their respective domains only."),
              mode='note', align='left', fs=11),
        ]
    )


def formula_emcdr():
    render_png(
        name="emcdr",
        title="EMCDR – Three-Phase Training",
        blocks=[
            dict(text=r"Phase 1:  Train movie MF-BPR  $\Rightarrow$  $\mathbf{U}_{\mathrm{movie}},\;\mathbf{V}_{\mathrm{movie}}$",
              mode='subhead', align='left', fs=12),
            dict(text=r"Phase 2:  Train game MF-BPR   $\Rightarrow$  $\mathbf{U}_{\mathrm{game}},\;\mathbf{V}_{\mathrm{game}}$",
              mode='subhead', align='left', fs=12),
            dict(text=r"Phase 3 (mapping loss on overlap users $\mathcal{O}$):",
              mode='subhead', align='left', fs=12),
            dict(text=r"$\mathcal{L}_{\mathrm{map}} = \frac{1}{|\mathcal{O}|}\sum_{u\in\mathcal{O}} \left\|f\!\left(\mathbf{U}_{\mathrm{movie}}[u]\right) - \mathbf{U}_{\mathrm{game}}[u]\right\|^{2}$",
              mode='formula', fs=13),
            dict(text=r"MLP $f$: one hidden layer $64\!\to\!64$, ReLU activation, MSE loss, Adam optimiser.",
              mode='note', align='left', fs=11),
        ]
    )


def formula_ptupcdr():
    render_png(
        name="ptupcdr",
        title="PTUPCDR – Bridge and Blend",
        blocks=[
            dict(text=r"$\mathrm{bridge}(u) = \sum_{k=1}^{8}\mathrm{gate}_{k}(\mathbf{u}_{\mathrm{movie}})\cdot\mathrm{Expert}_{k}(\mathbf{u}_{\mathrm{movie}})$",
              mode='formula', fs=13),
            dict(text=r"$\mathrm{gate}(\mathbf{u}_{\mathrm{movie}}) = \mathrm{softmax}(\mathbf{W}_{\mathrm{gate}}\,\mathbf{u}_{\mathrm{movie}})$   $\Rightarrow$   8 weights summing to 1",
              mode='formula', fs=12),
            dict(text=r"$\mathrm{emb}_{\mathrm{final}}(u) = w\cdot\mathrm{bridge}(\mathbf{u}_{\mathrm{movie}}) + (1-w)\cdot\mathbf{u}_{\mathrm{game}}$",
              mode='formula', fs=13),
            dict(text=(r"$w=1$ for cold-start users (no $\mathbf{u}_{\mathrm{game}}$).  "
                       r"Minimises MSE between $\mathrm{emb}_{\mathrm{final}}$ and $\mathbf{u}_{\mathrm{game}}$ over overlap users."),
              mode='note', align='left', fs=11),
        ]
    )


def formula_sbert():
    render_png(
        name="sbert",
        title="SBERT Scoring (No Training Required)",
        blocks=[
            dict(text=r"$\mathbf{e}_{i} = \mathrm{SBERT}(\mathrm{''title: [t].\ desc: [d]''})\;\in\;\mathbb{R}^{384}$",
              mode='formula', fs=13),
            dict(text=r"$\mathbf{p}_{u} = \frac{1}{|\mathcal{H}_{u}|}\sum_{i\in\mathcal{H}_{u}}\mathbf{e}_{i}$",
              mode='formula', fs=13),
            dict(text=r"$\mathrm{score}(u,g) = \cos(\mathbf{p}_{u},\mathbf{e}_{g}) = \frac{\mathbf{p}_{u}\cdot\mathbf{e}_{g}}{\|\mathbf{p}_{u}\|\;\|\mathbf{e}_{g}\|}$",
              mode='formula', fs=13),
            dict(text=(r"$\mathcal{H}_{u}$ = user's interaction history.  "
                       r"Pre-trained encoder used as-is — zero fine-tuning required."),
              mode='note', align='left', fs=11),
        ]
    )


def formula_cooc():
    render_png(
        name="cooc",
        title="Co-occurrence Scoring Formula",
        blocks=[
            dict(text=r"$\mathrm{cooc}(m,g) = \frac{|\mathrm{users\ who\ rated\ both}|}{\sqrt{|\mathrm{rated}\ m|\cdot|\mathrm{rated}\ g|}}$",
              mode='formula', fs=13),
            dict(text=r"$\mathrm{bonus}(u,g) = \sum_{m\in\mathcal{M}_{u}}\mathrm{cooc}(m,g)$",
              mode='formula', fs=13),
            dict(text=r"$\mathrm{score}_{\mathrm{final}}(g) = \mathrm{score}_{\mathrm{base}}(g) + \lambda\cdot\mathrm{bonus}(u,g)$",
              mode='formula', fs=13),
            dict(text=(r"$\mathcal{M}_{u}$ = user's rated movies; $\lambda$ tuned per model via grid search.  "
                       r"No training — computed directly from interaction counts."),
              mode='note', align='left', fs=11),
        ]
    )


# ── run all ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Rendering formula PNGs …")
    formula_bpr()
    formula_ncf()
    formula_lightgcn()
    formula_cmf()
    formula_emcdr()
    formula_ptupcdr()
    formula_sbert()
    formula_cooc()
    print("Done — all formula images saved to", OUTPUT_DIR)
