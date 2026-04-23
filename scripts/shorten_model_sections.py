"""Shorten §4.2 and §4.3 model explanations.

For every model in §4.2 (MF-BPR, NCF, LightGCN) and §4.3 (CMF, EMCDR,
PTUPCDR):

• Preserve the Architecture / Training / Inference subsection layout
  and every Figure/Equation caption that was already there.
• Rewrite the body paragraphs to be substantially shorter while
  remaining self-contained and understandable.
• Use proper math notation (U, V, uᵤ, vᵢ, Uᴹ, Vᴳ, ℝ^(|𝒰|×64), α,
  λ, f_θ, …) instead of code-style names like `n_users`, `V_game`.
• Also fix a few pre-existing glitches picked up along the way:
  – stray stub "T" inside NCF Architecture,
  – duplicated/mis-styled headings around §4.3.2,
  – insert an "Architecture" heading for MF-BPR and PTUPCDR so every
    model has the full three-subsection skeleton.

Strategy: target each paragraph by a unique prefix of its current text,
then nuke all runs and append one fresh run with the new text — the
paragraph's pPr (alignment / spacing / style) is preserved, only the
content changes.
"""

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _saved_rpr(p):
    first_r = p._p.find(qn("w:r"))
    if first_r is None:
        return None
    rPr = first_r.find(qn("w:rPr"))
    return deepcopy(rPr) if rPr is not None else None


def _clear_runs(p):
    for r in list(p._p.findall(qn("w:r"))):
        p._p.remove(r)


def _append_run(p, text, rPr=None):
    r = OxmlElement("w:r")
    if rPr is not None:
        r.append(deepcopy(rPr))
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    p._p.append(r)


def _retext(p, text):
    """Replace all runs in p with a single run whose text is `text`,
    preserving the original first-run rPr (font, size, bold-ness)."""
    rPr = _saved_rpr(p)
    _clear_runs(p)
    _append_run(p, text, rPr)


def _set_style(p, style_name):
    try:
        p.style = style_name
    except KeyError:
        pass


def _remove(p):
    p._p.getparent().remove(p._p)


def _find_para(doc, predicate):
    for p in doc.paragraphs:
        if predicate(p):
            return p
    return None


def _find_by_prefix(doc, prefix, style=None):
    for p in doc.paragraphs:
        if style and p.style.name != style:
            continue
        if p.text.startswith(prefix):
            return p
    return None


def _insert_heading_after(anchor, text, style_name="Heading 4"):
    """Insert a new heading paragraph right after `anchor`."""
    new_p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    pStyle = OxmlElement("w:pStyle")
    pStyle.set(qn("w:val"), style_name.replace(" ", ""))
    pPr.append(pStyle)
    new_p.append(pPr)
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    new_p.append(r)
    anchor._p.addnext(new_p)
    # Return a python-docx Paragraph wrapper so caller can restyle.
    from docx.text.paragraph import Paragraph
    out = Paragraph(new_p, anchor._parent)
    _set_style(out, style_name)
    return out


def _insert_para_after(anchor, text):
    """Insert a new body paragraph right after `anchor`, cloning its pPr
    so spacing carries over, then set text."""
    new_p = deepcopy(anchor._p)
    # wipe runs
    for r in list(new_p.findall(qn("w:r"))):
        new_p.remove(r)
    # add fresh text run (no explicit rPr so it inherits style defaults)
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    new_p.append(r)
    anchor._p.addnext(new_p)
    from docx.text.paragraph import Paragraph
    return Paragraph(new_p, anchor._parent)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

# Each entry: (anchor_prefix, new_text).
# anchor_prefix identifies the paragraph to replace by its current text.
REPLACEMENTS = [

    # ===== MF-BPR =====
    # Opening line before Figure 6 — drop it, replaced by Architecture below.
    ("The figure below walks through one training step end-to-end.",
     ""),

    # Rename "MF-BPR: Training process" heading → "MF-BPR: Training".
    ("MF-BPR: Training process",
     "MF-BPR: Training"),

    # Replace the long training paragraph.
    ("The training process initializes every user and item",
     "We train for 60 epochs on sampled triplets (u, i⁺, i⁻), where i⁺ "
     "is an item rated ≥ 4 by u and i⁻ is a uniformly sampled unobserved "
     "item. The BPR loss (Equation 1) maximises the margin uᵤ · vᵢ⁺ − "
     "uᵤ · vᵢ⁻ through a log-sigmoid, with L2 regularisation λ. Optimiser: "
     "plain SGD, learning rate 0.05."),

    # Replace the "BPR Loss. …" intro paragraph with a single short line
    # (the formula itself follows in Equation 1 + the "where" block).
    ("BPR Loss. The Bayesian Personalized Ranking",
     ""),

    # Shorten the "where …" for Equation 1.
    ("where D is the set of training triplets",
     "where 𝒟 is the set of training triplets (u, i⁺, i⁻); uᵤ, vᵢ ∈ ℝ⁶⁴ "
     "are the learnable user/item vectors; σ(·) is the logistic sigmoid; "
     "Θ = {U, V} collects all parameters; λ is the L2 strength."),

    # Drop three filler paragraphs that paraphrase the loss.
    ("When the positive score is much higher than the negative score",
     ""),
    ("When the scores are close or inverted, the loss is large",
     ""),
    ("The regularization term λ prevents vectors from growing too large",
     ""),

    # Shorten the inference paragraph.
    ("Upon completion of the training phase, the model produces",
     "At inference, the scores for user u over the whole catalogue are "
     "sᵤ = uᵤ Vᵀ ∈ ℝ^|ℐ|; we return the top-K items. Artifacts: "
     "U ∈ ℝ^(|𝒰|×64), V ∈ ℝ^(|ℐ|×64)."),

    # Trim strengths / weaknesses to one-liners.
    ("Strengths: Simple, fast to train",
     "Strengths. Simple, fast (≈2 min), a strong baseline."),
    ("Weaknesses: Each user and item is treated independently",
     "Weaknesses. Every user and item is treated in isolation — no "
     "structural prior — and a user without a learned row cannot be "
     "scored (no cold-start support)."),

    # ===== NCF =====
    # Opening Architecture paragraph.
    ("The NCF architecture utilizes two parallel branches",
     "NCF fuses two parallel branches over four independent 64-dim "
     "embedding tables (GMF-user, GMF-item, MLP-user, MLP-item). "
     "The two branch outputs are concatenated and passed through a "
     "sigmoid to produce ŷ ∈ (0, 1)."),

    # GMF branch.
    ("1. Generalized Matrix Factorization (GMF) branch",
     "1. GMF branch. Element-wise product uᵤᴳᴹᶠ ⊙ vᵢᴳᴹᶠ → 64-dim. "
     "Equivalent to a matrix factorisation with per-dimension weights."),

    # Stray "T" stub — drop.
    ("T",
     ""),

    # MLP branch.
    ("2. The MLP (Multi-Layer Perceptron) branch",
     "2. MLP branch. Concatenates uᵤᴹᴸᴾ and vᵢᴹᴸᴾ into a 128-dim input, "
     "then passes through ReLU layers 128 → 64 → 32. The non-linearity "
     "captures patterns a single dot product cannot."),

    # Combined.
    ("3. Combined Architecture. The final NeuMF",
     "3. Fusion. Concatenate the GMF (64-dim) and MLP (32-dim) outputs, "
     "then apply a final linear layer with sigmoid activation to produce "
     "ŷ ∈ (0, 1)."),

    # NCF Training body.
    ("NCF trains using Binary Cross-Entropy (BCE) loss on implicit",
     "NCF minimises binary cross-entropy (Equation 2) over observed pairs "
     "(y = 1) and uniformly sampled negatives (y = 0)."),

    # Eq-2 where.
    ("where D⁺ is the set of observed interactions",
     "where 𝒟⁺ is the observed-interaction set (y_{u,i} = 1) and 𝒟⁻ a "
     "same-size set of uniformly sampled negatives (y_{u,i} = 0); "
     "ŷ_{u,i} ∈ (0, 1) is the NeuMF sigmoid output from the fused "
     "GMF + MLP representation. A confident wrong prediction — e.g. "
     "ŷ = 0.9 on a true negative — contributes −log(0.1) ≈ 2.3 to the "
     "loss."),

    # Drop the BCE paraphrase.
    ("The BCE loss penalizes confident wrong predictions heavily",
     ""),

    # NCF Inference.
    ("The trained artifacts include 4 embedding matrices",
     "Scoring any (u, i) requires a forward pass through the fused "
     "network — there is no single matrix multiply. NCF is therefore "
     "slower than MF-BPR at inference but more expressive."),

    # ===== LightGCN =====
    # Architecture opening.
    ("Graph Propagation. Users and items form a bipartite graph",
     "Users and items form a bipartite graph G = (𝒰 ∪ ℐ, 𝐄). Each node v "
     "has a learnable layer-0 embedding eᵥ⁽⁰⁾ ∈ ℝ⁹⁶. For K = 3 hops we "
     "propagate with symmetric normalisation: "
     "eᵥ⁽ᵏ⁺¹⁾ = Σ_{u ∈ 𝒩(v)} eᵤ⁽ᵏ⁾ / √(|𝒩(u)| · |𝒩(v)|)."),

    # After K hops.
    ("After K hops, each node has K+1 representations",
     "The final representation is the layer mean "
     "ẽᵥ = (1/(K+1)) Σ_{k=0}^K eᵥ⁽ᵏ⁾. Averaging layers 0 – K balances "
     "the node itself (k = 0), direct neighbours (k = 1) and "
     "multi-hop community signal."),

    # Training.
    ("LightGCN uses the same BPR loss as MF-BPR",
     "Same pairwise BPR as MF-BPR (Equation 3), but scores are computed "
     "with the propagated ẽᵤ, ẽᵢ. Gradients flow back only to the "
     "layer-0 embeddings 𝐄⁽⁰⁾ — the propagation itself has no "
     "learnable parameters."),

    # Eq-3 where.
    ("where e_v^{(k)} is node v's embedding",
     "where eᵥ⁽ᵏ⁾ is node v's embedding after k propagation steps, "
     "ẽᵥ is the final layer-mean representation (K = 3), and the only "
     "trainable parameters are the layer-0 embeddings 𝐄⁽⁰⁾. The BPR "
     "objective is applied to the propagated user/item vectors."),

    # Inference.
    ("After training, we pre-compute the final embeddings",
     "We pre-compute ẽᵤ, ẽᵢ for every node once and cache to disk. "
     "Inference is then identical to MF-BPR: a single dot product "
     "ẽᵤ · ẽᵢ per candidate item, ranked by score."),

    # ===== CMF =====
    ("Two embedding spaces, one shared user identity",
     "Shared user matrix. CMF keeps one user matrix U ∈ ℝ^(|𝒰|×96) shared "
     "by both domains, plus domain-specific item matrices "
     "Vᴹ ∈ ℝ^(|ℐᴹ|×96) for movies and Vᴳ ∈ ℝ^(|ℐᴳ|×96) for games. "
     "Every user has exactly one 96-dim vector used in both domains."),

    ("Item matrices stay private.",
     "Item matrices stay private. Vᴹ receives only movie gradients, "
     "Vᴳ only game gradients; the shared U is therefore the sole "
     "transfer channel between the two domains."),

    ("Why 96 dimensions.",
     "Width. We swept embedding widths {32, 64, 96, 128} and picked 96 — "
     "large enough to carry both tastes, small enough not to decouple "
     "into independent sub-spaces."),

    ("Sampling. Each training step draws one BPR triplet per domain",
     "Sampling. Each step draws one BPR triplet per domain: "
     "(u, iᴹ⁺, iᴹ⁻) and (u, iᴳ⁺, iᴳ⁻) with uniform negatives. Users "
     "present in only one domain still contribute — their row of U is "
     "pulled by that domain's triplets alone."),

    ("Joint objective. The loss is a weighted sum",
     "Joint objective. The total loss is a convex combination of the two "
     "per-domain BPR terms sharing U (Equation 4). α ∈ [0, 1] trades "
     "movie signal (α → 1) for game signal (α → 0); our default is "
     "α = 0.5."),

    ("where U is the shared user embedding matrix used by both domains",
     "where U is the shared user matrix; Vᴹ, Vᴳ are the domain-specific "
     "item matrices; L_BPRᴹ, L_BPRᴳ are the per-domain BPR losses "
     "(same form as Equation 1); α ∈ [0, 1] balances the two domains; "
     "Θ = {U, Vᴹ, Vᴳ}; λ is the L2 strength."),

    ("Optimisation. We initialise U, V_m, V_g with Gaussian noise",
     "Optimisation. SGD, learning rate 0.05, L2 λ = 1e-5, 30 epochs. "
     "Figure 13 visualises one step — the same row of U is pulled "
     "simultaneously by a movie and a game triplet."),

    # CMF Inference body.
    ("The trained artifacts are U, V_movie, and V_game.",
     "Artifacts: U, Vᴹ, Vᴳ. For user u and game g, "
     "score(u, g) = uᵤ · v_g — identical to MF-BPR. Vᴹ is never touched "
     "at inference; it has done its job by shaping U during training. "
     "A cold-start game user with movie history still gets useful "
     "scores because uᵤ carries movie-derived signal."),

    # Drop the Figure 13b paraphrase (caption itself stays).
    ("Figure 13b walks through the scoring path for a single user",
     ""),

    # CMF Limitations.
    ("CMF's \"one profile fits all\" approach is a blunt instrument.",
     "The shared uᵤ is a compromise vector that must serve both domains "
     "and is therefore mediocre at each. In our experiments CMF produces "
     "the weakest CDR transfer — especially at cold-start, where uᵤ has "
     "been shaped almost entirely by movie data (≈95% of interactions)."),

    # ===== EMCDR =====
    # Keep analogy paragraph intact.
    # Architecture three blocks (shortened).
    ("Two independent embedding spaces. EMCDR keeps the two domains",
     "Two independent spaces. EMCDR trains one MF-BPR per domain: "
     "Uᴹ, Vᴹ for movies and Uᴳ, Vᴳ for games. The same user occupies "
     "both Uᴹ and Uᴳ as two unrelated 64-dim vectors (uᵤᴹ ≠ uᵤᴳ)."),

    ("A learned translator. The translator bridges them.",
     "A learned translator. A small MLP f_θ with two hidden layers of "
     "width 128 and ReLU maps movie-space to game-space: "
     "f_θ(uᵤᴹ) ≈ uᵤᴳ. Once fit, any user with a movie-side vector — "
     "including cold-start users — can be scored against Vᴳ."),

    ("Supervised by overlap. Supervision comes from overlap users only.",
     "Supervised by overlap. f_θ is trained only on overlap users "
     "𝒰ₒ = {u : u has interactions in both domains}. In our data "
     "|𝒰ₒ| = 19,880 — without overlap the translator has no signal."),

    # EMCDR Training blocks already short — leave as is.

    # Eq-5 where (minor tidy).
    ("where Uoverlap is the set of users with interactions",
     "where 𝒰ₒ is the set of overlap users; uᵤᴹ, uᵤᴳ are the user "
     "embeddings learned independently in Phases 1 and 2; f_θ is the "
     "MLP translator fit in Phase 3 by regressing the game-side vector "
     "from the movie-side vector."),

    # EMCDR Inference body.
    ("The artifacts that leave training are VG, UM, and the mapping",
     "Artifacts at inference: Vᴳ, Uᴹ, f_θ. For user u and game g, "
     "score(u, g) = f_θ(uᵤᴹ) · v_g. Uᴳ is never touched, so cold-start "
     "users (zero game history) are natively supported — this is why "
     "EMCDR shines in Lesson 6."),

    # ===== PTUPCDR =====
    # The existing "PTUPCDR Training" heading stays; its body is rewritten.
    ("Like EMCDR, PTUPCDR starts with Phase 1 (movie MF-BPR)",
     "Phases 1 and 2 are identical to EMCDR: train MF-BPR independently "
     "per domain to obtain Uᴹ, Vᴹ, Uᴳ, Vᴳ. Phase 3 then freezes both "
     "and fits the K-expert MoE by minimising the Equation 6 MSE on "
     "overlap users only (Adam, lr = 1e-3, 200 epochs)."),

    # Eq-6 where.
    ("where each f_k is one of K expert MLPs",
     "where f_k is one of K expert MLPs (K = 8) and "
     "gₖ = softmax(W_g · uᵤᴹ)_k is the corresponding gate weight. Each "
     "user's movie-side vector is routed through a different expert "
     "mix, capturing user-specific cross-domain patterns that a single "
     "global f_θ (EMCDR) cannot."),

    # PTUPCDR Inference body.
    ("The saved artifacts include V_game, the 8 expert MLPs",
     "Artifacts: Vᴳ, {f_k}, W_g, blend weight w. For user u: compute "
     "gate weights from uᵤᴹ, run each expert on uᵤᴹ, weight-sum to "
     "F(uᵤᴹ) = Σ_k gₖ · f_k(uᵤᴹ), then score every game by "
     "F(uᵤᴹ) · v_g. If the user already has game history, "
     "F(uᵤᴹ) is linearly blended with their learned uᵤᴳ by w."),
]


# New subsection headings to be inserted where missing.
# Each tuple: (anchor_prefix_of_H3, heading_text, body_text).
# The Architecture block is inserted right after the section's H3 heading.
NEW_ARCH_BLOCKS = [
    # MF-BPR: insert "MF-BPR: Architecture" + body before Figure 6 / Training.
    (
        "4.2.1 MF-BPR: Matrix Factorization",
        "MF-BPR: Architecture",
        "Each user u and item i has a single learnable 64-dim vector, "
        "uᵤ, vᵢ ∈ ℝ⁶⁴. The score for a (u, i) pair is the dot product "
        "s(u, i) = uᵤ · vᵢ. All parameters live in two matrices "
        "U ∈ ℝ^(|𝒰|×64) and V ∈ ℝ^(|ℐ|×64).",
    ),
    # PTUPCDR: insert "PTUPCDR Architecture" + body before Training.
    (
        "4.3.3 PTUPCDR: Personalized Transfer",
        "PTUPCDR Architecture",
        "PTUPCDR extends EMCDR by replacing the single global translator "
        "f_θ with a personalised mixture of K = 8 experts. Each expert "
        "is a small MLP f_k : ℝ⁶⁴ → ℝ⁶⁴. A gate network "
        "g(uᵤᴹ) = softmax(W_g · uᵤᴹ) ∈ ℝᴷ produces per-user expert "
        "weights gₖ. The personalised translation is "
        "F(uᵤᴹ) = Σ_k gₖ(uᵤᴹ) · f_k(uᵤᴹ) — different users are routed "
        "through different expert combinations, so the mapping is "
        "user-specific rather than global as in EMCDR.",
    ),
]


# Paragraphs to delete outright (junk / duplicates introduced by prior edits).
DELETE_BY_EXACT = [
    # stray Heading 3 "Collective Matrix Factorization" under §4.3.2
    ("Collective Matrix Factorization", "Heading 3"),
    # stray Normal "EMCDR Architecture" (should be the H4 below, which
    # already exists in text form — we'll turn it into a proper H4 instead).
]


# Paragraphs whose *style* needs to change (content left alone).
RESTYLE = [
    # "EMCDR Architecture" currently rendered as Normal — promote to H4.
    ("EMCDR Architecture", "Heading 4"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    doc = Document(str(DOC))

    # ---- 1. Apply in-place text replacements. ----------------------------
    # Collect applied flags so we only touch each paragraph once.
    applied = set()
    for prefix, new_text in REPLACEMENTS:
        target = None
        for p in doc.paragraphs:
            if id(p._p) in applied:
                continue
            if p.text.startswith(prefix):
                target = p
                break
        if target is None:
            print(f"  [skip] prefix not found: {prefix[:60]!r}")
            continue
        if new_text == "":
            _remove(target)
        else:
            _retext(target, new_text)
        applied.add(id(target._p))

    # ---- 2. Delete junk paragraphs. --------------------------------------
    for exact, style in DELETE_BY_EXACT:
        target = None
        for p in doc.paragraphs:
            if p.text.strip() == exact and p.style.name == style:
                target = p
                break
        if target is not None:
            _remove(target)
            print(f"  [del]  {exact!r}")

    # ---- 3. Fix mis-styled headings. -------------------------------------
    for exact, new_style in RESTYLE:
        for p in doc.paragraphs:
            if p.text.strip() == exact:
                _set_style(p, new_style)
                print(f"  [style] {exact!r} → {new_style}")
                break

    # ---- 4. Insert missing Architecture headings + bodies. ---------------
    for h3_prefix, heading_text, body_text in NEW_ARCH_BLOCKS:
        # Skip if heading already exists.
        already = _find_para(doc, lambda p, ht=heading_text:
                             p.text.strip() == ht
                             and p.style.name == "Heading 4")
        if already is not None:
            print(f"  [skip] heading already present: {heading_text!r}")
            continue

        h3 = _find_by_prefix(doc, h3_prefix, style="Heading 3")
        if h3 is None:
            print(f"  [warn] H3 anchor not found: {h3_prefix!r}")
            continue

        # Insert order: Body first (so it lands right after heading after
        # the heading is inserted).
        # Find first non-empty paragraph after H3 — we insert heading AFTER
        # that blank spacing paragraph so it lands cleanly.
        anchor = h3
        heading_p = _insert_heading_after(anchor, heading_text,
                                          style_name="Heading 4")
        _insert_para_after(heading_p, body_text)
        print(f"  [new]  {heading_text!r}")

    # ---- 5. Save. --------------------------------------------------------
    doc.save(str(DOC))
    print("✓ §4.2 / §4.3 model sections shortened")


if __name__ == "__main__":
    main()
