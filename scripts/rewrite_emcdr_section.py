"""Rewrite §4.3.2 EMCDR: Cross-Domain Embedding Mapping.

Actions
-------
1. Insert a new "EMCDR Architecture" H4 right after the §4.3.2 H3. It
   carries two–three labelled body paragraphs that describe the three
   building blocks (two independent MF-BPR spaces + the MLP translator)
   and closes with fig14a_emcdr_architecture.png.
2. Shorten "EMCDR Training" to a compact three-phase write-up that uses
   proper math notation (U^M, V^M, u_u^M, V^G, f_θ, …) instead of the
   old "U_movie / V_movie" text. Equation 5 + its "where …" explanation
   stay here, and Figure 14 (three-phase diagram) is re-anchored below
   the formula.
3. Rewrite "EMCDR Inference" with the same tight math notation and
   append fig14b_emcdr_inference.png + caption.

Only content under §4.3.2 is touched; adjacent sections are untouched.
"""

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"
FIGS = ROOT / "report_figures_v3"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
IMAGE_WIDTH_IN = 5.8
FORMULA_WIDTH_IN = 5.8


# ---- helpers --------------------------------------------------------

def _find_heading(doc, text):
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading") and p.text.strip() == text:
            return p
    raise ValueError(f"not found: {text!r}")


def _delete_between(start_heading, stop_heading):
    parent = start_heading._p.getparent()
    cur = start_heading._p.getnext()
    while cur is not None and cur is not stop_heading._p:
        nxt = cur.getnext()
        parent.remove(cur)
        cur = nxt


def _blank_p_after(anchor, alignment=None, space_before=None,
                   space_after=None, line_spacing=1.35,
                   style_name="normal"):
    new_p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    pStyle = OxmlElement("w:pStyle")
    pStyle.set(qn("w:val"), style_name)
    pPr.append(pStyle)
    new_p.append(pPr)
    anchor._p.addnext(new_p)
    para = Paragraph(new_p, anchor._parent)
    if alignment is not None:
        para.alignment = alignment
    if space_before is not None:
        para.paragraph_format.space_before = space_before
    if space_after is not None:
        para.paragraph_format.space_after = space_after
    para.paragraph_format.line_spacing = line_spacing
    return para


def _blank_heading_after(anchor, text, level=4):
    new_p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    pStyle = OxmlElement("w:pStyle")
    pStyle.set(qn("w:val"), f"Heading{level}")
    pPr.append(pStyle)
    new_p.append(pPr)
    anchor._p.addnext(new_p)
    para = Paragraph(new_p, anchor._parent)
    para.add_run(text)
    return para


def _fix_run_font(run):
    rPr = run._element.find(qn("w:rPr"))
    if rPr is None:
        rPr = OxmlElement("w:rPr")
        run._element.insert(0, rPr)
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    for attr in ("ascii", "hAnsi", "cs"):
        rFonts.set(qn(f"w:{attr}"), "Arial")


def _add_plain(para, text, size_pt=10, bold=False, italic=False):
    run = para.add_run(text)
    run.font.name = "Arial"
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    _fix_run_font(run)
    return run


def _add_math(para, text, size_pt=10, bold=False, italic=True,
              vert=None):
    """Add an Arial run for a math fragment. `vert` ∈ {None,'superscript',
    'subscript'}."""
    run = _add_plain(para, text, size_pt=size_pt, bold=bold, italic=italic)
    if vert is not None:
        rPr = run._element.find(qn("w:rPr"))
        vAlign = OxmlElement("w:vertAlign")
        vAlign.set(qn("w:val"), vert)
        rPr.append(vAlign)
    return run


def _add_label_para(anchor, label, body_builder):
    """Insert a paragraph after `anchor` shaped as: **Label.** <body>.
    `body_builder(para)` receives the paragraph and is expected to append
    its own runs to it. Returns the new paragraph."""
    p = _blank_p_after(anchor, space_before=Pt(0), space_after=Pt(6))
    _add_plain(p, label + " ", bold=True)
    body_builder(p)
    return p


def _insert_image(anchor, png_path, width_in):
    p = _blank_p_after(anchor, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                       space_before=Pt(6), space_after=Pt(0),
                       line_spacing=1.15)
    p.add_run().add_picture(str(png_path), width=Inches(width_in))
    return p


def _insert_caption(anchor, text):
    p = _blank_p_after(anchor, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                       space_before=Pt(0), space_after=Pt(6))
    _add_plain(p, text, italic=True)
    return p


def _insert_body(anchor, builder):
    p = _blank_p_after(anchor, space_before=Pt(0), space_after=Pt(6))
    builder(p)
    return p


# ---- small inline-math builders ------------------------------------
# Each builder appends runs to an existing paragraph, using italic
# letters for variables and Word super/subscript runs where useful.

def _bold_vec_with_super(para, base_italic, super_str):
    """Render e.g. **u**_{u}^{M} as bold-italic base + super text."""
    _add_plain(para, base_italic, bold=True, italic=True)
    _add_math(para, super_str, italic=False, vert="superscript")


def _var_with_super(para, base, super_str, bold=False):
    _add_plain(para, base, bold=bold, italic=True)
    _add_math(para, super_str, italic=False, vert="superscript")


def _var_with_sub(para, base, sub_str, bold=False):
    _add_plain(para, base, bold=bold, italic=True)
    _add_math(para, sub_str, italic=False, vert="subscript")


def _var_sub_super(para, base, sub_str, super_str, bold=False):
    _add_plain(para, base, bold=bold, italic=True)
    _add_math(para, sub_str, italic=False, vert="subscript")
    _add_math(para, super_str, italic=False, vert="superscript")


# ---- content builders ----------------------------------------------

def arch_para_1(p):
    _add_plain(p,
        "EMCDR keeps the two domains on completely separate feet. "
        "Each domain trains its own MF-BPR model end-to-end, producing "
        "an independent user matrix and item matrix: ")
    _var_with_super(p, "U", "M", bold=True)
    _add_plain(p, ", ")
    _var_with_super(p, "V", "M", bold=True)
    _add_plain(p, " for movies and ")
    _var_with_super(p, "U", "G", bold=True)
    _add_plain(p, ", ")
    _var_with_super(p, "V", "G", bold=True)
    _add_plain(p, " for games. The two user matrices encode the "
        "same users but in two completely unrelated 64-dimensional "
        "spaces — row ")
    _var_sub_super(p, "u", "u", "M", bold=True)
    _add_plain(p, " and row ")
    _var_sub_super(p, "u", "u", "G", bold=True)
    _add_plain(p, " are not comparable out of the box.")


def arch_para_2(p):
    _add_plain(p,
        "The translator bridges them. A small multi-layer perceptron ")
    _add_plain(p, "f", italic=True)
    _add_math(p, "θ", italic=False, vert="subscript")
    _add_plain(p, " (two hidden layers of width 128 with ReLU) maps a "
        "movie-space vector to a game-space vector: ")
    _add_plain(p, "f", italic=True)
    _add_math(p, "θ", italic=False, vert="subscript")
    _add_plain(p, "(")
    _var_sub_super(p, "u", "u", "M", bold=True)
    _add_plain(p, ") ≈ ")
    _var_sub_super(p, "u", "u", "G", bold=True)
    _add_plain(p, ". Once ")
    _add_plain(p, "f", italic=True)
    _add_math(p, "θ", italic=False, vert="subscript")
    _add_plain(p,
        " is fit, any user who has a movie-side vector — including "
        "cold-start users with zero game history — can be translated "
        "into game space and scored against ")
    _var_with_super(p, "V", "G", bold=True)
    _add_plain(p, ".")


def arch_para_3(p):
    _add_plain(p, "Supervision comes from overlap users only. "
        "The translator is fit by minimising the squared distance "
        "between ")
    _add_plain(p, "f", italic=True)
    _add_math(p, "θ", italic=False, vert="subscript")
    _add_plain(p, "(")
    _var_sub_super(p, "u", "u", "M", bold=True)
    _add_plain(p, ") and ")
    _var_sub_super(p, "u", "u", "G", bold=True)
    _add_plain(p, " over the set of overlap users ")
    _var_with_sub(p, "U", "overlap")
    _add_plain(p,
        " (those with interactions in both domains). The 19,880 "
        "overlap users in our data are what makes this supervision "
        "possible — without them there is no signal to align the two "
        "embedding spaces.")


ARCHITECTURE_BLOCKS = [
    ("Two independent embedding spaces.", arch_para_1),
    ("A learned translator.", arch_para_2),
    ("Supervised by overlap.", arch_para_3),
]


def training_para_phase1(p):
    _add_plain(p, "Train MF-BPR on movies for 20 epochs to get ")
    _var_with_super(p, "U", "M", bold=True)
    _add_plain(p, " and ")
    _var_with_super(p, "V", "M", bold=True)
    _add_plain(p, " (64-dim each).")


def training_para_phase2(p):
    _add_plain(p, "Independently train a second MF-BPR on games for 20 "
        "epochs to get ")
    _var_with_super(p, "U", "G", bold=True)
    _add_plain(p, " and ")
    _var_with_super(p, "V", "G", bold=True)
    _add_plain(p, ". No parameters are shared with Phase 1.")


def training_para_phase3(p):
    _add_plain(p, "Freeze both MF-BPR models. Using the overlap users "
        "only, fit ")
    _add_plain(p, "f", italic=True)
    _add_math(p, "θ", italic=False, vert="subscript")
    _add_plain(p, " to regress ")
    _var_sub_super(p, "u", "u", "G", bold=True)
    _add_plain(p, " from ")
    _var_sub_super(p, "u", "u", "M", bold=True)
    _add_plain(p, " with the MSE loss in Equation 5. Adam, lr = 1e-3, "
        "200 epochs, early stopping on a 10% overlap-user validation "
        "split.")


TRAINING_PHASE_BLOCKS = [
    ("Phase 1 — movie MF-BPR.", training_para_phase1),
    ("Phase 2 — game MF-BPR.", training_para_phase2),
    ("Phase 3 — fit the translator.", training_para_phase3),
]


def eq5_where(p):
    _add_plain(p, "where ")
    _var_with_sub(p, "U", "overlap")
    _add_plain(p, " is the set of users with interactions in both "
        "domains; ")
    _var_sub_super(p, "u", "u", "M", bold=True)
    _add_plain(p, " and ")
    _var_sub_super(p, "u", "u", "G", bold=True)
    _add_plain(p, " are the user embeddings learned independently in "
        "Phases 1 and 2; ")
    _add_plain(p, "f", italic=True)
    _add_math(p, "θ", italic=False, vert="subscript")
    _add_plain(p, " is the MLP translator fit in Phase 3 by regressing "
        "the game-side vector from the movie-side vector.")


def inference_para(p):
    _add_plain(p,
        "The artifacts that leave training are ")
    _var_with_super(p, "V", "G", bold=True)
    _add_plain(p, ", ")
    _var_with_super(p, "U", "M", bold=True)
    _add_plain(p, ", and the mapping network ")
    _add_plain(p, "f", italic=True)
    _add_math(p, "θ", italic=False, vert="subscript")
    _add_plain(p, ". For any user ")
    _add_plain(p, "u", italic=True)
    _add_plain(p, " we look up ")
    _var_sub_super(p, "u", "u", "M", bold=True)
    _add_plain(p, ", push it through the translator to get ")
    _add_plain(p, "f", italic=True)
    _add_math(p, "θ", italic=False, vert="subscript")
    _add_plain(p, "(")
    _var_sub_super(p, "u", "u", "M", bold=True)
    _add_plain(p, "), then score every game by the dot product ")
    _add_plain(p, "f", italic=True)
    _add_math(p, "θ", italic=False, vert="subscript")
    _add_plain(p, "(")
    _var_sub_super(p, "u", "u", "M", bold=True)
    _add_plain(p, ") · ")
    _var_with_sub(p, "v", "g", bold=True)
    _add_plain(p, ". Because the scoring path never touches ")
    _var_with_super(p, "U", "G", bold=True)
    _add_plain(p,
        ", EMCDR is a natural fit for cold-start users — any user with "
        "a movie-side vector can be served, even if they have zero "
        "game interactions (Lesson 6).")


# ---- main -----------------------------------------------------------

def main():
    doc = Document(str(DOC))

    h_emcdr = _find_heading(doc, "4.3.2 EMCDR: Cross-Domain Embedding Mapping")
    h_train = _find_heading(doc, "EMCDR Training")
    h_infer = _find_heading(doc, "EMCDR Inference")
    # Lookup for the subsequent heading so we can bound Inference body.
    # PTUPCDR section starts with "4.3.3" H3.
    h_next = _find_heading(doc, "4.3.3 PTUPCDR: Personalized Transfer with Experts")

    # --- Strip any existing content between h_emcdr and h_train (there is
    # an empty paragraph). We will then insert a brand-new Architecture
    # subsection there.
    _delete_between(h_emcdr, h_train)

    # Insert a new H4 "EMCDR Architecture" right after h_emcdr, then its
    # body, then fig14a + caption. Build bottom-up because each insert
    # is anchored on h_emcdr.

    # Insert caption + image first (they'll end up at the bottom of the
    # new subsection).
    _insert_caption(
        h_emcdr,
        "Figure 14a: EMCDR architecture — two independent MF-BPR "
        "spaces bridged by the learned translator f_θ."
    )
    _insert_image(h_emcdr, FIGS / "fig14a_emcdr_architecture.png",
                  width_in=IMAGE_WIDTH_IN)

    # Architecture body paragraphs (reverse so final order matches list).
    for label, builder in reversed(ARCHITECTURE_BLOCKS):
        _add_label_para(h_emcdr, label, builder)

    # Finally, insert the new "EMCDR Architecture" heading directly after
    # h_emcdr — this ends up on top of all the content we just inserted.
    _blank_heading_after(h_emcdr, "EMCDR Architecture", level=4)

    # --- Wipe Training body and rebuild it compactly.
    _delete_between(h_train, h_infer)

    # Build Training bottom-up: we want final order
    #   [intro line] → Phase 1 → Phase 2 → Phase 3 → Eq5 img →
    #   Eq5 caption → where-explanation → Fig 14 img → Fig 14 caption.

    _insert_caption(
        h_train,
        "Figure 14: EMCDR three-phase training — separate domain models, "
        "then a learned mapping network."
    )
    _insert_image(h_train, FIGS / "fig14_emcdr.png",
                  width_in=IMAGE_WIDTH_IN)

    _insert_body(h_train, eq5_where)

    _insert_caption(
        h_train,
        "Equation 5: EMCDR phase-3 mapping loss (MSE on overlap users)."
    )
    _insert_image(h_train, FIGS / "fig_loss_emcdr.png",
                  width_in=FORMULA_WIDTH_IN)

    for label, builder in reversed(TRAINING_PHASE_BLOCKS):
        _add_label_para(h_train, label, builder)

    # Training intro line.
    def training_intro(p):
        _add_plain(p, "Training proceeds in three sequential phases:")
    _insert_body(h_train, training_intro)

    # --- Wipe Inference body and rebuild.
    _delete_between(h_infer, h_next)

    # End with "HyperParameter Tuning:" paragraph so we don't strand the
    # old HP block; keep it plain and simple.
    _insert_body(h_infer, lambda p: _add_plain(p, "HyperParameter Tuning:"))

    _insert_caption(
        h_infer,
        "Figure 14b: EMCDR inference — one forward pass through f_θ "
        "translates the user's movie vector into game space, then a "
        "dot product ranks every game."
    )
    _insert_image(h_infer, FIGS / "fig14b_emcdr_inference.png",
                  width_in=IMAGE_WIDTH_IN)

    _insert_body(h_infer, inference_para)

    doc.save(str(DOC))
    print("EMCDR section rewritten.")


if __name__ == "__main__":
    main()
