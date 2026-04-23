"""Move hyperparameter content out of §5.9 into a new top-level §6.

Three things happen:

1. Remove the empty "HyperParameter Tuning:" stub paragraphs that sit at
   the tail of each model subsection in §4.

2. Physically relocate the §5.9 Hyperparameter Analysis block to sit
   *after* §5.10 Lessons Recap, re-title it "6. Hyperparameter Tuning"
   (Heading 1) and promote its five sub-sections (Heading 3 → Heading 2)
   with new 6.1–6.5 numbering.

3. Renumber downstream top-level sections so the sequence stays
   monotonic: §6 Discussion → §7, §7 System Design → §8, §8 Conclusion
   → §9, §9 References → §10, and their sub-sections in lock-step.
   Also update the plain-text ToC entries at the front of the doc.
"""

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

# --- Empty stubs to drop in §4 ------------------------------------------------
STUB_TEXT = "HyperParameter Tuning:"

# --- §5.9 block ---------------------------------------------------------------
OLD_SECTION_HEADING = "5.9 Hyperparameter Analysis"
NEW_SECTION_TITLE = "6. Hyperparameter Tuning"

# Sub-heading renaming within the block (Heading 3 → Heading 2, renumber).
SUBHEADING_RENAMES = {
    "CMF learning rate and α weight":
        "6.1 CMF learning rate and α weight",
    "EMCDR co-occurrence interaction":
        "6.2 EMCDR co-occurrence interaction",
    "LightGCN propagation depth":
        "6.3 LightGCN propagation depth",
    "PTUPCDR expert count and gate temperature":
        "6.4 PTUPCDR expert count and gate temperature",
    "SBERT-CDR source weight":
        "6.5 SBERT-CDR source weight",
}

# --- Downstream renumbering of top-level sections -----------------------------
# Order matters — match longer strings first so "6. Discussion" is not
# substring-matched inside "6.3 …".
RENUMBER_HEADINGS = [
    # top-level
    ("6. Discussion",                   "7. Discussion"),
    ("7. System Design",                "8. System Design"),
    ("8. Conclusion and Future Work",   "9. Conclusion and Future Work"),
    ("9. References",                   "10. References"),
    # §6 subsections → §7
    ("6.1 Routing Decision Rule",               "7.1 Routing Decision Rule"),
    ("6.2 Key Insights",                        "7.2 Key Insights"),
    ("6.3 Limitations and Threats to Validity", "7.3 Limitations and Threats to Validity"),
    ("6.4 Recommendation Rows Design",          "7.4 Recommendation Rows Design"),
    # §7 subsections → §8
    ("7.1 High-Level Architecture",     "8.1 High-Level Architecture"),
    ("7.2 Recommendation Flow",         "8.2 Recommendation Flow"),
    ("7.3 Frontend Architecture",       "8.3 Frontend Architecture"),
    ("7.4 Refresh Architecture",        "8.4 Refresh Architecture"),
    ("7.5 Key Design Decisions",        "8.5 Key Design Decisions"),
    ("7.6 Model Selection Rationale",   "8.6 Model Selection Rationale"),
]

# --- ToC updates --------------------------------------------------------------
TOC_RENAMES = [
    # drop §5.9 entry, keep §5.10 but renumber to §5.9
    ("   5.9 Hyperparameter Analysis", None),   # delete
    ("   5.10 Lessons Recap",          "   5.9 Lessons Recap"),
]

# Insert new ToC block after the last §5.x entry — placed ahead of
# "6. Discussion" (which will become "7. Discussion" below).
NEW_TOC_ENTRIES = [
    "6. Hyperparameter Tuning",
    "   6.1 CMF learning rate and α weight",
    "   6.2 EMCDR co-occurrence interaction",
    "   6.3 LightGCN propagation depth",
    "   6.4 PTUPCDR expert count and gate temperature",
    "   6.5 SBERT-CDR source weight",
]

# Body references to update (cross-refs that mention "§5.9" etc.)
BODY_REFS = [
    ("§5.9 turns from comparing models to tuning them",
     "§6 turns from comparing models to tuning them"),
    ("reported there",
     "reported there"),
    ("tuned values are from the 5.9 sweeps",
     "tuned values are from the §6 sweeps"),
]


def _find_index(paragraphs, matcher):
    for i, p in enumerate(paragraphs):
        if matcher(p):
            return i
    return -1


def _retext_paragraph(p, new_text):
    """Replace all runs of `p` with a single run whose text is `new_text`.

    Preserves the paragraph's pPr (style) and uses the first run's rPr
    when available so font/size carry over.
    """
    # Capture first-run rPr (if any) before nuking runs.
    first_r = p._p.find(qn("w:r"))
    saved_rPr = None
    if first_r is not None:
        rPr = first_r.find(qn("w:rPr"))
        if rPr is not None:
            from copy import deepcopy
            saved_rPr = deepcopy(rPr)

    # Remove every <w:r> in the paragraph.
    for r in list(p._p.findall(qn("w:r"))):
        p._p.remove(r)

    # Build a fresh run with the saved rPr.
    from docx.oxml import OxmlElement
    new_r = OxmlElement("w:r")
    if saved_rPr is not None:
        new_r.append(saved_rPr)
    t = OxmlElement("w:t")
    t.text = new_text
    t.set(qn("xml:space"), "preserve")
    new_r.append(t)
    p._p.append(new_r)


def _set_style(p, style_name):
    # python-docx accepts display names; fallback to the mangled style id.
    try:
        p.style = style_name
    except KeyError:
        pass


def main():
    doc = Document(str(DOC))
    body = doc.paragraphs[0]._p.getparent()
    paras = doc.paragraphs

    # --- 1. Remove empty "HyperParameter Tuning:" stubs in §4 ----------------
    removed_stubs = 0
    for p in list(paras):
        if p.text.strip() == STUB_TEXT:
            p._p.getparent().remove(p._p)
            removed_stubs += 1
    print(f"Removed {removed_stubs} empty HyperParameter Tuning stubs")

    # Reload list after mutation.
    paras = doc.paragraphs

    # --- 2. Detach §5.9 block and move it after §5.10 ------------------------
    start_idx = _find_index(
        paras, lambda p: p.text.strip() == OLD_SECTION_HEADING
                          and p.style.name == "Heading 2",
    )
    if start_idx < 0:
        raise RuntimeError("Could not locate §5.9 Hyperparameter Analysis")

    # End of block = just before the next Heading 2 or Heading 1 sibling.
    end_idx = start_idx + 1
    while end_idx < len(paras):
        s = paras[end_idx].style.name
        if s in ("Heading 1", "Heading 2"):
            break
        end_idx += 1
    # end_idx now points at §5.10 Lessons Recap (Heading 2).

    # Identify the paragraph AFTER §5.10 ends (next Heading 1 — §6).
    after_510_idx = end_idx + 1
    while after_510_idx < len(paras):
        s = paras[after_510_idx].style.name
        if s == "Heading 1":
            break
        after_510_idx += 1
    # after_510_idx points at "6. Discussion" (Heading 1).

    # Snapshot the XML elements to move BEFORE we mutate the tree.
    block_elements = [paras[i]._p for i in range(start_idx, end_idx)]
    insertion_anchor = paras[after_510_idx]._p  # we insert before this

    # Detach block elements from their current position.
    for el in block_elements:
        el.getparent().remove(el)

    # Re-insert them before the anchor (§6 Discussion).
    for el in block_elements:
        insertion_anchor.addprevious(el)

    # Reload list after mutation.
    paras = doc.paragraphs

    # --- 2b. Retitle the moved heading and promote subheadings ---------------
    # The first paragraph of the moved block is the Heading 2
    # "5.9 Hyperparameter Analysis" — retitle + promote to Heading 1.
    for p in paras:
        if (p.style.name == "Heading 2"
                and p.text.strip() == OLD_SECTION_HEADING):
            _retext_paragraph(p, NEW_SECTION_TITLE)
            _set_style(p, "Heading 1")
            break

    # Promote each Heading 3 inside the moved block to Heading 2 and rename.
    for p in paras:
        if p.style.name != "Heading 3":
            continue
        t = p.text.strip()
        if t in SUBHEADING_RENAMES:
            _retext_paragraph(p, SUBHEADING_RENAMES[t])
            _set_style(p, "Heading 2")

    # --- 3. Renumber downstream Heading 1 / Heading 2 sections ---------------
    # We iterate a single time and dispatch on exact text match.
    renumber_map = {old: new for old, new in RENUMBER_HEADINGS}
    for p in paras:
        if p.style.name not in ("Heading 1", "Heading 2"):
            continue
        t = p.text.strip()
        if t in renumber_map:
            _retext_paragraph(p, renumber_map[t])

    # --- 4. Update ToC entries ----------------------------------------------
    # ToC lives in the first ~60 paragraphs as Normal-style lines.
    toc_map = {old: new for old, new in TOC_RENAMES}

    # First pass: rename/delete existing ToC entries.
    deleted_toc = []
    for p in list(paras):
        t = p.text
        # rename
        if t in toc_map:
            new = toc_map[t]
            if new is None:
                p._p.getparent().remove(p._p)
                deleted_toc.append(t)
            else:
                _retext_paragraph(p, new)

    # Renumber §6→§7, §7→§8 etc. in the ToC block by exact-string match.
    toc_renumber = {
        "6. Discussion":                  "7. Discussion",
        "   6.1 Routing Decision Rule":   "   7.1 Routing Decision Rule",
        "   6.2 Key Insights":            "   7.2 Key Insights",
        "   6.3 Limitations and Threats to Validity":
            "   7.3 Limitations and Threats to Validity",
        "   6.4 Recommendation Rows Design":
            "   7.4 Recommendation Rows Design",
        "7. System Design":               "8. System Design",
        "   7.1 High-Level Architecture": "   8.1 High-Level Architecture",
        "   7.2 Recommendation Flow":     "   8.2 Recommendation Flow",
        "   7.3 Frontend Architecture":   "   8.3 Frontend Architecture",
        "   7.4 Refresh Architecture":    "   8.4 Refresh Architecture",
        "   7.5 Key Design Decisions":    "   8.5 Key Design Decisions",
        "   7.6 Model Selection Rationale":
            "   8.6 Model Selection Rationale",
        "8. Conclusion and Future Work":  "9. Conclusion and Future Work",
        "9. References":                  "10. References",
    }
    for p in paras:
        if p.text in toc_renumber:
            _retext_paragraph(p, toc_renumber[p.text])

    # Insert new §6 ToC block before (the now-)§7 Discussion ToC entry.
    anchor_p = None
    for p in paras:
        if p.text == "7. Discussion":
            anchor_p = p
            break
    if anchor_p is None:
        raise RuntimeError("ToC anchor for new §6 insertion not found")

    # Clone the anchor's paragraph to preserve ToC styling; then overwrite text.
    from copy import deepcopy
    parent = anchor_p._p.getparent()
    anchor_idx = list(parent).index(anchor_p._p)
    for k, entry in enumerate(NEW_TOC_ENTRIES):
        new_p = deepcopy(anchor_p._p)
        # Replace text of this clone.
        # Drop every <w:r> and add a fresh one with our text.
        for r in list(new_p.findall(qn("w:r"))):
            new_p.remove(r)
        from docx.oxml import OxmlElement
        r = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.text = entry
        t.set(qn("xml:space"), "preserve")
        r.append(t)
        new_p.append(r)
        parent.insert(anchor_idx + k, new_p)

    # --- 5. Update body cross-references ------------------------------------
    paras = doc.paragraphs
    for p in paras:
        t = p.text
        changed = False
        new_t = t
        for old, new in BODY_REFS:
            if old in new_t and old != new:
                new_t = new_t.replace(old, new)
                changed = True
        if changed:
            _retext_paragraph(p, new_t)

    doc.save(str(DOC))
    print("✓ Moved hyperparameter content to new top-level §6")
    print(f"  ToC entries deleted: {deleted_toc}")


if __name__ == "__main__":
    main()
