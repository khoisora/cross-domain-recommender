"""Rewrite the static TOC so it matches the actual section structure.

The TOC lives in paragraphs 14..54 (one entry per paragraph, single run in
Arial 10pt). This pass replaces those entries with a list that mirrors the
body's current Heading 1/2 tree, preserving each paragraph's styling.

Content is not changed outside the TOC block.
"""

from pathlib import Path
from docx import Document
from docx.shared import Pt
from copy import deepcopy

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

# TOC entries — indentation uses leading spaces visible in the rendered doc.
TOC_LINES = [
    "1. Executive Summary",
    "2. Introduction",
    "   2.1 Background and Problem Statement",
    "   2.2 Research Methodology and Experiment Design",
    "3. Data",
    "   3.1 Data Source",
    "   3.2 Data Pipeline",
    "   3.3 Data Train/Test Splitting Protocol",
    "4. Methodology",
    "   4.1 Evaluation Protocol",
    "   4.2 Single Domain Models",
    "      4.2.1 MF-BPR: Matrix Factorization with BPR",
    "      4.2.2 NCF (NeuMF): Neural Collaborative Filtering",
    "      4.2.3 LightGCN: Graph Convolution",
    "   4.3 Cross Domain Models",
    "      4.3.1 CMF: Collective Matrix Factorization",
    "      4.3.2 EMCDR: Cross-Domain Embedding Mapping",
    "      4.3.3 PTUPCDR: Personalized Transfer with Experts",
    "   4.4 Content-Based and Post-Processing Models",
    "      4.4.1 SBERT-CDR: Content-Based Semantic Matching",
    "      4.4.2 Co-occurrence Reranking",
    "5. Experiments and Results",
    "   5.1 Lesson 1: Explicit vs Implicit Ranking",
    "   5.2 Lesson 2: Low Overlap Kills CDR",
    "   5.3 Lesson 3: Overlap Filtering Rescues CDR",
    "   5.4 Lesson 4: Source-Rich / Target-Sparse",
    "   5.5 Lesson 5: Catalog Sharpening",
    "   5.6 Lesson 6: Cold-Start Protocol",
    "   5.7 Lesson 7: Content-Aware CDR (SBERT)",
    "   5.8 Lesson 8: Co-occurrence Reranking",
    "   5.9 Hyperparameter Analysis",
    "   5.10 Lessons Recap",
    "6. Discussion",
    "   6.1 Routing Decision Rule",
    "   6.2 Key Insights",
    "   6.3 Limitations and Threats to Validity",
    "   6.4 Recommendation Rows Design",
    "7. System Design",
    "   7.1 High-Level Architecture",
    "   7.2 Recommendation Flow",
    "   7.3 Frontend Architecture",
    "   7.4 Refresh Architecture",
    "   7.5 Key Design Decisions",
    "   7.6 Model Selection Rationale",
    "8. Conclusion and Future Work",
    "9. References",
]

# Range of existing TOC paragraph indices to replace.
TOC_START = 14
TOC_END_EXCLUSIVE = 55  # paragraphs 14..54


def _clone_style_template(paragraph):
    """Return a deep-copied <w:p> element that preserves pPr + rPr only."""
    clone = deepcopy(paragraph._p)
    # Remove all runs from the clone, keep pPr.
    for r in clone.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r'):
        clone.remove(r)
    return clone


def main():
    doc = Document(str(DOC))
    paragraphs = doc.paragraphs
    template = paragraphs[TOC_START]  # use first TOC para as styling template
    # Capture the template run's font so we can replicate.
    tpl_run = template.runs[0]
    font_name = tpl_run.font.name
    font_size = tpl_run.font.size

    # Remove existing TOC paragraphs (14..54 inclusive).
    to_remove = paragraphs[TOC_START:TOC_END_EXCLUSIVE]
    anchor = paragraphs[TOC_START - 1]._p  # "Table of Contents" heading element

    # Build new paragraphs and insert after the anchor in reverse order
    # (addnext keeps them in visual order when applied reverse).
    for line in reversed(TOC_LINES):
        new_p = deepcopy(template._p)
        # Clear runs
        for r in list(new_p.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r')):
            new_p.remove(r)
        # Insert new run with same formatting
        from docx.oxml import OxmlElement
        r = OxmlElement("w:r")
        rPr = OxmlElement("w:rPr")
        if font_name:
            rFonts = OxmlElement("w:rFonts")
            rFonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii', font_name)
            rFonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hAnsi', font_name)
            rPr.append(rFonts)
        if font_size:
            sz = OxmlElement("w:sz")
            sz.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', str(int(font_size.pt * 2)))
            rPr.append(sz)
        r.append(rPr)
        t = OxmlElement("w:t")
        t.text = line
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        r.append(t)
        new_p.append(r)
        anchor.addnext(new_p)

    # Now delete the old TOC paragraphs.
    for p in to_remove:
        p._p.getparent().remove(p._p)

    doc.save(str(DOC))
    print(f"Replaced {len(to_remove)} old TOC lines with {len(TOC_LINES)} new ones")


if __name__ == "__main__":
    main()
