"""Swap the 42 images in project_report_v2.docx with modernized versions.

Usage:
  python scripts/swap_docx.py
"""

from __future__ import annotations
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG_ROOT = HERE.parent / "new_figures"
PROJECT = Path("/sessions/happy-optimistic-gates/mnt/NewCrossDomainRecommenders")
SRC_DOCX = PROJECT / "project_report_v2.docx"
OUT_DOCX = PROJECT / "project_report_v2.docx"

# image<N> -> PNG path (relative to FIG_ROOT)
MAPPING = {
    1:  "model_family_strengths.png",
    2:  "cdr_concept.png",
    3:  "lesson_flow.png",
    4:  "data_pipeline.png",
    5:  "eval_protocol.png",
    6:  "mfbpr_training.png",
    7:  "formulas/formula_bpr.png",
    8:  "mfbpr_inference.png",
    9:  "ncf_gmf.png",
    10: "ncf_mlp.png",
    11: "ncf_combined.png",
    12: "formulas/formula_ncf.png",
    13: "lightgcn_graph.png",
    14: "formulas/formula_lightgcn.png",
    15: "lightgcn_training.png",
    16: "cmf_mechanism.png",
    17: "formulas/formula_cmf.png",
    18: "emcdr_phases.png",
    19: "formulas/formula_emcdr.png",
    20: "ptupcdr_moe.png",
    21: "formulas/formula_ptupcdr.png",
    22: "sbert_encoding.png",
    23: "sbert_cdr.png",
    24: "formulas/formula_sbert.png",
    25: "cooc_matrix.png",
    26: "cooc_reranking.png",
    27: "formulas/formula_cooc.png",
    28: "results/lesson_1.png",
    29: "results/lesson_2.png",
    30: "results/overlap_impact.png",
    31: "results/lesson_3.png",
    32: "results/lesson_4.png",
    33: "results/lesson_5.png",
    34: "coldstart_comparison.png",
    35: "results/lesson_6.png",
    36: "results/lesson_7.png",
    37: "results/lesson_7_subgroups.png",
    38: "results/results_summary.png",
    39: "routing_rule.png",
    40: "system_architecture.png",
    41: "recommendation_flow.png",
    42: "frontend_architecture.png",
}


def main():
    assert SRC_DOCX.exists(), SRC_DOCX

    # Make a temp file we write to, then rename
    tmp = OUT_DOCX.with_suffix(".new.docx")
    backup = PROJECT / "project_report_v2.backup.docx"

    # back up the current docx once (if not already)
    if not backup.exists():
        shutil.copy2(SRC_DOCX, backup)
        print(f"Backed up current docx → {backup.name}")

    # Verify replacement images exist
    missing = []
    for idx, rel in MAPPING.items():
        p = FIG_ROOT / rel
        if not p.exists():
            missing.append((idx, rel))
    if missing:
        print("Missing PNGs:")
        for idx, rel in missing:
            print(f"  image{idx}: {rel}")
        sys.exit(1)

    # Rewrite zip, replacing word/media/image<N>.png entries
    with zipfile.ZipFile(SRC_DOCX, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            name = item.filename
            if name.startswith("word/media/image") and name.endswith(".png"):
                idx = int(name.split("image")[1].split(".")[0])
                rel = MAPPING.get(idx)
                if rel:
                    data = (FIG_ROOT / rel).read_bytes()
                    zout.writestr(name, data)
                    print(f"  image{idx:<3d} ← {rel} ({len(data)} bytes)")
                    continue
            zout.writestr(item, zin.read(name))

    tmp.replace(OUT_DOCX)
    print(f"\nWrote updated docx: {OUT_DOCX}")
    print(f"Backup of original:  {backup}")


if __name__ == "__main__":
    main()
