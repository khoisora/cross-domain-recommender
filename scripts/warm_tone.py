"""Light tone-warming pass.

Replaces a small set of impersonal openers ("This project…", "were
chosen", "were selected", "was selected") with first-person equivalents.
Only touches specific paragraphs; does not rewrite substantive content.
"""

from pathlib import Path
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

# Ordered list of (old, new) substring swaps applied across the document.
# Each is applied once per occurrence, independently.
SWAPS = [
    # Executive summary abstract
    ("Abstract. This project investigates cross-domain recommendation (CDR)",
     "Abstract. We set out to investigate cross-domain recommendation (CDR)"),
    # Methodology model-selection wording
    ("The model algorithms were chosen to provide a systematic comparison",
     "We picked these model algorithms to give us a systematic comparison"),
    ("These models were selected to represent the three major collaborative filtering paradigms",
     "We picked these models to cover the three major collaborative filtering paradigms"),
    ("These models were chosen to cover different mechanisms for transferring knowledge",
     "We picked these models to cover the different ways knowledge can be transferred"),
    ("It was selected because it is the only model that can recommend niche/unpopular items effectively",
     "We picked it because it is the only model that can recommend niche/unpopular items effectively"),
    # Conclusion
    ("This project demonstrates that cross-domain recommendation from movies to games is effective",
     "Our experiments show that cross-domain recommendation from movies to games works"),
    ("This project makes four contributions:",
     "We make four contributions:"),
    # Intro sentence that sounds institutional
    ("The experiments presented in this report are structured as a clean, progressive lesson series",
     "We present the experiments as a clean, progressive lesson series"),
]


def _replace_in_paragraph(p, old, new):
    """Replace `old` with `new` within paragraph text, preserving the first
    matching run's formatting. Returns True if a change was made.

    Strategy: concatenate run texts, check for match; if the match is contained
    within a single run, do a direct run-text substitution (keeps all runs
    and formatting intact). Otherwise, rewrite the paragraph's run sequence
    by merging the matched span into the run that holds its first char.
    """
    full = "".join(r.text for r in p.runs)
    if old not in full:
        return False

    # Fast path: single-run match.
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new, 1)
            return True

    # Multi-run match: locate the run containing the first char and collapse.
    idx = full.index(old)
    end = idx + len(old)
    cursor = 0
    first_run_idx = None
    first_run_start = 0
    for i, r in enumerate(p.runs):
        L = len(r.text)
        if cursor <= idx < cursor + L:
            first_run_idx = i
            first_run_start = idx - cursor
            break
        cursor += L
    if first_run_idx is None:
        return False

    # Rebuild text across affected runs.
    new_full = full[:idx] + new + full[end:]

    # Redistribute: put everything into the first affected run, clear the rest.
    # Preserve prefix (before idx) in any earlier runs untouched.
    # Simplest: keep runs before first_run_idx intact, put the rest in first_run_idx,
    # then clear subsequent runs.
    prefix_text = full[: idx - first_run_start]  # text that stays in prior runs
    remaining = new_full[len(prefix_text):]
    runs = list(p.runs)
    runs[first_run_idx].text = remaining
    for r in runs[first_run_idx + 1 :]:
        r.text = ""
    return True


def main():
    doc = Document(str(DOC))
    applied = 0
    for old, new in SWAPS:
        hit = False
        for p in doc.paragraphs:
            if _replace_in_paragraph(p, old, new):
                hit = True
                applied += 1
                break
        if not hit:
            print(f"  ✗ not found: {old[:60]}…")
    doc.save(str(DOC))
    print(f"Applied {applied} warm-tone swaps → {DOC}")


if __name__ == "__main__":
    main()
