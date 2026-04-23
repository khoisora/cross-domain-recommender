"""Remove three redundant paragraphs that predate the typeset equation
blocks and are now superseded by the new "where …" explanations:

  - MF-BPR : "Where σ is the sigmoid function that squashes the score
             difference into [0,1]."
  - NCF    : "Where y=1 for observed interactions and y=0 for sampled
             negatives."
  - CMF    : a duplicate of "CMF maintains three embedding matrices …"
             that sits under "CMF Training" and exactly repeats the
             paragraph under "CMF Architecture".

Each target is matched on a unique text prefix; only the matches listed
below are deleted.
"""

from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

# (prefix, how_many_occurrences_to_delete, delete_which_by_index)
# For the CMF duplicate we want to delete only the 2nd occurrence.
TARGETS = [
    dict(prefix="Where σ is the sigmoid function that squashes",
         keep_first=False),  # single occurrence — delete it
    dict(prefix="Where y=1 for observed interactions and y=0",
         keep_first=False),
    dict(prefix="CMF maintains three embedding matrices",
         keep_first=True),   # two occurrences — delete the 2nd
]


def _delete(para):
    para._p.getparent().remove(para._p)


def main():
    doc = Document(str(DOC))
    deleted = 0
    for spec in TARGETS:
        hits = [p for p in doc.paragraphs if p.text.strip().startswith(spec["prefix"])]
        if not hits:
            print(f"  ✗ no match: {spec['prefix'][:60]}…")
            continue
        to_delete = hits[1:] if spec["keep_first"] else hits
        for p in to_delete:
            _delete(p)
            deleted += 1
            print(f"  ✓ removed: {spec['prefix'][:60]}…")
    doc.save(str(DOC))
    print(f"\nDeleted {deleted} redundant paragraphs")


if __name__ == "__main__":
    main()
