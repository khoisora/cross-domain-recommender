"""Verify §5 edits applied cleanly."""

from pathlib import Path
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"


def main():
    doc = Document(str(DOC))
    seen_s5 = False
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        s = p.style.name
        if s == "Heading 1":
            if t.startswith("5. Experiments"):
                seen_s5 = True
                print(f"=== {t} ===")
                continue
            if seen_s5:
                break
        if not seen_s5:
            continue
        if (t.startswith("In this lesson")
                or t.startswith("Experiment Setup")
                or t.startswith("Question.")
                or t.startswith("Variable Changed")
                or t.startswith("Held Constant")
                or t.startswith("Expectation")
                or s.startswith("Heading")
                or "Bullet" in s or "List" in s):
            print(f"p#{i} [{s}] {t[:110]}")


if __name__ == "__main__":
    main()
