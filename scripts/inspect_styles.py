"""Peek at paragraph styles near the §5 heading."""

from pathlib import Path
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"


def main():
    doc = Document(str(DOC))
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        s = p.style.name
        if t.startswith("5. Experiments") or t.startswith("6.") or t.startswith("Question.") or t.startswith("Experiment Setup."):
            print(f"p#{i} [{s}] {t[:120]}")


if __name__ == "__main__":
    main()
