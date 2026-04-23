"""Dump full text of §5 Question. and Experiment Setup. paragraphs."""

from pathlib import Path
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"


def main():
    doc = Document(str(DOC))
    in_s5 = False
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        s = p.style.name
        if s == "Heading 1":
            in_s5 = t.startswith("5. Experiments")
        if not in_s5:
            continue
        if t.startswith("Question.") or t.startswith("Experiment Setup."):
            # Preserve line breaks (soft break w:br renders as \n in .text)
            full = p.text
            print(f"p#{i} [{s}]")
            print(full)
            print("---")


if __name__ == "__main__":
    main()
