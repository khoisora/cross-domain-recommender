"""Survey §5 — list Experiment Setup paragraphs, Question paragraphs, and
all tables that fall within the Experiments and Results section, so we can
plan the three edits (bullets, light-blue table headers, Question rewrite)."""

from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"


def main():
    doc = Document(str(DOC))
    body = doc.element.body

    # Walk children in order, tracking "are we inside §5?"
    in_s5 = False
    idx_p = 0
    idx_t = 0
    questions = []
    setups = []
    tables_in_s5 = []

    for child in body.iterchildren():
        tag = child.tag.split("}", 1)[-1]
        if tag == "p":
            # Extract text + style
            runs_text = "".join(t.text or "" for t in child.iter(qn("w:t")))
            pPr = child.find(qn("w:pPr"))
            pStyle = pPr.find(qn("w:pStyle")) if pPr is not None else None
            style_val = pStyle.get(qn("w:val")) if pStyle is not None else None

            stripped = runs_text.strip()
            # Detect entering §5 / leaving §5
            if style_val == "heading1":
                if stripped.startswith("5. Experiments"):
                    in_s5 = True
                elif in_s5:
                    in_s5 = False

            if in_s5:
                if stripped.startswith("Question."):
                    questions.append((idx_p, stripped[:140]))
                if stripped.startswith("Experiment Setup."):
                    setups.append((idx_p, stripped[:200]))
            idx_p += 1
        elif tag == "tbl":
            if in_s5:
                # First-row first-cell text
                first_cell = child.find(
                    qn("w:tr") + "/" + qn("w:tc"))
                first_text = "".join(
                    t.text or "" for t in first_cell.iter(qn("w:t"))
                ) if first_cell is not None else ""
                tables_in_s5.append((idx_t, first_text[:80]))
            idx_t += 1

    print(f"§5 Questions ({len(questions)}):")
    for i, (idx, t) in enumerate(questions):
        print(f"  [{i}] p#{idx}: {t}")
    print()
    print(f"§5 Experiment Setups ({len(setups)}):")
    for i, (idx, t) in enumerate(setups):
        print(f"  [{i}] p#{idx}: {t}")
    print()
    print(f"§5 Tables ({len(tables_in_s5)}):")
    for i, (idx, t) in enumerate(tables_in_s5):
        print(f"  [{i}] table#{idx}: {t}")


if __name__ == "__main__":
    main()
