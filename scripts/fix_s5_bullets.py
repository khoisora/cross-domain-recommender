"""The previous polish run set pStyle=ListBullet but this template has no
such style, so the three post-Setup paragraphs ended up as plain Normal
paragraphs without a bullet glyph.

Fix: find each triplet directly (Variable Changed / Held Constant /
Expectation immediately following an "Experiment Setup." paragraph inside
§5), and retrofit each one with:
  - a leading bullet run ("•  "),
  - a hanging indent so wrapped lines align with the label start, and
  - the bold label run re-applied to the label portion.
"""

from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

LABELS = ("Variable Changed:", "Held Constant:", "Expectation:")


def _make_run(text, *, bold=False, font="Arial", sz="20"):
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    for attr in ("ascii", "hAnsi", "cs"):
        rFonts.set(qn(f"w:{attr}"), font)
    rPr.append(rFonts)
    if bold:
        rPr.append(OxmlElement("w:b"))
    for tag in ("w:sz", "w:szCs"):
        e = OxmlElement(tag)
        e.set(qn("w:val"), sz)
        rPr.append(e)
    r.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    return r


def _retrofit_bullet(p_element, label, body):
    """Rewrite an existing <w:p> to render as a bulleted item."""
    # Replace/insert pPr with hanging indent so bullet + label sit flush.
    old_pPr = p_element.find(qn("w:pPr"))
    if old_pPr is not None:
        p_element.remove(old_pPr)
    pPr = OxmlElement("w:pPr")

    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "540")    # ~0.375 in indent
    ind.set(qn("w:hanging"), "270") # ~0.19 in hang for the bullet
    pPr.append(ind)

    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "60")
    spacing.set(qn("w:after"), "60")
    pPr.append(spacing)

    # Put pPr at the start.
    p_element.insert(0, pPr)

    # Strip existing runs.
    for r in list(p_element.findall(qn("w:r"))):
        p_element.remove(r)

    # Build new runs: bullet, bold label, body.
    p_element.append(_make_run("•  "))
    p_element.append(_make_run(f"{label} ", bold=True))
    p_element.append(_make_run(body))


def _para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("w:t")))


def main():
    doc = Document(str(DOC))
    body = doc.element.body
    # Collect all paragraphs in document order (as XML elements).
    paragraphs = [c for c in body.iterchildren()
                  if c.tag.split("}", 1)[-1] == "p"]

    # Locate §5 start / §6 start.
    def _is_heading1_text(p, prefix):
        pPr = p.find(qn("w:pPr"))
        if pPr is None:
            return False
        pStyle = pPr.find(qn("w:pStyle"))
        if pStyle is None:
            return False
        return (pStyle.get(qn("w:val")) in ("Heading1", "heading1")
                and _para_text(p).strip().startswith(prefix))

    s5_start = None
    s5_end = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if s5_start is None and _is_heading1_text(p, "5. Experiments"):
            s5_start = i
        elif s5_start is not None and _is_heading1_text(p, "6."):
            s5_end = i
            break
    if s5_start is None:
        raise RuntimeError("Could not find §5 heading")

    s5 = paragraphs[s5_start:s5_end]
    fixes = 0
    for i, p in enumerate(s5):
        if _para_text(p).strip() != "Experiment Setup.":
            continue
        # The next three paragraphs should start with our labels.
        for j in range(1, 4):
            if i + j >= len(s5):
                break
            target = s5[i + j]
            text = _para_text(target).strip()
            label = LABELS[j - 1]
            if not text.startswith(label):
                print(f"  !! setup at §5[{i}] child {j} missing label {label}: "
                      f"{text[:80]}")
                continue
            # Strip label prefix (and any existing bold run duplicating it).
            body_text = text[len(label):].strip()
            _retrofit_bullet(target, label, body_text)
            fixes += 1

    doc.save(str(DOC))
    print(f"Retrofitted {fixes} bullet paragraphs in §5")


if __name__ == "__main__":
    main()
