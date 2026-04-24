"""Restyle §4 to match the blue/dark colour scheme applied to §5.

1. Heading 2 (4.1, 4.2, 4.3) → 28pt bold blue #2E5AAC
2. Heading 3 (4.2.1, 4.2.2, etc.) → 24pt bold blue #2E5AAC
3. Heading 4 (Architecture, Training, Inference) → 20pt bold blue #2E5AAC
4. Body text → colour #1F2933
5. Figure/Equation captions (italic) → grey #52606D
"""

from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

BLUE = "2E5AAC"
BODY = "1F2933"
GREY = "52606D"


def _set_run_props(el, *, color=None, sz=None, bold=None):
    """Set properties on all runs in a paragraph element."""
    for r in el.findall(qn("w:r")):
        rPr = r.find(qn("w:rPr"))
        if rPr is None:
            rPr = OxmlElement("w:rPr")
            r.insert(0, rPr)
        # Font
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rPr.insert(0, rFonts)
        for attr in ("ascii", "hAnsi", "cs"):
            rFonts.set(qn(f"w:{attr}"), "Arial")
        if color:
            for old in rPr.findall(qn("w:color")):
                rPr.remove(old)
            c = OxmlElement("w:color")
            c.set(qn("w:val"), color)
            rPr.append(c)
        if sz:
            for tag in ("w:sz", "w:szCs"):
                sz_el = rPr.find(qn(tag))
                if sz_el is None:
                    sz_el = OxmlElement(tag)
                    rPr.append(sz_el)
                sz_el.set(qn("w:val"), sz)
        if bold is True and rPr.find(qn("w:b")) is None:
            rPr.append(OxmlElement("w:b"))


def _para_text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def _is_heading(el, style_ids):
    pPr = el.find(qn("w:pPr"))
    if pPr is None:
        return False
    pStyle = pPr.find(qn("w:pStyle"))
    if pStyle is None:
        return False
    return pStyle.get(qn("w:val")) in style_ids


def _is_italic_para(el):
    """Check if first run is italic."""
    r = el.find(qn("w:r"))
    if r is None:
        return False
    rPr = r.find(qn("w:rPr"))
    if rPr is None:
        return False
    return rPr.find(qn("w:i")) is not None


def main():
    doc = Document(str(DOC))
    body = doc.element.body
    children = list(body)

    H1 = ("Heading1", "heading1")
    H2 = ("Heading2", "heading2")
    H3 = ("Heading3", "heading3")
    H4 = ("Heading4", "heading4")

    # Find §4 boundaries
    s4_start = s4_end = None
    for idx, ch in enumerate(children):
        tag = ch.tag.split("}", 1)[-1]
        if tag == "p" and _is_heading(ch, H1):
            txt = _para_text(ch).strip()
            if txt.startswith("4.") and "Methodology" in txt:
                s4_start = idx
            elif s4_start is not None and s4_end is None:
                s4_end = idx
    if s4_start is None:
        raise RuntimeError("§4 not found")
    if s4_end is None:
        s4_end = len(children)

    stats = {"h2": 0, "h3": 0, "h4": 0, "body": 0, "caption": 0}

    for idx in range(s4_start, s4_end):
        ch = children[idx]
        tag = ch.tag.split("}", 1)[-1]
        if tag != "p":
            continue

        txt = _para_text(ch).strip()
        if not txt:
            continue

        # Heading 2 → 28pt bold blue
        if _is_heading(ch, H2):
            _set_run_props(ch, color=BLUE, sz="28", bold=True)
            stats["h2"] += 1
            continue

        # Heading 3 → 24pt bold blue
        if _is_heading(ch, H3):
            _set_run_props(ch, color=BLUE, sz="24", bold=True)
            stats["h3"] += 1
            continue

        # Heading 4 → 20pt bold blue
        if _is_heading(ch, H4):
            _set_run_props(ch, color=BLUE, sz="20", bold=True)
            stats["h4"] += 1
            continue

        # Figure/Equation captions (italic paragraphs starting with
        # "Figure" or "Equation") → grey
        if _is_italic_para(ch) and (txt.startswith("Figure") or txt.startswith("Equation")):
            _set_run_props(ch, color=GREY)
            stats["caption"] += 1
            continue

        # Body text → #1F2933
        _set_run_props(ch, color=BODY)
        stats["body"] += 1

    doc.save(str(DOC))
    print("§4 restyled:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("✓ Done")


if __name__ == "__main__":
    main()
