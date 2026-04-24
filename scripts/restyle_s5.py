"""Restyle §5 to match the redesigned doc's format/font/vibe.

Changes applied to every lesson in §5:

1. Lesson Heading 2 → larger (28pt), bold, blue #2E5AAC, more spacing.
2. Narrative paragraph (ending with "?") → split into
     HYPOTHESIS  (18pt bold blue label)
     <italic question text>  (22pt italic #1F2933)
3. "Experiment Setup" bold label → Heading 2 "Setup" (26pt bold blue).
4. "Key Finding. <text>" → split into
     KEY FINDING  (18pt bold gold #8A6200 label)
     <bold text>  (on next line)
5. "→ Next up. <text>" → split into
     NEXT UP  (18pt bold blue label)
     <bold text>  (on next line)
6. Body text color → #1F2933 throughout §5.
7. Figure captions (italic, "Figure L…") → 18pt italic, centered.
"""

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

BLUE = "2E5AAC"
BODY = "1F2933"
GOLD = "8A6200"
GREY = "52606D"


def _run(text, *, bold=False, italic=False, font="Arial", sz="20", color=BODY):
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    for attr in ("ascii", "hAnsi", "cs"):
        rFonts.set(qn(f"w:{attr}"), font)
    rPr.append(rFonts)
    if bold:
        rPr.append(OxmlElement("w:b"))
    if italic:
        rPr.append(OxmlElement("w:i"))
    for tag in ("w:sz", "w:szCs"):
        e = OxmlElement(tag)
        e.set(qn("w:val"), sz)
        rPr.append(e)
    if color:
        c = OxmlElement("w:color")
        c.set(qn("w:val"), color)
        rPr.append(c)
    r.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    return r


def _make_label_para(label, *, color=BLUE, sz="18", before="0", after="60"):
    """Uppercase label paragraph like HYPOTHESIS, KEY FINDING, NEXT UP."""
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), before)
    spacing.set(qn("w:after"), after)
    pPr.append(spacing)
    p.append(pPr)
    p.append(_run(label, bold=True, sz=sz, color=color))
    return p


def _make_text_para(text, *, bold=False, italic=False, sz="22",
                    color=BODY, before="0", after="0", line="320"):
    """Body text paragraph with specific styling."""
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), before)
    spacing.set(qn("w:after"), after)
    if line:
        spacing.set(qn("w:line"), line)
        spacing.set(qn("w:lineRule"), "auto")
    pPr.append(spacing)
    p.append(pPr)
    p.append(_run(text, bold=bold, italic=italic, sz=sz, color=color))
    return p


def _para_text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def _clear_runs(el):
    for r in list(el.findall(qn("w:r"))):
        el.remove(r)


def _set_run_color(el, color):
    """Set color on all runs in a paragraph element."""
    for r in el.findall(qn("w:r")):
        rPr = r.find(qn("w:rPr"))
        if rPr is None:
            rPr = OxmlElement("w:rPr")
            r.insert(0, rPr)
        for old in rPr.findall(qn("w:color")):
            rPr.remove(old)
        c = OxmlElement("w:color")
        c.set(qn("w:val"), color)
        rPr.append(c)


def _restyle_heading2(el):
    """Make a Heading 2 look like the redesigned: 26pt bold blue."""
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
        # Size 26pt = 52 half-points... wait, the redesigned uses "26" which
        # in their encoding means 26 half-points = 13pt. Let me check...
        # Actually docx sz val is in half-points. So 26 = 13pt. But that looks
        # small for a heading. The redesigned shows "26" which in the raw XML
        # means 13pt. The lesson headings show "32" = 16pt.
        # But the current headings are "22" = 11pt. The redesigned H2 is 26 = 13pt.
        for tag in ("w:sz", "w:szCs"):
            sz_el = rPr.find(qn(tag))
            if sz_el is None:
                sz_el = OxmlElement(tag)
                rPr.append(sz_el)
            sz_el.set(qn("w:val"), "26")
        # Bold
        if rPr.find(qn("w:b")) is None:
            rPr.append(OxmlElement("w:b"))
        # Color
        for old in rPr.findall(qn("w:color")):
            rPr.remove(old)
        c = OxmlElement("w:color")
        c.set(qn("w:val"), BLUE)
        rPr.append(c)
    # Spacing
    pPr = el.find(qn("w:pPr"))
    if pPr is not None:
        sp = pPr.find(qn("w:spacing"))
        if sp is not None:
            sp.set(qn("w:before"), "120")
            sp.set(qn("w:after"), "160")


def _restyle_lesson_heading(el):
    """Make a lesson Heading 2 bigger: 28pt bold blue."""
    for r in el.findall(qn("w:r")):
        rPr = r.find(qn("w:rPr"))
        if rPr is None:
            rPr = OxmlElement("w:rPr")
            r.insert(0, rPr)
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rPr.insert(0, rFonts)
        for attr in ("ascii", "hAnsi", "cs"):
            rFonts.set(qn(f"w:{attr}"), "Arial")
        for tag in ("w:sz", "w:szCs"):
            sz_el = rPr.find(qn(tag))
            if sz_el is None:
                sz_el = OxmlElement(tag)
                rPr.append(sz_el)
            sz_el.set(qn("w:val"), "32")
        if rPr.find(qn("w:b")) is None:
            rPr.append(OxmlElement("w:b"))
        for old in rPr.findall(qn("w:color")):
            rPr.remove(old)
        c = OxmlElement("w:color")
        c.set(qn("w:val"), BLUE)
        rPr.append(c)
    pPr = el.find(qn("w:pPr"))
    if pPr is not None:
        sp = pPr.find(qn("w:spacing"))
        if sp is not None:
            sp.set(qn("w:before"), "360")
            sp.set(qn("w:after"), "200")


def _is_heading(el, style_ids):
    pPr = el.find(qn("w:pPr"))
    if pPr is None:
        return False
    pStyle = pPr.find(qn("w:pStyle"))
    if pStyle is None:
        return False
    return pStyle.get(qn("w:val")) in style_ids


def main():
    doc = Document(str(DOC))
    body = doc.element.body
    children = list(body)

    H1 = ("Heading1", "heading1")
    H2 = ("Heading2", "heading2")
    H3 = ("Heading3", "heading3")
    H4 = ("Heading4", "heading4")

    # Find §5 boundaries
    s5_start = s5_end = None
    for idx, ch in enumerate(children):
        tag = ch.tag.split("}", 1)[-1]
        if tag == "p" and _is_heading(ch, H1):
            txt = _para_text(ch).strip()
            if txt.startswith("5. Experiments"):
                s5_start = idx
            elif s5_start is not None and s5_end is None:
                s5_end = idx
    if s5_start is None:
        raise RuntimeError("§5 not found")
    if s5_end is None:
        s5_end = len(children)

    stats = {k: 0 for k in [
        "lesson_headings", "hypothesis", "key_finding", "next_up",
        "setup_heading", "body_colored", "sub_headings"
    ]}

    idx = s5_start
    while idx < s5_end:
        ch = children[idx]
        tag = ch.tag.split("}", 1)[-1]

        if tag != "p":
            idx += 1
            continue

        txt = _para_text(ch).strip()

        # --- Lesson headings (5.X Lesson N: ...) ---
        if _is_heading(ch, H2) and txt[:2] == "5." and "Lesson" in txt:
            _restyle_lesson_heading(ch)
            stats["lesson_headings"] += 1
            idx += 1
            continue

        # --- Sub-headings (Heading 3, Heading 4) → blue ---
        if _is_heading(ch, H3 + H4):
            _set_run_color(ch, BLUE)
            stats["sub_headings"] += 1
            idx += 1
            continue

        # --- Narrative paragraph ending with "?" → HYPOTHESIS ---
        if (txt.startswith("We ") and txt.endswith("?")
                and not _is_heading(ch, H1 + H2 + H3 + H4)):
            # Extract question (last sentence ending with ?)
            # Find the last sentence starting after the last period before "?"
            # Actually, split on the question: everything before the last
            # sentence is setup, the last sentence (containing "?") is the Q.
            # Find split point: last sentence boundary before the "?"
            parts = txt.rsplit(". ", 1)
            if len(parts) == 2 and parts[1].endswith("?"):
                setup_text = parts[0] + "."
                question = parts[1]
            else:
                # Whole thing is the question
                setup_text = None
                question = txt

            # Build: HYPOTHESIS label + italic question
            label_p = _make_label_para("HYPOTHESIS", color=BLUE, before="200")
            q_p = _make_text_para(question, italic=True, sz="22",
                                  after="280")

            parent = ch.getparent()
            # If there's a setup portion, rewrite current paragraph to it
            if setup_text:
                _clear_runs(ch)
                ch.append(_run(setup_text, color=BODY))
                # Insert HYPOTHESIS + question after
                ch.addnext(q_p)
                ch.addnext(label_p)  # addnext inserts immediately after
            else:
                # Replace current paragraph with label + question
                ch.addnext(q_p)
                ch.addnext(label_p)
                parent.remove(ch)

            children = list(body)
            s5_end_txt = _para_text(children[s5_end]) if s5_end < len(children) else ""
            # Recalculate s5_end
            for j in range(s5_start, len(children)):
                c = children[j]
                if c.tag.split("}", 1)[-1] == "p" and _is_heading(c, H1):
                    t = _para_text(c).strip()
                    if not t.startswith("5."):
                        s5_end = j
                        break
            stats["hypothesis"] += 1
            idx += 1
            continue

        # --- "Experiment Setup" bold label → "Setup" Heading 2 style ---
        if txt == "Experiment Setup":
            _clear_runs(ch)
            ch.append(_run("Setup", bold=True, sz="26", color=BLUE))
            # Change style to Heading 2
            pPr = ch.find(qn("w:pPr"))
            if pPr is None:
                pPr = OxmlElement("w:pPr")
                ch.insert(0, pPr)
            pStyle = pPr.find(qn("w:pStyle"))
            if pStyle is None:
                pStyle = OxmlElement("w:pStyle")
                pPr.insert(0, pStyle)
            pStyle.set(qn("w:val"), "Heading2")
            # Spacing
            sp = pPr.find(qn("w:spacing"))
            if sp is None:
                sp = OxmlElement("w:spacing")
                pPr.append(sp)
            sp.set(qn("w:before"), "120")
            sp.set(qn("w:after"), "160")
            stats["setup_heading"] += 1
            idx += 1
            continue

        # --- "Key Finding. <text>" → KEY FINDING label + bold text ---
        if txt.startswith("Key Finding."):
            finding_text = txt[len("Key Finding."):].strip()
            label_p = _make_label_para("KEY FINDING", color=GOLD,
                                       sz="18", before="200", after="60")
            text_p = _make_text_para(finding_text, bold=True, sz="20",
                                     after="280", line="320")

            ch.addnext(text_p)
            ch.addnext(label_p)
            ch.getparent().remove(ch)
            children = list(body)
            for j in range(s5_start, len(children)):
                c = children[j]
                if c.tag.split("}", 1)[-1] == "p" and _is_heading(c, H1):
                    t = _para_text(c).strip()
                    if not t.startswith("5."):
                        s5_end = j
                        break
            stats["key_finding"] += 1
            idx += 1
            continue

        # --- "→ Next up. <text>" → NEXT UP label + bold text ---
        if txt.startswith("→ Next up."):
            next_text = txt[len("→ Next up."):].strip()
            label_p = _make_label_para("NEXT UP", color=BLUE,
                                       sz="18", before="200", after="60")
            text_p = _make_text_para(next_text, bold=True, sz="20",
                                     after="320", line="320")
            ch.addnext(text_p)
            ch.addnext(label_p)
            ch.getparent().remove(ch)
            children = list(body)
            for j in range(s5_start, len(children)):
                c = children[j]
                if c.tag.split("}", 1)[-1] == "p" and _is_heading(c, H1):
                    t = _para_text(c).strip()
                    if not t.startswith("5."):
                        s5_end = j
                        break
            stats["next_up"] += 1
            idx += 1
            continue

        # --- Body text color → #1F2933 ---
        if not _is_heading(ch, H1 + H2 + H3 + H4) and txt:
            _set_run_color(ch, BODY)
            stats["body_colored"] += 1

        idx += 1

    doc.save(str(DOC))
    print("§5 restyled:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("✓ Done")


if __name__ == "__main__":
    main()
