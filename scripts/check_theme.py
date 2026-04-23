"""Extract theme colors from the document."""
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"

doc = Document(str(DOC))
theme = doc.element.find(qn("w:settings"))

# Theme lives in the .docx package part
import zipfile, io
with zipfile.ZipFile(str(DOC)) as z:
    if "word/theme/theme1.xml" in z.namelist():
        theme_xml = z.read("word/theme/theme1.xml")
        root = etree.fromstring(theme_xml)
        # Find color scheme
        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        scheme = root.find(".//a:themeElements/a:clrScheme", ns)
        if scheme is not None:
            print(f"Color scheme: {scheme.get('name')}")
            for child in scheme:
                tag = child.tag.split("}")[-1]
                for sub in child:
                    stag = sub.tag.split("}")[-1]
                    val = sub.get("val") or sub.get("lastClr")
                    print(f"  {tag}: {stag}={val}")
