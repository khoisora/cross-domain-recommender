from pathlib import Path
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "project_report_v2.docx"
doc = Document(str(DOC))
for s in doc.styles:
    try:
        print(f"{s.type}  id={s.style_id}  name={s.name}")
    except Exception:
        pass
