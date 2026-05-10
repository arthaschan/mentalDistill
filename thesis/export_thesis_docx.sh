#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_MD="$ROOT_DIR/thesis_v2.md"
FRONT_MD="$ROOT_DIR/front_matter_submit.md"
COMBINED_MD="$ROOT_DIR/thesis_submission.md"
OUT_DOCX="$ROOT_DIR/thesis_submission.docx"
REFERENCE_DOC="${REFERENCE_DOC:-$ROOT_DIR/MSAAI Master Thesis example 2024 v1b.docx}"
PANDOC_BIN="${PANDOC_BIN:-/home/student/anaconda3/bin/pandoc}"

if [[ ! -f "$MAIN_MD" ]]; then
  echo "Missing main markdown: $MAIN_MD" >&2
  exit 1
fi

if [[ ! -f "$FRONT_MD" ]]; then
  echo "Missing front matter markdown: $FRONT_MD" >&2
  exit 1
fi

if [[ ! -x "$PANDOC_BIN" && ! -f "$PANDOC_BIN" ]]; then
  if ! command -v pandoc >/dev/null 2>&1; then
    echo "pandoc is required" >&2
    exit 1
  fi
  PANDOC_BIN="$(command -v pandoc)"
fi

build_markdown() {
  local output_md="$1"
  FRONT_MD="$FRONT_MD" MAIN_MD="$MAIN_MD" OUTPUT_MD="$output_md" python3 - <<'PY'
import os
from pathlib import Path

front = Path(os.environ['FRONT_MD']).read_text(encoding='utf-8')
main_lines = Path(os.environ['MAIN_MD']).read_text(encoding='utf-8').splitlines()
body = '\n'.join(main_lines[2:]).lstrip()

toc_block = '## 目录\n\n'
body = body.replace('## 第一章 绪论', toc_block + '## 第一章 绪论', 1)

combined = front.rstrip() + '\n\n' + body + '\n'
Path(os.environ['OUTPUT_MD']).write_text(combined, encoding='utf-8')
PY
}

render_docx() {
  local input_md="$1"
  local output_docx="$2"
  local -a pandoc_args

  pandoc_args=(
    "$PANDOC_BIN"
    "$input_md"
    -o
    "$output_docx"
  )

  if [[ -f "$REFERENCE_DOC" ]]; then
    pandoc_args+=(--reference-doc="$REFERENCE_DOC")
  fi

  "${pandoc_args[@]}"
}

patch_docx_toc() {
  local docx_path="$1"
  DOCX_PATH="$docx_path" python3 - <<'PY'
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
XML_NS = 'http://www.w3.org/XML/1998/namespace'
ET.register_namespace('w', W_NS)

docx_path = Path(os.environ['DOCX_PATH'])
tmp_path = docx_path.with_suffix('.tmp.docx')

ns = {'w': W_NS}

with ZipFile(docx_path, 'r') as zin, ZipFile(tmp_path, 'w', compression=ZIP_DEFLATED) as zout:
    document_xml = zin.read('word/document.xml')
    settings_xml = zin.read('word/settings.xml')

    doc_root = ET.fromstring(document_xml)
    body = doc_root.find('w:body', ns)
    if body is None:
        raise RuntimeError('DOCX body not found')

    paragraphs = list(body)
    toc_idx = None
    chapter_idx = None
    for idx, elem in enumerate(paragraphs):
        if elem.tag != f'{{{W_NS}}}p':
            continue
        text = ''.join(t.text or '' for t in elem.findall('.//w:t', ns)).strip()
        if text == '目录':
            toc_idx = idx
        elif toc_idx is not None and text == '第一章 绪论':
            chapter_idx = idx
            break

    if toc_idx is None or chapter_idx is None or chapter_idx <= toc_idx:
        raise RuntimeError('Could not locate TOC insertion range')

    for idx in range(chapter_idx - 1, toc_idx, -1):
        body.remove(paragraphs[idx])

    p = ET.Element(f'{{{W_NS}}}p')
    p_pr = ET.SubElement(p, f'{{{W_NS}}}pPr')
    ET.SubElement(p_pr, f'{{{W_NS}}}pStyle', {f'{{{W_NS}}}val': 'BodyText'})

    def add_run(parent, *, fld_char=None, instr_text=None, text=None):
        run = ET.SubElement(parent, f'{{{W_NS}}}r')
        if fld_char is not None:
            ET.SubElement(run, f'{{{W_NS}}}fldChar', {f'{{{W_NS}}}fldCharType': fld_char})
        if instr_text is not None:
            instr = ET.SubElement(run, f'{{{W_NS}}}instrText')
            instr.set(f'{{{XML_NS}}}space', 'preserve')
            instr.text = instr_text
        if text is not None:
            t = ET.SubElement(run, f'{{{W_NS}}}t')
            t.text = text
        return run

    add_run(p, fld_char='begin')
    add_run(p, instr_text=' TOC \\o "1-3" \\h \\z \\u ')
    add_run(p, fld_char='separate')
    add_run(p, text='右键单击目录并选择“更新域”，即可刷新页码。')
    add_run(p, fld_char='end')

    body.insert(toc_idx + 1, p)

    settings_root = ET.fromstring(settings_xml)
    update_fields = settings_root.find('w:updateFields', ns)
    if update_fields is None:
        update_fields = ET.SubElement(settings_root, f'{{{W_NS}}}updateFields')
    update_fields.set(f'{{{W_NS}}}val', 'true')

    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == 'word/document.xml':
            data = ET.tostring(doc_root, encoding='utf-8', xml_declaration=True)
        elif item.filename == 'word/settings.xml':
            data = ET.tostring(settings_root, encoding='utf-8', xml_declaration=True)
        zout.writestr(item, data)

tmp_path.replace(docx_path)
PY
}

build_markdown "$COMBINED_MD"
render_docx "$COMBINED_MD" "$OUT_DOCX"
patch_docx_toc "$OUT_DOCX"

echo "Generated: $COMBINED_MD"
echo "Generated: $OUT_DOCX"