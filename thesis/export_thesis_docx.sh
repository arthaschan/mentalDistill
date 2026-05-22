#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_MD="$ROOT_DIR/thesis_v2.md"
FRONT_MD="$ROOT_DIR/front_matter_submit.md"
COMBINED_MD="$ROOT_DIR/thesis_submission.md"
OUT_DOCX="$ROOT_DIR/thesis_submission.docx"
REFERENCE_DOC="${REFERENCE_DOC:-$ROOT_DIR/MSAAI Master Thesis example 2024 v1b.docx}"
FORMAT_TEMPLATE_DOC="${FORMAT_TEMPLATE_DOC:-$ROOT_DIR/陈天元 256360231-3.docx}"
PANDOC_BIN="${PANDOC_BIN:-/home/student/anaconda3/bin/pandoc}"

if [[ ! -f "$FORMAT_TEMPLATE_DOC" ]]; then
  FORMAT_TEMPLATE_DOC="$REFERENCE_DOC"
fi

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
    --resource-path="$ROOT_DIR"
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
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
XML_NS = 'http://www.w3.org/XML/1998/namespace'
MC_NS = 'http://schemas.openxmlformats.org/markup-compatibility/2006'
PRESERVED_ROOT_NAMESPACES = {
  'mc': MC_NS,
  'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
  'w15': 'http://schemas.microsoft.com/office/word/2012/wordml',
  'w16se': 'http://schemas.microsoft.com/office/word/2015/wordml/symex',
  'w16cid': 'http://schemas.microsoft.com/office/word/2016/wordml/cid',
  'w16': 'http://schemas.microsoft.com/office/word/2018/wordml',
  'w16cex': 'http://schemas.microsoft.com/office/word/2018/wordml/cex',
  'w16sdtdh': 'http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash',
  'w16sdtfl': 'http://schemas.microsoft.com/office/word/2024/wordml/sdtformatlock',
  'w16du': 'http://schemas.microsoft.com/office/word/2023/wordml/word16du',
}

ET.register_namespace('mc', MC_NS)
ET.register_namespace('w', W_NS)

docx_path = Path(os.environ['DOCX_PATH'])
tmp_path = docx_path.with_suffix('.tmp.docx')

ns = {'w': W_NS}


def qn(tag: str) -> str:
  return f'{{{W_NS}}}{tag}'


def ensure_child(parent, tag, attrs=None):
  child = parent.find(f'w:{tag}', ns)
  if child is None:
    child = ET.SubElement(parent, qn(tag), attrs or {})
  elif attrs:
    child.attrib.update(attrs)
  return child


def strip_attrs(elem, names):
  for name in names:
    elem.attrib.pop(qn(name), None)


def set_run_properties(rpr, *, size=None, east_asia='SimSun', latin='Times New Roman'):
  rfonts = ensure_child(rpr, 'rFonts')
  strip_attrs(rfonts, ['asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'cstheme'])
  rfonts.set(qn('ascii'), latin)
  rfonts.set(qn('hAnsi'), latin)
  rfonts.set(qn('cs'), latin)
  rfonts.set(qn('eastAsia'), east_asia)

  lang = ensure_child(rpr, 'lang')
  strip_attrs(lang, ['eastAsia', 'bidi', 'val'])
  lang.set(qn('val'), 'en-US')
  lang.set(qn('eastAsia'), 'zh-CN')
  lang.set(qn('bidi'), 'ar-SA')

  kern = rpr.find('w:kern', ns)
  if kern is not None:
    rpr.remove(kern)

  spacing = rpr.find('w:spacing', ns)
  if spacing is not None:
    rpr.remove(spacing)

  if size is not None:
    ensure_child(rpr, 'sz', {qn('val'): str(size)})
    ensure_child(rpr, 'szCs', {qn('val'): str(size)})


def set_paragraph_spacing(ppr, *, after=None, line=None, before=None):
  spacing = ensure_child(ppr, 'spacing')
  if before is not None:
    spacing.set(qn('before'), str(before))
  if after is not None:
    spacing.set(qn('after'), str(after))
  if line is not None:
    spacing.set(qn('line'), str(line))
    spacing.set(qn('lineRule'), 'auto')


def set_table_borders(tbl_pr):
  tbl_borders = ensure_child(tbl_pr, 'tblBorders')
  for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
    border = ensure_child(tbl_borders, edge)
    border.set(qn('val'), 'single')
    border.set(qn('sz'), '8')
    border.set(qn('space'), '0')
    border.set(qn('color'), '000000')


def serialize_xml(root, *, preserve_root_namespaces=False):
  xml_text = ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')
  if not preserve_root_namespaces:
    return xml_text.encode('utf-8')

  xml_text = xml_text.replace('xmlns:ns2="http://schemas.microsoft.com/office/word/2010/wordml"', 'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"')
  xml_text = xml_text.replace('ns2:', 'w14:')

  match = re.search(r'<w:styles\b[^>]*>', xml_text)
  if match is None:
    raise RuntimeError('Could not locate XML root tag during serialization')

  root_tag = match.group(0)
  root_tag = root_tag.replace('xmlns:ns1="%s"' % MC_NS, 'xmlns:mc="%s"' % MC_NS)
  root_tag = root_tag.replace('ns1:Ignorable=', 'mc:Ignorable=')

  for prefix, uri in PRESERVED_ROOT_NAMESPACES.items():
    decl = f'xmlns:{prefix}="{uri}"'
    if decl not in root_tag:
      root_tag = root_tag[:-1] + f' {decl}>'

  xml_text = xml_text[:match.start()] + root_tag + xml_text[match.end():]
  return xml_text.encode('utf-8')

with ZipFile(docx_path, 'r') as zin, ZipFile(tmp_path, 'w', compression=ZIP_DEFLATED) as zout:
    document_xml = zin.read('word/document.xml')
    settings_xml = zin.read('word/settings.xml')
    styles_xml = zin.read('word/styles.xml')

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
    styles_root = ET.fromstring(styles_xml)

    # Remove East Asian character grid controls that compress Chinese text in Word.
    for elem_name in ['characterSpacingControl']:
        elem = settings_root.find(f'w:{elem_name}', ns)
        if elem is not None:
            settings_root.remove(elem)

    for sect_pr in doc_root.findall('.//w:sectPr', ns):
        doc_grid = sect_pr.find('w:docGrid', ns)
        if doc_grid is not None:
            sect_pr.remove(doc_grid)

    # Normalize default run properties so body text does not depend on template-only settings.
    doc_defaults = styles_root.find('w:docDefaults', ns)
    if doc_defaults is not None:
        rpr_default = doc_defaults.find('w:rPrDefault/w:rPr', ns)
        if rpr_default is None:
            rpr_default_parent = ensure_child(doc_defaults, 'rPrDefault')
            rpr_default = ensure_child(rpr_default_parent, 'rPr')
        set_run_properties(rpr_default, size=24)

        ppr_default = doc_defaults.find('w:pPrDefault/w:pPr', ns)
        if ppr_default is None:
            ppr_default_parent = ensure_child(doc_defaults, 'pPrDefault')
            ppr_default = ensure_child(ppr_default_parent, 'pPr')
        set_paragraph_spacing(ppr_default, after=0, line=360)

    style_sizes = {
        'BodyText': 24,
        'FirstParagraph': 24,
        'Compact': 21,
        'ImageCaption': 21,
        'CaptionedFigure': 21,
        'Heading2': 32,
        'Heading3': 28,
        'Heading4': 24,
    }

    for style in styles_root.findall('w:style', ns):
        style_id = style.get(qn('styleId'))
        style_type = style.get(qn('type'))
        if style_type == 'character':
            rpr = ensure_child(style, 'rPr')
            set_run_properties(rpr)
        if style_id in style_sizes:
            rpr = ensure_child(style, 'rPr')
            set_run_properties(rpr, size=style_sizes[style_id])

            ppr = ensure_child(style, 'pPr')
            if style_id in {'BodyText', 'FirstParagraph'}:
                set_paragraph_spacing(ppr, after=0, line=360)
            elif style_id == 'Compact':
                set_paragraph_spacing(ppr, after=0, line=300)

        if style_id == 'Table':
            tbl_pr = ensure_child(style, 'tblPr')
            set_table_borders(tbl_pr)

    for tbl in doc_root.findall('.//w:tbl', ns):
        tbl_pr = tbl.find('w:tblPr', ns)
        if tbl_pr is None:
            tbl_pr = ET.Element(qn('tblPr'))
            tbl.insert(0, tbl_pr)
        set_table_borders(tbl_pr)

        tbl_w = tbl_pr.find('w:tblW', ns)
        if tbl_w is not None:
            tbl_w.set(qn('type'), 'auto')
            tbl_w.set(qn('w'), '0')

    update_fields = settings_root.find('w:updateFields', ns)
    if update_fields is None:
        update_fields = ET.SubElement(settings_root, f'{{{W_NS}}}updateFields')
    update_fields.set(f'{{{W_NS}}}val', 'true')

    for item in zin.infolist():
      data = zin.read(item.filename)
      if item.filename == 'word/document.xml':
        data = serialize_xml(doc_root)
      elif item.filename == 'word/settings.xml':
        data = serialize_xml(settings_root)
      elif item.filename == 'word/styles.xml':
        data = serialize_xml(styles_root, preserve_root_namespaces=True)
      zout.writestr(item, data)

tmp_path.replace(docx_path)
PY
}

normalize_docx_format() {
  local docx_path="$1"
  local template_doc="$2"

  if [[ ! -f "$template_doc" ]]; then
  echo "Warning: format template not found, skip DOCX format normalization: $template_doc" >&2
  return 0
  fi

  DOCX_PATH="$docx_path" TEMPLATE_DOC="$template_doc" python3 - <<'PY'
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from docx import Document

docx_path = Path(os.environ['DOCX_PATH'])
template_doc = Path(os.environ['TEMPLATE_DOC'])

format_parts = [
  'word/styles.xml',
  'word/fontTable.xml',
  'word/webSettings.xml',
  'word/theme/theme1.xml',
]


def transplant_format_parts(template_path: Path, target_path: Path) -> None:
  with tempfile.TemporaryDirectory() as target_dir, tempfile.TemporaryDirectory() as template_dir:
    target_root = Path(target_dir)
    template_root = Path(template_dir)

    with zipfile.ZipFile(target_path) as archive:
      archive.extractall(target_root)
    with zipfile.ZipFile(template_path) as archive:
      archive.extractall(template_root)

    for part in format_parts:
      source = template_root / part
      if not source.exists():
        continue
      destination = target_root / part
      destination.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(source, destination)

    tmp_docx = target_path.with_suffix('.tmp.docx')
    with zipfile.ZipFile(tmp_docx, 'w', zipfile.ZIP_DEFLATED) as archive:
      for path in target_root.rglob('*'):
        if path.is_file():
          archive.write(path, path.relative_to(target_root).as_posix())
    tmp_docx.replace(target_path)


def pick_body_style(paragraph, document, current_index):
  text = paragraph.text.strip()
  style_names = {style.name for style in document.styles}
  body_text_style = document.styles['Body Text'] if 'Body Text' in style_names else document.styles['Normal']
  first_paragraph_style = document.styles['First Paragraph'] if 'First Paragraph' in style_names else body_text_style
  caption_style = None
  if 'Captioned Figure' in style_names:
    caption_style = document.styles['Captioned Figure']
  elif 'Image Caption' in style_names:
    caption_style = document.styles['Image Caption']

  if text.startswith('图 ') and caption_style is not None:
    return caption_style

  prev_nonempty = None
  paragraphs = document.paragraphs
  for prev_index in range(current_index - 1, -1, -1):
    if paragraphs[prev_index].text.strip():
      prev_nonempty = paragraphs[prev_index]
      break

  prev_style_name = prev_nonempty.style.name if prev_nonempty is not None and prev_nonempty.style else ''
  if prev_style_name.startswith('Heading'):
    return first_paragraph_style
  return body_text_style


def normalize_paragraph_styles(target_path: Path) -> None:
  document = Document(target_path)
  protected_prefixes = ('Heading',)
  protected_names = {
    'Title',
    'Subtitle',
    'First Paragraph',
    'Body Text',
    'Image Caption',
    'Captioned Figure',
    'TOC Heading',
    'Contents 1',
    'Contents 2',
    'Contents 3',
    'Contents 4',
  }

  for current_index, paragraph in enumerate(document.paragraphs):
    if not paragraph.text.strip():
      continue
    style_name = paragraph.style.name if paragraph.style else ''
    if style_name in protected_names or style_name.startswith(protected_prefixes):
      continue
    paragraph.style = pick_body_style(paragraph, document, current_index)

  document.save(target_path)


transplant_format_parts(template_doc, docx_path)
normalize_paragraph_styles(docx_path)
PY
}

build_markdown "$COMBINED_MD"
render_docx "$COMBINED_MD" "$OUT_DOCX"
patch_docx_toc "$OUT_DOCX"
normalize_docx_format "$OUT_DOCX" "$FORMAT_TEMPLATE_DOC"

echo "Generated: $COMBINED_MD"
echo "Generated: $OUT_DOCX"