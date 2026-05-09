#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_MD="$ROOT_DIR/thesis_v2.md"
FRONT_MD="$ROOT_DIR/front_matter_submit.md"
COMBINED_MD="$ROOT_DIR/thesis_submission.md"
OUT_PDF="$ROOT_DIR/thesis_submission.pdf"

PASS1_MD="$ROOT_DIR/.thesis_submission_pass1.md"
PASS1_ODT="$ROOT_DIR/.thesis_submission_pass1.odt"
PASS1_PATCHED_ODT="$ROOT_DIR/.thesis_submission_pass1_patched.odt"
PASS1_PDF="$ROOT_DIR/.thesis_submission_pass1.pdf"

PASS2_MD="$ROOT_DIR/.thesis_submission_pass2.md"
PASS2_ODT="$ROOT_DIR/.thesis_submission_pass2.odt"
PASS2_PATCHED_ODT="$ROOT_DIR/.thesis_submission_pass2_patched.odt"
PASS2_PDF="$ROOT_DIR/.thesis_submission_pass2.pdf"

FINAL_ODT="$ROOT_DIR/.thesis_submission_tmp.odt"
FINAL_PATCHED_ODT="$ROOT_DIR/.thesis_submission_tmp_patched.odt"

TOC_PASS1="$ROOT_DIR/.thesis_toc_pass1.md"
TOC_PASS2="$ROOT_DIR/.thesis_toc_pass2.md"
HEADINGS_JSON="$ROOT_DIR/.thesis_headings.json"

PANDOC_BIN="${PANDOC_BIN:-/home/student/anaconda3/bin/pandoc}"
LIBREOFFICE_BIN="${LIBREOFFICE_BIN:-/usr/bin/libreoffice}"

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

if [[ ! -x "$LIBREOFFICE_BIN" && ! -f "$LIBREOFFICE_BIN" ]]; then
  if ! command -v libreoffice >/dev/null 2>&1; then
    echo "libreoffice is required" >&2
    exit 1
  fi
  LIBREOFFICE_BIN="$(command -v libreoffice)"
fi

build_markdown() {
  local output_md="$1"
  local toc_md="${2:-}"
  ROOT_DIR="$ROOT_DIR" FRONT_MD="$FRONT_MD" MAIN_MD="$MAIN_MD" OUTPUT_MD="$output_md" TOC_MD="$toc_md" HEADINGS_JSON="$HEADINGS_JSON" python3 - <<'PY'
import json
import os
from pathlib import Path

front = Path(os.environ['FRONT_MD']).read_text(encoding='utf-8')
main_lines = Path(os.environ['MAIN_MD']).read_text(encoding='utf-8').splitlines()
body = '\n'.join(main_lines[2:]).lstrip()

toc_path = os.environ['TOC_MD']
if toc_path:
    toc_text = Path(toc_path).read_text(encoding='utf-8').rstrip()
    body = body.replace('## 第一章 绪论', toc_text + '\n\n## 第一章 绪论', 1)

combined = front.rstrip() + '\n\n' + body + '\n'
Path(os.environ['OUTPUT_MD']).write_text(combined, encoding='utf-8')

headings = []
skip_level2 = {
  '摘要',
  'Abstract',
  '符号与缩写名称列表',
    'A Knowledge Distillation-Based Automatic Dental Multiple-Choice Question Answering System',
    '学位论文独创性声明',
    '学位论文版权使用授权声明',
    '目录',
}
for raw_line in body.splitlines():
    line = raw_line.strip()
    if line.startswith('## '):
        title = line[3:].strip()
        if title in skip_level2:
            continue
        headings.append({'level': 2, 'title': title})

Path(os.environ['HEADINGS_JSON']).write_text(json.dumps(headings, ensure_ascii=False, indent=2), encoding='utf-8')
PY
}

patch_odt() {
  local src_odt="$1"
  local dst_odt="$2"
  SRC_ODT="$src_odt" DST_ODT="$dst_odt" python3 - <<'PY'
import os
import re
import zipfile

src = os.environ['SRC_ODT']
dst = os.environ['DST_ODT']

with zipfile.ZipFile(src, 'r') as zin, zipfile.ZipFile(dst, 'w') as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == 'content.xml':
            text = data.decode('utf-8')
            text = text.replace(
                '<style:style style:name="TableHeaderRowCell" style:family="table-cell">\n      <style:table-cell-properties fo:border="none" />\n    </style:style>',
                '<style:style style:name="TableHeaderRowCell" style:family="table-cell">\n      <style:table-cell-properties fo:border="0.75pt solid #000000" fo:padding="0.03in" />\n    </style:style>'
            )
            text = text.replace(
                '<style:style style:name="TableRowCell" style:family="table-cell">\n      <style:table-cell-properties fo:border="none" />\n    </style:style>',
                '<style:style style:name="TableRowCell" style:family="table-cell">\n      <style:table-cell-properties fo:border="0.75pt solid #000000" fo:padding="0.03in" />\n    </style:style>'
            )

            page_break_bookmarks = {
              '学位论文独创性声明',
              '学位论文版权使用授权声明',
              '摘要',
              'abstract',
              '符号与缩写名称列表',
              '目录',
              '第一章 绪论',
              '第二章 研究背景与理论基础',
              '第三章 设计与方法',
              '第四章 实验结果与分析',
              '第五章 讨论与结论',
              '参考文献',
              '附录',
              '致谢',
            }

            def replace_heading(match):
              bookmark_name = match.group(1)
              if bookmark_name in page_break_bookmarks:
                return match.group(0).replace('Heading_20_2', 'Heading_20_2_PageBreak', 1)
              return match.group(0)

            text = re.sub(
              r'<text:h text:style-name="Heading_20_2" text:outline-level="2"><text:bookmark-start text:name="([^"]+)" />',
              replace_heading,
              text,
            )
            data = text.encode('utf-8')
        elif item.filename == 'styles.xml':
            text = data.decode('utf-8')
            text = text.replace('fo:page-width="8.5in"', 'fo:page-width="210mm"')
            text = text.replace('fo:page-height="11in"', 'fo:page-height="297mm"')
            if 'style:name="MP2"' not in text:
              text = text.replace(
                '</office:automatic-styles>',
                '    <style:style style:name="MP2" style:family="paragraph"\n    style:parent-style-name="Header">\n      <style:paragraph-properties fo:text-align="center"\n      style:justify-single-word="false" />\n      <style:text-properties style:font-name="Liberation Serif" fo:font-size="12pt" style:font-name-asian="Noto Serif CJK SC" style:font-size-asian="12pt" style:font-name-complex="Liberation Serif" style:font-size-complex="12pt" />\n    </style:style>\n    <style:page-layout style:name="MpmBody">\n      <style:page-layout-properties fo:page-width="210mm"\n      fo:page-height="297mm" style:num-format="1"\n      style:print-orientation="portrait" fo:margin-top="25mm"\n      fo:margin-bottom="25mm" fo:margin-left="25.4mm"\n      fo:margin-right="25.4mm" style:writing-mode="lr-tb"\n      style:footnote-max-height="0in">\n        <style:footnote-sep style:width="0.0071in"\n        style:distance-before-sep="0.0398in"\n        style:distance-after-sep="0.0398in" style:line-style="none"\n        style:adjustment="left" style:rel-width="25%"\n        style:color="#000000" />\n      </style:page-layout-properties>\n      <style:header-style>\n        <style:header-footer-properties fo:min-height="0.35in"\n        fo:margin-left="0in" fo:margin-right="0in"\n        fo:margin-bottom="0.08in" style:dynamic-spacing="false"\n        fo:border-bottom="1pt solid #000000" fo:padding-bottom="0.02in" />\n      </style:header-style>\n      <style:footer-style>\n        <style:header-footer-properties fo:min-height="0.4in"\n        fo:margin-left="0in" fo:margin-right="0in"\n        fo:margin-top="0.2in" style:dynamic-spacing="false" />\n      </style:footer-style>\n    </style:page-layout>\n  </office:automatic-styles>'
              )
            text = re.sub(
                r'<style:style style:name="Text_20_body"[\s\S]*?</style:style>',
                '<style:style style:name="Text_20_body" style:display-name="Text body" style:family="paragraph" style:parent-style-name="Standard" style:class="text">\n      <style:paragraph-properties fo:margin-top="0.0598in" fo:margin-bottom="0.0598in" fo:line-height="150%" fo:text-align="justify" style:justify-single-word="false" style:contextual-spacing="false" />\n      <style:text-properties style:font-name="Liberation Serif" fo:font-size="14pt" style:font-name-asian="Noto Serif CJK SC" style:font-size-asian="14pt" style:font-name-complex="Liberation Serif" style:font-size-complex="14pt" />\n    </style:style>',
                text,
            )
            text = re.sub(
                r'<style:style style:name="First_20_paragraph"[\s\S]*?</style:style>',
                '<style:style style:name="First_20_paragraph" style:display-name="First paragraph" style:family="paragraph" style:parent-style-name="Text_20_body" style:next-style-name="Text_20_body" style:class="text">\n      <style:paragraph-properties fo:text-indent="0.2917in" fo:line-height="150%" fo:text-align="justify" style:justify-single-word="false" />\n      <style:text-properties style:font-name="Liberation Serif" fo:font-size="14pt" style:font-name-asian="Noto Serif CJK SC" style:font-size-asian="14pt" style:font-name-complex="Liberation Serif" style:font-size-complex="14pt" />\n    </style:style>',
                text,
            )
            text = re.sub(
                r'<style:style style:name="Heading_20_2"[\s\S]*?</style:style>',
                '<style:style style:name="Heading_20_2" style:display-name="Heading 2" style:family="paragraph" style:parent-style-name="Heading" style:next-style-name="Text_20_body" style:default-outline-level="2" style:class="text">\n      <style:paragraph-properties fo:margin-top="0.20in" fo:margin-bottom="0.10in" fo:text-align="justify" style:contextual-spacing="false" fo:keep-with-next="always" />\n      <style:text-properties style:font-name="Liberation Serif" fo:font-size="18pt" fo:font-weight="bold" style:font-name-asian="Noto Serif CJK SC" style:font-size-asian="18pt" style:font-weight-asian="bold" style:font-name-complex="Liberation Serif" style:font-size-complex="18pt" style:font-weight-complex="bold" />\n    </style:style>',
                text,
            )
            text = re.sub(
                r'<style:style style:name="Heading_20_3"[\s\S]*?</style:style>',
                '<style:style style:name="Heading_20_3" style:display-name="Heading 3" style:family="paragraph" style:parent-style-name="Heading" style:next-style-name="Text_20_body" style:default-outline-level="3" style:class="text">\n      <style:paragraph-properties fo:margin-top="0.12in" fo:margin-bottom="0.06in" fo:text-align="justify" style:contextual-spacing="false" fo:keep-with-next="always" />\n      <style:text-properties style:font-name="Liberation Serif" fo:font-size="16pt" fo:font-weight="bold" style:font-name-asian="Noto Serif CJK SC" style:font-size-asian="16pt" style:font-weight-asian="bold" style:font-name-complex="Liberation Serif" style:font-size-complex="16pt" style:font-weight-complex="bold" />\n    </style:style>',
                text,
            )
            if 'Heading_20_2_PageBreak' not in text:
                text = text.replace(
                    '</office:styles>',
                '<style:style style:name="Heading_20_2_PageBreak" style:family="paragraph" style:parent-style-name="Heading_20_2" style:display-name="Heading 2 PageBreak" style:class="text">\n      <style:paragraph-properties fo:break-before="page" style:master-page-name="BodyPage" fo:margin-top="0.20in" fo:margin-bottom="0.10in" fo:text-align="justify" style:contextual-spacing="false" fo:keep-with-next="always" />\n      <style:text-properties style:font-name="Liberation Serif" fo:font-size="18pt" fo:font-weight="bold" style:font-name-asian="Noto Serif CJK SC" style:font-size-asian="18pt" style:font-weight-asian="bold" style:font-name-complex="Liberation Serif" style:font-size-complex="18pt" style:font-weight-complex="bold" />\n    </style:style>\n</office:styles>'
                )
            else:
                text = re.sub(
                    r'<style:style style:name="Heading_20_2_PageBreak"[\s\S]*?</style:style>',
                '<style:style style:name="Heading_20_2_PageBreak" style:family="paragraph" style:parent-style-name="Heading_20_2" style:display-name="Heading 2 PageBreak" style:class="text">\n      <style:paragraph-properties fo:break-before="page" style:master-page-name="BodyPage" fo:margin-top="0.20in" fo:margin-bottom="0.10in" fo:text-align="justify" style:contextual-spacing="false" fo:keep-with-next="always" />\n      <style:text-properties style:font-name="Liberation Serif" fo:font-size="18pt" fo:font-weight="bold" style:font-name-asian="Noto Serif CJK SC" style:font-size-asian="18pt" style:font-weight-asian="bold" style:font-name-complex="Liberation Serif" style:font-size-complex="18pt" style:font-weight-complex="bold" />\n    </style:style>',
                    text,
                )
            if 'style:name="BodyPage"' not in text:
              text = text.replace(
                '</office:master-styles>',
                '    <style:master-page style:name="BodyPage"\n    style:page-layout-name="MpmBody" style:next-style-name="BodyPage">\n      <style:header>\n        <text:p text:style-name="MP2">HKCHC MSAAI 硕士论文</text:p>\n      </style:header>\n      <style:footer>\n        <text:p text:style-name="MP1">\n          <text:page-number text:select-page="current">\n          1</text:page-number>\n        </text:p>\n      </style:footer>\n    </style:master-page>\n  </office:master-styles>'
              )
            data = text.encode('utf-8')
        zout.writestr(item, data)
PY
}

render_pdf() {
  local input_md="$1"
  local output_odt="$2"
  local patched_odt="$3"
  local output_pdf="$4"

  "$PANDOC_BIN" "$input_md" \
    --metadata title="基于知识蒸馏的牙科选择题自动答题系统" \
    -o "$output_odt"

  patch_odt "$output_odt" "$patched_odt"
  "$LIBREOFFICE_BIN" --headless --convert-to pdf --outdir "$ROOT_DIR" "$patched_odt" >/tmp/export_thesis_pdf.log 2>&1
  mv -f "${patched_odt%.odt}.pdf" "$output_pdf"
}

build_static_toc() {
  local input_pdf="$1"
  local output_toc="$2"
  INPUT_PDF="$input_pdf" OUTPUT_TOC="$output_toc" HEADINGS_JSON="$HEADINGS_JSON" python3 - <<'PY'
import json
import os
import subprocess
from pathlib import Path

pdf = Path(os.environ['INPUT_PDF'])
headings = json.loads(Path(os.environ['HEADINGS_JSON']).read_text(encoding='utf-8'))

pdfinfo = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True, check=True).stdout.splitlines()
pages = next(int(line.split(':', 1)[1].strip()) for line in pdfinfo if line.startswith('Pages:'))

page_texts = {}
for page in range(1, pages + 1):
    text = subprocess.run(
        ['pdftotext', '-f', str(page), '-l', str(page), str(pdf), '-'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    page_texts[page] = [line.strip() for line in text.splitlines()]

mapped = []
for item in headings:
    last_page = None
    for page in range(1, pages + 1):
        if item['title'] in page_texts[page]:
            last_page = page
    if last_page is not None:
        mapped.append({**item, 'page': last_page})

def format_entry(title: str, page: int, indent: str) -> str:
    width = 34 if not indent else 30
    dots = '.' * max(6, width - len(title))
    return f'{indent}{title} {dots} {page}'

lines = ['## 目录', '']
for item in mapped:
    indent = '' if item['level'] == 2 else '  '
    lines.append(format_entry(item['title'], item['page'], indent) + '  ')

lines.append('\\newpage')

Path(os.environ['OUTPUT_TOC']).write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
}

build_markdown "$PASS1_MD"
render_pdf "$PASS1_MD" "$PASS1_ODT" "$PASS1_PATCHED_ODT" "$PASS1_PDF"

build_static_toc "$PASS1_PDF" "$TOC_PASS1"
build_markdown "$PASS2_MD" "$TOC_PASS1"
render_pdf "$PASS2_MD" "$PASS2_ODT" "$PASS2_PATCHED_ODT" "$PASS2_PDF"

build_static_toc "$PASS2_PDF" "$TOC_PASS2"
build_markdown "$COMBINED_MD" "$TOC_PASS2"
render_pdf "$COMBINED_MD" "$FINAL_ODT" "$FINAL_PATCHED_ODT" "$OUT_PDF"

echo "Generated: $COMBINED_MD"
echo "Generated: $OUT_PDF"