# Thesis Format Handover

## Archive Status

- Canonical retained source is `thesis_submission.md`.
- Supporting retained files are `front_matter_submit.md`, `reference_citation_map.md`, `generate_result_figures.py`, and the `figures/` plus `reference/` directories.
- The thesis has already been submitted; temporary export scripts, answer-defense materials, PDF exports, extracted media, and local ODT patch intermediates have been removed from the workspace.

## Historical Notes Kept For Context

- Before cleanup, the thesis export flow had already converged on A4 layout, serif body typography, stronger heading hierarchy, chapter-level static TOC handling, and a body-page header path for the abstract and later pages.
- Thesis wording was synchronized so the contribution is described as a reproducible dental-task evaluation setup and data split, rather than claiming creation of a brand-new benchmark.
- The terminology preference that should remain true in any later edits is:
  - use `评测设置` / `主要评测对象` / `数据划分与评测设置` for the thesis contribution layer;
  - keep `CMExam` itself described as an existing benchmark or evaluation source when discussing prior work.

## Retained Materials

- `thesis_submission.md`: thesis main text source.
- `front_matter_submit.md`: front matter source.
- `FORMAT_HANDOVER.md`: this archive note.
- `reference_citation_map.md`: citation mapping notes.
- `generate_result_figures.py`: figure generation script.
- `figures/`: retained thesis figures.
- `reference/`: retained literature attachments and supporting notes.

## Cleanup Notes

- Removed after submission: export scripts, exported PDF artifacts, patched ODT intermediates, temporary pass files, extracted docx media, and defense-only PPT materials.
- If thesis exports are ever needed again in the future, a new export pipeline should be rebuilt from `thesis_submission.md` instead of relying on the deleted temporary files.