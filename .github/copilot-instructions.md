# Copilot Instructions for mentalDistill

This repository is a standalone extraction of the rebuild distillation workspace from EasyEdit.
It is organized as a migration-ready project for dental multiple-choice distillation experiments.

## Project layout

- `00_baseline_gt_sft/` – Qwen2.5-7B pure GT SFT baseline
- `01_qwen32b_whitebox/` – Qwen2.5-32B -> 7B white-box distillation
- `02_deepseek_v3_choice_head/` – DeepSeek-V3 -> 7B two-stage distillation
- `03_doubao_choice_head/` – Doubao -> 7B two-stage distillation
- `04_kimi_choice_head/` – Kimi -> 7B two-stage distillation
- `05_qwen14b_choice_head/` – Qwen2.5-14B -> 7B two-stage distillation
- `shared/` – shared scripts for evaluation, label generation, training, web serving, and scheduling

## Operating assumptions

- This repository should stay lightweight: do not commit logs, outputs, checkpoints, generated teacher labels, or local model copies.
- `.gitignore` intentionally excludes runtime artifacts and generated data.
- Use `setup.example.env` as the template for local environment variables.
- `check_env.sh` is the first command to run on a new machine.

## Model paths

Preferred model path configuration is via environment variables:

- `EASYEDIT_PY`
- `BASE_MODEL_7B`
- `BASE_MODEL_14B`
- `BASE_MODEL_32B`
- optionally `MODELS_DIR`

Do not hardcode machine-specific absolute paths in scripts unless the user explicitly asks for that.

## Execution entrypoints

Each module should keep a stable interface:

- `scripts/run_teacher_eval.sh`
- `scripts/run_train.sh`
- `scripts/run_eval.sh`
- `scripts/run_pipeline.sh`
- `scripts/start_web.sh`
- `scripts/start_best_web.sh`

Top-level orchestration:

- `run_module_batch.sh`
- `run_night_api_tasks.sh`
- `run_all_readme.sh`

## Documentation conventions

- Keep each module README focused on reproducibility: goal, historical result, required inputs, execution order, outputs, caveats.
- When updating experimental defaults, prefer editing module README + config together.
- Preserve the modular structure. Do not collapse all experiments into a single monolithic script.

## Safety checks before committing

Before staging changes, verify that the commit does not include:

- `outputs/`
- `runs/`
- `logs/`
- `batch_logs/`
- `night_logs/`
- generated `teacher_*.jsonl`
- `train_head_distill.jsonl`
- local `setup.env`
- local `models/`
