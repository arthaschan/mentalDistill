# Copilot Instructions for mentalDistill

This repository is a standalone CMExam medical multiple-choice knowledge distillation experiment platform.
It contains 19 experiment modules (00–18) covering GT SFT baseline, white-box/black-box/cross-architecture teacher distillation, multi-teacher fusion, rationale distillation, full-data scaling, and information geometry analysis.

## Project layout

- `00_baseline_gt_sft/` – Qwen2.5-7B pure GT SFT baseline
- `01_qwen32b_whitebox/` – Qwen2.5-32B → 7B white-box KL distillation
- `02_deepseek_v3_choice_head/` – DeepSeek-V3 → 7B two-stage distillation
- `03_doubao_choice_head/` – Doubao → 7B two-stage distillation
- `04_kimi_choice_head/` – Kimi → 7B two-stage distillation (negative example)
- `05_qwen14b_choice_head/` – Qwen2.5-14B → 7B two-stage distillation (negative example)
- `06_qwen32b_api_choice_head/` – Qwen2.5-32B API → 7B
- `07_step2_static_fusion/` – Multi-teacher static label fusion (negative result)
- `08_step3_consistency_filter/` – Multi-teacher real multi-vote + self-consistency filtering
- `09_qwen7b_selfdistill/` – Qwen2.5-7B self-distillation
- `10_qwen14b_selfdistill/` – Qwen2.5-14B self-distillation
- `11_multi_seed_ensemble/` – Multi-seed ensemble majority vote
- `12_rationale_distill/` – Doubao CoT rationale distillation
- `13_14b_deepseek_distill/` – DeepSeek-V3 → 14B student
- `14_qwen3_14b_deepseek_distill/` – DeepSeek-V3 → Qwen3-14B student
- `15_fulldata_resplit/` – Full CMExam 6591-question resplit (7B + 14B, dual eval)
- `16_llama70b_choice_head/` – Llama-3.3-70B-AWQ → 14B cross-architecture distillation
- `17_alpha_divergence/` – α-divergence family replacing KL in Choice-Head distillation
- `18_fisher_rao_analysis/` – Fisher-Rao distance teacher quality analysis (information geometry)
- `shared/` – shared scripts for training, evaluation, label generation, web serving, quiz UI
- `docs/` – thesis report, defense QA, analysis documents

## Operating assumptions

- This repository should stay lightweight: do not commit logs, outputs, checkpoints, generated teacher labels, or local model copies.
- `.gitignore` intentionally excludes runtime artifacts and generated data.
- Use `setup.example.env` as the template for local environment variables.
- `check_env.sh` is the first command to run on a new machine.
- `requirements.txt` lists Python dependencies.

## Model paths

Preferred model path configuration is via environment variables in `setup.env`:

- `EASYEDIT_PY` – Python interpreter
- `BASE_MODEL_7B` – Qwen2.5-7B-Instruct
- `BASE_MODEL_14B` – Qwen2.5-14B-Instruct
- `BASE_MODEL_32B` – Qwen2.5-32B-Instruct
- `BASE_MODEL_QWEN3_8B` – Qwen3-14B
- `LLAMA70B_MODEL` – Llama-3.3-70B-Instruct-AWQ
- optionally `MODELS_DIR`

Do not hardcode machine-specific absolute paths in scripts unless the user explicitly asks for that.

## API keys

- `DEEPSEEK_API_KEY` – Modules 02, 06, 13, 15
- `DOUBAO_API_KEY` – Modules 03, 08, 12
- `MOONSHOT_API_KEY` – Module 04

## Execution entrypoints

Each module should keep a stable interface:

- `scripts/run_teacher_eval.sh` – evaluate teacher on test set
- `scripts/run_train.sh` – train student model
- `scripts/run_eval.sh` – evaluate trained student
- `scripts/run_pipeline.sh` – full serial pipeline
- `scripts/start_web.sh` – model inference web UI (port 7860)
- `scripts/start_quiz.sh` – human quiz web UI (port 7870)
- `scripts/start_best_web.sh` – quick-start with best adapter

Top-level orchestration:

- `run_module_batch.sh` – batch executor for multiple modules
- `run_night_api_tasks.sh` – overnight API teacher label scheduling
- `run_all_readme.sh` – quick module summary viewer

## Key shared scripts

- `shared/train_gt_sft.py` – GT SFT training
- `shared/train_choice_head_distill.py` – Choice-Head two-stage distillation
- `shared/train_alpha_distill.py` – α-divergence distillation (Module 17)
- `shared/train_whitebox_distill.py` – white-box KL distillation
- `shared/train_rationale_sft.py` – CoT rationale SFT
- `shared/evaluate_model.py` – unified evaluation
- `shared/serve_model_app.py` – model inference web UI
- `shared/quiz_app.py` – human quiz web UI
- `shared/generate_teacher_labels_api.py` – API teacher labels
- `shared/generate_teacher_labels_vllm.py` – vLLM local teacher labels
- `shared/generate_teacher_labels_local_logprobs.py` – local logprobs teacher labels
- `shared/build_selective_distill_dataset.py` – distillation dataset construction
- `shared/fisher_rao_analysis.py` – Fisher-Rao information geometry analysis
- `shared/common_env.sh` – environment variable resolution

## Data conventions

- Training: `data/train.jsonl`, Validation: `data/val.jsonl`, Test: `data/test.jsonl`
- Modules 00–14: 83-question dental test set
- Modules 15–16: 991-question full test set + 125-question dental subset
- Teacher labels: `data/teacher_train.jsonl`, `data/teacher_test.jsonl` (generated, not committed)

## Key experiment results

- 7B best (83-question): 81.93% (Module 02, DeepSeek-V3 teacher, seed 11)
- 14B best (83-question): 84.34% (Module 13, DeepSeek-V3 teacher, seed 11)
- 14B best (991-question full): 89.10% (Module 15, seed 8)
- 14B Llama-70B teacher (991-question): 87.59% (Module 16, seeds 42/8)
- Negative examples: Module 04 (weak teacher), 05 (same-capacity teacher), 07 (fake soft label fusion)

## Documentation conventions

- Keep each module README focused on reproducibility: goal, historical result, model download, execution order, eval commands, quiz commands.
- When updating experimental defaults, prefer editing module README + config together.
- Preserve the modular structure. Do not collapse all experiments into a single monolithic script.
- Thesis report: `docs/thesis_experiment_report.md` (19 sections, comprehensive)

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

## Hardware

- Development GPU: NVIDIA H100 NVL 95GB
- 7B LoRA training: ~20GB VRAM
- 14B LoRA training: ~35GB VRAM
- 32B white-box distillation: ~90GB VRAM
- Llama-70B-AWQ inference (vLLM): ~38GB VRAM
