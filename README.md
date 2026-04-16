# mentalDistill — CMExam 医学选择题知识蒸馏实验平台

基于 CMExam（中国医学考试真题）的多教师 → 小模型知识蒸馏实验。覆盖 17 个实验模块（Module 00–16），包含 GT SFT 基线、白盒/黑盒/异构教师蒸馏、多教师融合、推理链蒸馏、全量数据扩展等方案。

---

## 快速开始

### 1. 克隆 & 安装依赖

```bash
git clone <repo-url> mentalDistill
cd mentalDistill
pip install -r requirements.txt
```

### 2. 下载模型

**代码已全部上传至 GitHub，但模型权重不包含在仓库中**（`models/` 已被 `.gitignore` 排除）。

根据需要下载以下模型：

```bash
# 必需：7B 学生模型（Module 00–12 均用到）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct

# Module 05/10/13/15/16: 需要 14B 模型
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models/Qwen2.5-14B-Instruct

# Module 01/06: 需要 32B 模型（白盒蒸馏需 ~90GB 显存）
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir models/Qwen2.5-32B-Instruct

# Module 14: Qwen3-14B 学生
huggingface-cli download Qwen/Qwen3-14B --local-dir models/Qwen3-14B

# Module 16: Llama-70B-AWQ 异构教师（约 38GB，需 vLLM）
huggingface-cli download casperhansen/llama-3.3-70b-instruct-awq --local-dir models/Llama-3.3-70B-Instruct-AWQ
```

国内加速可用 modelscope：
```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir models/Qwen2.5-7B-Instruct
```

### 3. 配置环境变量

```bash
cp setup.example.env setup.env
# 编辑 setup.env，填入你的模型路径和 API key
```

`setup.env` 主要变量：

| 变量 | 说明 | 用于模块 |
|------|------|---------|
| `EASYEDIT_PY` | Python 解释器路径 | 全部 |
| `BASE_MODEL_7B` | Qwen2.5-7B-Instruct 路径 | 00–12 |
| `BASE_MODEL_14B` | Qwen2.5-14B-Instruct 路径 | 05/10/13/15/16 |
| `BASE_MODEL_32B` | Qwen2.5-32B-Instruct 路径 | 01/06 |
| `BASE_MODEL_QWEN3_8B` | Qwen3-14B 路径 | 14 |
| `DEEPSEEK_API_KEY` | DeepSeek API | 02/06/13/15 |
| `DOUBAO_API_KEY` | 豆包 API | 03/08/12 |
| `MOONSHOT_API_KEY` | Kimi API | 04 |

### 4. 环境自检

```bash
source setup.env
bash check_env.sh
```

---

## 实验模块总览

| 模块 | 教师 | 学生 | 方法 | 最佳结果 |
|------|------|------|------|---------|
| 00 | — | 7B | 纯 GT SFT 基线 | 77.11% |
| 01 | Qwen2.5-32B（白盒） | 7B | KL 白盒蒸馏 | 77.11% |
| 02 | DeepSeek-V3 | 7B | Choice-Head 两阶段 | **81.93%** |
| 03 | Doubao | 7B | Choice-Head 两阶段 | 80.72% |
| 04 | Kimi | 7B | Choice-Head 两阶段 | 77.11%（负例） |
| 05 | Qwen2.5-14B | 7B | Choice-Head 两阶段 | 75.90%（负例） |
| 06 | Qwen2.5-32B（API） | 7B | Choice-Head 两阶段 | — |
| 07 | 多教师融合 | 7B | 静态标签融合 | 79.52%（负例） |
| 08 | 多教师融合 | 7B | 真实多票 + 自一致性 | — |
| 09 | 7B 自身 | 7B | 自蒸馏 | — |
| 10 | 14B 自身 | 14B | 自蒸馏 | — |
| 11 | 多 Seed | 7B | 集成投票 | 79.52% |
| 12 | Doubao（推理） | 7B | CoT 推理链蒸馏 | — |
| 13 | DeepSeek-V3 | **14B** | Choice-Head S1 only | 84.34% |
| 14 | DeepSeek-V3 | **Qwen3-14B** | Choice-Head 两阶段 | — |
| 15 | DeepSeek-V3 | 7B/14B | **全量 6591 题重分割** | 14B: **89.10%** |
| 16 | Llama-70B-AWQ | **14B** | 异构教师蒸馏 | 14B: **87.59%** |

> Module 15/16 使用 991 题测试集；Module 00–14 使用 83 题测试集。两组绝对值不可直接比较。

---

## 运行单个模块

每个模块目录结构统一，入口脚本一致：

```bash
source setup.env

# 完整串行流程（教师标签 → 训练 → 评估）
bash <module>/scripts/run_pipeline.sh

# 单独训练
bash <module>/scripts/run_train.sh

# 单独评估
bash <module>/scripts/run_eval.sh

# 启动模型推理 Web 界面
bash <module>/scripts/start_web.sh

# 启动人机答题界面（浏览器选择题测试，http://0.0.0.0:7870）
bash <module>/scripts/start_quiz.sh
```

## 批量执行

```bash
# 查看所有模块
bash run_module_batch.sh list

# 批量评估指定模块
bash run_module_batch.sh eval 00_baseline_gt_sft 02_deepseek_v3_choice_head

# 批量执行完整流程
bash run_module_batch.sh pipeline 00_baseline_gt_sft 02_deepseek_v3_choice_head

# 夜间调度 API 教师任务（自动限速）
bash run_night_api_tasks.sh
```

批量执行日志默认写到 `batch_logs/<timestamp>/`，并生成 `summary.txt` 汇总各模块成功/失败情况。

---

## 共享组件 (`shared/`)

| 脚本 | 功能 |
|------|------|
| `train_gt_sft.py` | GT SFT 训练 |
| `train_choice_head_distill.py` | Choice-Head 两阶段蒸馏训练 |
| `train_whitebox_distill.py` | 白盒 KL 蒸馏训练 |
| `train_rationale_sft.py` | CoT 推理链 SFT |
| `evaluate_model.py` | 统一评估脚本 |
| `serve_model_app.py` | 模型推理 Web 界面 |
| `quiz_app.py` | 人机答题 Web 界面 |
| `generate_teacher_labels_api.py` | API 教师标签生成 |
| `generate_teacher_labels_vllm.py` | vLLM 本地教师标签生成 |
| `build_selective_distill_dataset.py` | 蒸馏数据集构造 |
| `common_env.sh` | 环境变量解析 |

---

## 不会上传到 GitHub 的内容

以下文件/目录由 `.gitignore` 排除，克隆后需本地重新生成：

- `models/` — 模型权重（见上方下载说明）
- `runs/`、`outputs/` — 训练产物与 checkpoint
- `logs/`、`batch_logs/`、`night_logs/` — 运行日志
- `data/teacher_*.jsonl` — 教师标签（需重新调用 API 或本地推理生成）
- `data/train_head_distill.jsonl` — 蒸馏数据集（由构造脚本生成）
- `setup.env` — 本地环境变量（含 API key，不可提交）

---

## 推荐复现顺序

1. **Module 00** — GT SFT 基线（验证环境可用）
2. **Module 02** — DeepSeek-V3 两阶段蒸馏（7B 最优方案）
3. **Module 13** — 14B 学生蒸馏（升级学生规模）
4. **Module 15** — 全量数据重分割（最终统计可靠评估）
5. **Module 16** — Llama-70B 异构教师（跨架构蒸馏对比）

其余模块为对照实验 / 消融实验 / 负例实验，按需复现。

---

## 硬件需求

| 场景 | 显存需求 |
|------|---------|
| 7B LoRA 训练 | ~20 GB |
| 14B LoRA 训练 | ~35 GB |
| 32B 白盒蒸馏（教师+学生同时加载） | ~90 GB |
| Llama-70B-AWQ 推理 (vLLM) | ~38 GB |

推荐 GPU：NVIDIA H100 NVL 95GB 或 A100 80GB。

---

## 文档

- [`docs/thesis_experiment_report.md`](docs/thesis_experiment_report.md) — 完整实验报告（含所有模块结果与分析）
- [`docs/defense_qa_preparation.md`](docs/defense_qa_preparation.md) — 答辩 Q&A 准备
- [`docs/analysis_distill_bottleneck.md`](docs/analysis_distill_bottleneck.md) — 蒸馏瓶颈分析
- 各模块 `README.md` — 模块级复现指南

---

## License

本项目仅供学术研究使用。CMExam 数据集版权归原作者所有。
