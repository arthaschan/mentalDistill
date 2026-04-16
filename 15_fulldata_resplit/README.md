# Module 15: 全量数据重分割实验 (Full-Data Resplit)

> **目标**：使用 CMExam 全量数据（6591 题，覆盖 7 个学科）重新划分训练/验证/测试集，  
> 用最优蒸馏配置训练 7B 和 14B 学生模型，通过更大测试集获得统计更可靠的评估结果。  
> **策略**：全量训练 + 双测试（全量 991 + 牙科子集 125）。

---

## 动机

前序实验（Module 00–14）仅使用牙科单科 830 题，测试集仅 83 题（每题 ±1.2pp），
种子方差大（std 2–3pp），难以判断方法差异的统计显著性。
本实验扩展到全量 6591 题单选，测试集 991 题（95% CI ±2.5pp），并保留牙科子集 125 题兼容原始研究目标。

---

## 数据

| 集合 | 样本数 | 说明 |
|------|--------|------|
| train.jsonl | 4608 | 按难度分层抽样，seed=2026 |
| val.jsonl | 991 | 同上 |
| test.jsonl | 991 | 全学科测试 |
| val_dental.jsonl | 125 | 牙科子集（口腔医学） |
| test_dental.jsonl | 125 | 牙科子集 |

- 源文件：`shared/cmexam_full.csv`（6811 行 → 去重 + 仅单选 = 6591）
- 分割比例：70 / 15 / 15，按 `Exam_Level` 分层
- 生成脚本：`scripts/resplit_fulldata.py`

---

## 教师标签

| 教师 | 覆盖率 | 训练集准确率 | 测试集准确率 (991) | 牙科测试 (125) |
|------|--------|-------------|-------------------|---------------|
| DeepSeek-V3 | 4608/4608 (100%) | — | **87.18%** | **79.20%** |

- 软标签参数：`smooth_eps=0.25, min_entropy=0.20, min_margin=0.03`
- 教师-GT 不一致率：12.2% (562/4608) — 处于最优蒸馏信号区间

---

## 实验配置

### 7B: DeepSeek → Qwen2.5-7B 两阶段（复制 Module 02 最优配）

| 参数 | 值 |
|------|-----|
| 学生模型 | Qwen2.5-7B-Instruct |
| Stage 1 | Choice-Head KL 蒸馏，alpha=0.35，1 epoch |
| Stage 2 | GT SFT，2 epochs |
| 学习率 | 1.2e-4 |
| LoRA | r=16, α=32 |
| Batch | 2 (grad_accum=4) |
| Seeds | 11, 42, 7 |

### 14B: DeepSeek → Qwen2.5-14B 仅 Stage 1（复制 Module 13 最优配）

| 参数 | 值 |
|------|-----|
| 学生模型 | Qwen2.5-14B-Instruct |
| Stage 1 | Choice-Head KL 蒸馏，alpha=0.35，1 epoch |
| Stage 2 | **跳过**（Module 13 证明 Stage 2 对强学生有害） |
| 学习率 | 1e-4 |
| LoRA | r=16, α=32 |
| Seeds | 11, 42, 8 |

---

## 结果

### 基线（零样本）

| 模型 | 全量 (991) | 牙科 (125) |
|------|-----------|-----------|
| Qwen2.5-7B | 76.49% | 68.80% |
| Qwen2.5-14B | 83.55% | 74.40% |
| DeepSeek-V3 (Teacher) | 87.18% | 79.20% |

### 7B 蒸馏结果

| Seed | Stage | 全量 (991) | 牙科 (125) |
|------|-------|-----------|-----------|
| 11 | Stage1 | **86.28%** (855/991) | **76.00%** (95/125) |
| 11 | Stage2 | 85.77% (850/991) | 74.40% (93/125) |
| 42 | Stage1 | 85.57% (848/991) | 73.60% (92/125) |
| 42 | Stage2 | 85.07% (843/991) | 73.60% (92/125) |
| 7 | Stage1 | 84.96% (842/991) | 71.20% (89/125) |
| 7 | Stage2 | 84.76% (840/991) | 72.00% (90/125) |
| **S1 均值** | | **85.60%** | **73.60%** |
| **S2 均值** | | **85.20%** | **73.33%** |

### 14B 蒸馏结果（Stage 1 only）

| Seed | 全量 (991) | 牙科 (125) |
|------|-----------|-----------|
| 11 | 88.50% (877/991) | 79.20% (99/125) |
| 42 | 88.40% (876/991) | **80.00%** (100/125) |
| 8 | **89.10%** (883/991) | 78.40% (98/125) |
| **均值** | **88.67%** | **79.20%** |

### 提升幅度汇总

| 模型 | 阶段 | 全量 Δ | 牙科 Δ |
|------|------|--------|--------|
| 7B | S1 均值 | **+9.11 pp** (76.49→85.60) | **+4.80 pp** (68.80→73.60) |
| 7B | S1 最佳(s11) | +9.79 pp (76.49→86.28) | +7.20 pp (68.80→76.00) |
| 14B | S1 均值 | **+5.12 pp** (83.55→88.67) | **+4.80 pp** (74.40→79.20) |
| 14B | S1 最佳(s8) | +5.55 pp (83.55→89.10) | +4.00 pp (74.40→78.40) |

---

## 关键发现

1. **14B 超越教师**：14B 蒸馏均值 88.67%（最佳 89.10%）超过 DeepSeek 教师 87.18%，说明学生在全量数据训练下可通过教师软标签 + GT 联合学习超越教师。
2. **14B 极低种子方差**：3-seed 极差仅 0.70pp (88.40–89.10)，远低于旧 83 题测试的 4.82pp 极差，验证了大测试集的统计稳定性。
3. **Stage 2 对 7B 也轻微有害**：全量 S1 均值 85.60% > S2 均值 85.20%（-0.40pp），牙科 S1 73.60% > S2 73.33%（-0.27pp）。此前仅在 14B 观察到此现象，现在 7B 在大数据量下也确认。
4. **全量训练不牺牲牙科**：14B 牙科 79.20% 均值与教师 79.20% 持平，7B 牙科 76.00% 最佳也显著高于 68.80% 零样本。
5. **统计可靠性大幅提升**：991 题测试集 95% CI ≈ ±2.5pp，远优于旧 83 题的 ±8.6pp。

---

## 与旧实验对比

| 维度 | 旧实验 (Module 02/13) | 本实验 |
|------|----------------------|--------|
| 训练集 | 672 题（牙科单科） | 4608 题（全学科） |
| 测试集 | 83 题（牙科） | 991 题（全学科）+ 125 题（牙科） |
| 14B 最佳 | 84.34% (83 题) | 89.10% (991 题全量) / 78.40% (125 题牙科) |
| 14B 均值 | 82.33% | 88.67% / 79.20% |
| 7B 最佳 | 81.93% (83 题) | 86.28% (991 题全量) / 76.00% (125 题牙科) |
| 种子方差 | 14B std=2.55pp | 14B std≈0.35pp |

> **注意**：旧实验 83 题与新实验 991/125 题不是同一测试集，不可直接比较绝对值。但新实验的统计置信度远高于旧实验。

---

## 目录结构

```
15_fulldata_resplit/
├── README.md
├── configs/
│   ├── grid_params_7b.json
│   ├── grid_params_14b.json
│   └── teacher_candidate.json
├── data/
│   ├── train.jsonl (4608)
│   ├── val.jsonl (991)
│   ├── test.jsonl (991)
│   ├── val_dental.jsonl (125)
│   ├── test_dental.jsonl (125)
│   ├── teacher_train.jsonl (4608)
│   ├── teacher_train_soft.jsonl (4608)
│   ├── teacher_test.jsonl (991)
│   └── train_head_distill.jsonl (4607)
├── scripts/
│   ├── resplit_fulldata.py
│   ├── generate_teacher_labels.sh
│   ├── prepare_soft_labels.sh
│   ├── build_head_dataset.sh
│   ├── run_train_7b.sh
│   ├── run_train_14b.sh
│   ├── run_eval_dual.py
│   └── run_pipeline.sh
├── runs/
│   ├── 20260414_014238_fulldata_7b/
│   └── 20260414_095842_fulldata_14b/
└── logs/
```

## 复现

```bash
cd /home/student/arthas/mentalDistill
source setup.env
bash 15_fulldata_resplit/scripts/run_pipeline.sh
```

需要设置 `DEEPSEEK_API_KEY` 环境变量用于教师标签生成。

## 模型准备（GitHub 不含模型文件）

本仓库上传到 GitHub 时 **不包含模型权重**（`models/` 已被 `.gitignore` 排除）。克隆代码后需自行下载：

```bash
# 7B 学生模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
# 14B 学生模型
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models/Qwen2.5-14B-Instruct

# 国内可用 modelscope 加速：
# pip install modelscope
# modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir models/Qwen2.5-7B-Instruct
# modelscope download --model Qwen/Qwen2.5-14B-Instruct --local_dir models/Qwen2.5-14B-Instruct
```

在 `setup.env` 中配置：
```bash
export BASE_MODEL_7B="/your/path/to/models/Qwen2.5-7B-Instruct"
export BASE_MODEL_14B="/your/path/to/models/Qwen2.5-14B-Instruct"
export DEEPSEEK_API_KEY="your-api-key"
```

> **注意**：教师标签（`data/teacher_*.jsonl`）、训练产物（`runs/`）、日志均不上传 GitHub，需本地重新生成。

## 测试集评估

```bash
# 评估最新训练的模型（自动检测最新 runs 目录）
bash scripts/run_eval.sh

# 手动指定学生模型大小
STUDENT_SIZE=14b bash scripts/run_eval.sh
STUDENT_SIZE=7b bash scripts/run_eval.sh
```

## 人机交互测试

```bash
bash scripts/start_quiz.sh
# 默认加载全量测试集（991 题），地址 http://0.0.0.0:7870
# 切换为牙科子集：
TEST_SET=dental bash scripts/start_quiz.sh
```

## 启动模型推理 Web 界面

```bash
bash scripts/start_web.sh
# 默认加载 14B 模型，切换 7B：
STUDENT_SIZE=7b bash scripts/start_web.sh
```
