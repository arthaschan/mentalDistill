# Module 16: Llama-3.3-70B-Instruct-AWQ → Qwen2.5-14B 异构教师蒸馏

## 实验目标

使用非 Qwen 系的大模型（Llama-3.3-70B-Instruct-AWQ）作为教师，对 Qwen2.5-14B 学生进行 Choice-Head 蒸馏。

**核心假设**：跨架构教师（Llama 70B, 72.8% 准确率）虽然远弱于 DeepSeek-V3（87.18%），但其"认知盲区"与 Qwen 系不同，可能提供互补知识信号。

---

## 历史结果

### 教师表现

| 指标 | 值 |
|------|-----|
| 测试集准确率 | 72.45% (718/991) |
| 训练集准确率 | 72.79% (3355/4608) |
| 训练集覆盖率 | 100% (4608/4608) |
| 教师-GT 不一致率 | 27.21% (1254/4608) |
| 软标签保留率 | 48.24% (2223/4608)，其余回退 GT one-hot |

### 14B 学生蒸馏结果（Stage 1 Only）

| Seed | Val (991) | Test 全量 (991) | Test 牙科 (125) |
|------|-----------|----------------|----------------|
| 11 | 86.28% | 86.58% | 79.20% |
| 42 | 84.96% | 87.59% | 80.00% |
| 8 | 84.76% | 87.59% | 80.80% |
| **均值** | **85.33%** | **87.25%** | **80.00%** |

### 与 DeepSeek 教师对比

| 教师 | 教师准确率 | 14B 学生均值 | 差距 |
|------|-----------|-------------|------|
| DeepSeek-V3 | 87.18% | 88.67% | +1.49 pp |
| Llama-70B-AWQ | 72.45% | 87.25% | -1.42 pp |

> **关键发现**：教师准确率相差 14.73pp，但学生差距仅 1.42pp——说明 14B 学生具有极强的纠错能力。

---

## 数据

复用 Module 15 的全量重分割数据集：

| 集合 | 文件 | 样本数 |
|------|------|--------|
| 训练集 | `data/train.jsonl` | 4608 |
| 验证集 | `../15_fulldata_resplit/data/val.jsonl` | 991 |
| 测试集（全量） | `../15_fulldata_resplit/data/test.jsonl` | 991 |
| 测试集（牙科） | `../15_fulldata_resplit/data/test_dental.jsonl` | 125 |
| 教师标签 | `data/teacher_train.jsonl` | 4608 |

---

## 模型准备（GitHub 不含模型文件）

本仓库上传到 GitHub 时 **不包含模型权重**（`models/` 已被 `.gitignore` 排除）。克隆代码后需自行下载所有必要模型：

```bash
# 14B 学生模型（约 28GB）
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models/Qwen2.5-14B-Instruct

# 70B AWQ 教师模型（约 38GB，需要 vLLM 推理）
huggingface-cli download casperhansen/llama-3.3-70b-instruct-awq --local-dir models/Llama-3.3-70B-Instruct-AWQ

# 国内可用 modelscope 加速：
# pip install modelscope
# modelscope download --model Qwen/Qwen2.5-14B-Instruct --local_dir models/Qwen2.5-14B-Instruct
```

在 `setup.env` 中配置：
```bash
export BASE_MODEL_14B="/your/path/to/models/Qwen2.5-14B-Instruct"
export LLAMA70B_MODEL="/your/path/to/models/Llama-3.3-70B-Instruct-AWQ"
```

> **注意**：
> - Llama-70B-AWQ 教师模型约 38GB，需要 vLLM 进行推理（`pip install vllm`）
> - 教师标签（`data/teacher_*.jsonl`）、训练产物（`runs/`）不上传 GitHub
> - 运行前需确保 Module 15 的数据集已生成（`15_fulldata_resplit/data/`）

---

## 执行步骤

### 1. 环境检查

```bash
source setup.env
bash check_env.sh
```

### 2. 生成教师标签（需要 vLLM + 70B 模型）

```bash
bash 16_llama70b_choice_head/scripts/generate_teacher_labels.sh
```

### 3. 构造蒸馏数据

```bash
bash 16_llama70b_choice_head/scripts/prepare_soft_labels.sh
bash 16_llama70b_choice_head/scripts/build_head_dataset.sh
```

### 4. 训练（3 seeds）

```bash
bash 16_llama70b_choice_head/scripts/run_train.sh
```

### 5. 评估

```bash
bash 16_llama70b_choice_head/scripts/run_eval.sh
```

### 6. 一键串行执行

```bash
bash 16_llama70b_choice_head/scripts/run_pipeline.sh
```

---

## 训练配置

| 参数 | 值 |
|------|-----|
| 学生模型 | Qwen2.5-14B-Instruct |
| 教师模型 | Llama-3.3-70B-Instruct-AWQ (via vLLM) |
| Stage | Stage 1 Only（Choice-Head KL 蒸馏） |
| α (KL 权重) | 0.35 |
| 学习率 | 1e-4 |
| LoRA rank / alpha | 16 / 32 |
| Batch size | 2 × grad_accum 4 = effective 8 |
| Seeds | 11, 42, 8 |

---

## 测试集评估

```bash
bash scripts/run_eval.sh
# 或手动指定学生大小
STUDENT_SIZE=14b bash scripts/run_eval.sh
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
```

---

## 目录结构

```
16_llama70b_choice_head/
├── README.md
├── configs/
│   ├── grid_params_best.json
│   └── teacher_candidate.json
├── data/
│   ├── train.jsonl
│   ├── teacher_train.jsonl
│   ├── teacher_test.jsonl
│   ├── teacher_train_soft.jsonl
│   └── train_head_distill.jsonl
├── scripts/
│   ├── generate_teacher_labels.sh
│   ├── prepare_soft_labels.sh
│   ├── build_head_dataset.sh
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── run_pipeline.sh
│   ├── start_web.sh
│   └── start_quiz.sh
├── runs/ (gitignored)
└── logs/ (gitignored)
```

## 复现注意事项

- 该实验是重要的异构教师对比实验，证明了弱教师在大数据量下仍有价值
- 70B AWQ 模型需要约 38GB 显存，建议使用 H100 或 A100 80GB
- vLLM 推理服务需要先启动，再运行标签生成脚本
