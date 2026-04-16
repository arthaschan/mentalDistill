# 12_rationale_distill — 推理链蒸馏 (Rationale Distillation)

## 目标

改变蒸馏范式——不再蒸馏 5-dim 答案分布，而是蒸馏教师（Doubao 推理模型）的 **思考过程**。

## 动机

- 当前最佳蒸馏方案仅传递"选什么"（5维概率），但教师的真正价值在于"为什么选"
- Doubao 推理模型 test 准确率 98.8%，其推理链包含：排除逻辑、关键知识点、解题方法论
- CoT 推理链 = 自然语言级别的知识蒸馏，信号维度远高于 5-dim 软标签

## 方法

### Stage 1: CoT 生成
- 调用 Doubao API 为 672 条训练题生成 CoT 推理链
- System prompt 要求逐步分析 + `答案：X` 结尾
- 仅保留 CoT 答案与 GT 一致的样本（`--filter_correct`）

### Stage 2: 推理链 SFT
- 训练数据：Question + Options → 完整推理链（含答案）
- System prompt: "分析推理后给出答案"
- LoRA: rank=16, alpha=32, target=q/k/v/o_proj
- SFT 损失仅在推理链 + 答案 token 上计算

### 评估
- 评估时使用简短 prompt（只输出字母），与其他模块一致
- 测试集 83 题

## 执行顺序

```bash
# 1. 生成 CoT（需要 Doubao API key）
bash scripts/generate_cot.sh

# 2. 训练
SEED=42 bash scripts/run_train.sh

# 3. 评估
SEED=42 bash scripts/run_eval.sh

# 或一键执行
bash scripts/run_pipeline.sh
```

## 配置

- 教师模型: Doubao 推理模型 (`configs/teacher_candidate.json`)
- 学生模型: Qwen2.5-7B-Instruct + LoRA
- max_length: 2048 (比纯答案 SFT 的 1024 更长，容纳推理链)
- 训练超参: lr=1e-4, epochs=6, batch_size=2, warmup_ratio=0.1, cosine schedule

## 输入

| 文件 | 说明 |
|------|------|
| `data/train.jsonl` | 原始 672 条训练集（含 GT Answer） |
| `data/val.jsonl` | 74 条验证集 |
| `data/test.jsonl` | 83 条测试集 |

## 输出

| 文件 | 说明 |
|------|------|
| `data/train_cot.jsonl` | Doubao CoT 推理链（生成产物） |
| `outputs/seed_*/best/` | 最佳 LoRA adapter |
| `outputs/seed_*/final/` | 最终 LoRA adapter |
| `logs/generate_cot.log` | CoT 生成日志 |
| `logs/train_seed_*.log` | 训练日志 |

## 与其他模块对比

| 方法 | 蒸馏信号 | 信号维度 |
|------|---------|---------|
| 00 GT SFT | 答案字母 | 1-dim |
| 02-05 两阶段 | 5-dim 概率 | 5-dim |
| 07-08 多教师融合 | 加权 5-dim | 5-dim |
| **12 推理链蒸馏** | **自然语言推理链** | **~100-300 token** |

## 模型准备（GitHub 不含模型文件）

本仓库上传到 GitHub 时 **不包含模型权重**。克隆代码后需自行下载：

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

在 `setup.env` 中配置：
```bash
export BASE_MODEL_7B="/your/path/to/models/Qwen2.5-7B-Instruct"
```

还需配置豆包 API key（用于 CoT 推理链生成）。

> **注意**：生成的 CoT 数据（`data/train_cot.jsonl`）、训练产物（`outputs/`）、日志不上传 GitHub。

## 测试集评估

```bash
SEED=42 bash scripts/run_eval.sh
```

## 人机交互测试

```bash
bash scripts/start_quiz.sh
# 默认地址 http://0.0.0.0:7870
```
