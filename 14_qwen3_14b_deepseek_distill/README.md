# Qwen3-8B 学生 + DeepSeek-V3 教师 Choice-Head 蒸馏

## 实验目标

使用新一代 Qwen3-8B 作为学生模型，利用 DeepSeek-V3 教师标签进行 Choice-Head 蒸馏。

**核心假设**：Qwen3-8B（8.2B 参数，36T token 预训练）的推理能力和知识储备应优于 Qwen2.5-7B，
有望在蒸馏后突破 7B 的 79.52% 天花板，同时训练显存远低于 14B 方案。

## 模型信息

| 角色 | 模型 | 架构 | 参数量 | 说明 |
|------|------|------|--------|------|
| 学生 | Qwen3-8B | Qwen3ForCausalLM | 8.2B | `$BASE_MODEL_QWEN3_8B` |
| 教师 | DeepSeek-V3 | MoE 671B | — | API (87.95% test acc) |

### Qwen3 兼容性

- **Tokenizer**：与 Qwen2.5 完全兼容（同一 vocab_size=151643, 同一 ABCDE token ID, 同一 ChatML 格式）
- **新增 token**：`<think>` (151667) / `</think>` (151668)，用于原生思维链
- **Prompt 格式**：`<|im_start|>` / `<|im_end|>` ChatML，与现有训练脚本直接兼容
- **LoRA target**：q_proj / k_proj / v_proj / o_proj（标准 Transformer 结构，与 Qwen2.5 一致）
- **transformers 版本**：4.57.6 原生支持 `qwen3` model_type

### 零样本基线

| 模式 | Test Acc | 说明 |
|------|----------|------|
| Non-thinking (直接答题) | 71.08% (59/83) | prompt 末尾加 `<think>\n\n</think>\n\n` 禁用思考 |
| Thinking (允许推理) | 待补充 | max_new_tokens=1536，模型先 `<think>` 推理再答题 |

> 注：Non-thinking 零样本低于 Qwen2.5-7B (77.11%)，可能因为 Qwen3-8B 的 non-thinking 模式
> 在直接答题场景表现弱于专门的 Instruct 微调模型。蒸馏训练后预期有显著提升。

## 硬件需求

- GPU: NVIDIA H100 NVL 95GB
- Qwen3-8B BF16 + LoRA rank=16: ~22GB 显存
- 教师标签复用 Module 02，无需额外 API 调用

## 训练配置

| 参数 | Stage 1 (Choice-Head) | Stage 2 (GT SFT) |
|------|----------------------|-------------------|
| 训练轮数 | 1 epoch | 5 epochs |
| 学习率 | 1e-4 | 1e-4 |
| α (KL 权重) | 0.35 | 0.0 (纯 SFT) |
| LoRA rank | 16 | 16 (继承 Stage 1) |
| LoRA alpha | 32 | 32 |
| Batch size | 2 × 4 grad_accum = 8 | 2 × 4 = 8 |
| 种子 | 42, 11, 8 | 42, 11, 8 |

> **注**：根据 Module 13 经验，Stage 2 GT SFT 可能对强模型有害。
> 训练完成后需对比 Stage 1 Only vs Two-Stage 结果，择优选取。

## 数据来源

教师标签复用自 `02_deepseek_v3_choice_head/data/`（DeepSeek-V3 已生成的硬标签）。

如需重新 API 调用（场景：想更新教师标签），执行 `scripts/generate_teacher_labels.sh`。

## 执行步骤

### 快速启动（复用已有教师标签）

```bash
source setup.env

# 1. 复制 Module 02 的教师标签（避免重复 API 调用）
cp 02_deepseek_v3_choice_head/data/teacher_train.jsonl 14_qwen3_8b_deepseek_distill/data/
cp 02_deepseek_v3_choice_head/data/teacher_test.jsonl 14_qwen3_8b_deepseek_distill/data/

# 2. 生成软标签
bash 14_qwen3_8b_deepseek_distill/scripts/prepare_soft_labels.sh

# 3. 构造蒸馏数据集
bash 14_qwen3_8b_deepseek_distill/scripts/build_head_dataset.sh

# 4. 两阶段训练
bash 14_qwen3_8b_deepseek_distill/scripts/run_train.sh

# 5. 评估
bash 14_qwen3_8b_deepseek_distill/scripts/run_eval.sh
```

### 完整串行流程（含 API 标签生成）

```bash
source setup.env
bash 14_qwen3_8b_deepseek_distill/scripts/run_pipeline.sh
```

## 目录内容

- `configs/teacher_candidate.json`: DeepSeek API 配置
- `configs/grid_params_best.json`: 训练超参 (3 seeds)
- `data/`: 训练/验证/测试数据 + 教师标签 (生成后)
- `scripts/`: 完整流程脚本
- `runs/`: 训练产物 (gitignored)
- `outputs/`: 评估结果
- `logs/`: 日志 (gitignored)

## 环境变量

需要在 `setup.env` 中添加：

```bash
export BASE_MODEL_QWEN3_8B="/path/to/Qwen3-8B"
```

## 与前序模块对照

| 方法 | 学生 | Test Acc (best seed) | Test Acc (均值) |
|------|------|---------------------|----------------|
| Module 00 纯 GT SFT | 7B | 77.11% | — |
| Module 02 DeepSeek 两阶段 | 7B | 81.93% (s11) | — |
| Module 07 四教师融合 | 7B | 79.52% | — |
| Module 13 14B Stage1 | 14B | 84.34% (s11) | 82.33% |
| Module 13 14B Two-Stage | 14B | 81.93% (s42) | 80.72% |
| **Module 14 Qwen3 蒸馏** | **Qwen3-8B** | **待实验** | **待实验** |

## 注意事项

1. Qwen3-8B 的 non-thinking 零样本基线 (71.08%) 低于 Qwen2.5-7B (77.11%)，但这不影响蒸馏效果 — 蒸馏训练会覆盖模型的 zero-shot 行为
2. 训练脚本使用 ChatML 格式 + 短答案模式（max_new_tokens=4），不触发 Qwen3 的 thinking 模式
3. 如果 Stage 2 导致性能下降（参考 Module 13 经验），应只使用 Stage 1 结果
