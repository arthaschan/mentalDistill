# 中文牙科"学生超越教师"（Qwen3-32B → DeepSeek-V3）

## 实验目标

在中文牙科子集（CMExam "口腔医学"，test_dental 125 题）上，用 Qwen3-32B 学生做 α=0 纯 GT SFT 蒸馏（1 epoch），验证"学生训练后超越 DeepSeek 老师"。

核心机制：学生零样本 77.60%，老师 DeepSeek 79.20%，领先幅度(headroom)仅 1.60pp，小于蒸馏增益(~4.8pp)，故训练后反超。

## 历史结果（3 次独立训练）

| 指标 | 数值 |
|---|---|
| 教师 DeepSeek-V3 零样本（test_dental 125 题） | 79.20% |
| 学生 Qwen3-32B 零样本 | 77.60% |
| 学生训练后（3-seed 均值） | **82.40% ± 0.80%** |
| 各 seed | s11 81.6 / s42 82.4 / s8 83.2 |
| 增益（零样本→训练后） | +4.80pp |
| 超越幅度（学生−教师） | **+3.20pp（3 个 seed 全部超过）** |

结果文件：`data/eval_results_qwen3_cn_dental.json`

## 目录内容

- `data/train.jsonl`（4608 题，全学科训练集，来自 CMExam 重分割）
- `data/test_dental.jsonl`（125 题，牙科测试集）
- `data/val_dental.jsonl`（125 题，牙科验证集，训练时做 val）
- `scripts/run_train.sh`：3-seed 训练入口
- `scripts/eval_dental.py`：3-seed 评估（零样本 + 训练后）
- `scripts/eval_single.py`：单模型评估（零样本 或 指定 adapter）

## 依赖

- 基座模型：`../models/Qwen3-32B`（Qwen3 混合思考模型，训练/评估自动关 thinking）
- 训练代码：`../shared/train_choice_head_distill.py`（Choice-Head 蒸馏）
- Python：`$HOME/anaconda3/bin/python3`（torch 2.9 / transformers 4.57 / peft）
- GPU：单 H100 95GB（32B bf16 约 62GB）

## 最优参数

- 蒸馏方式：Choice-Head，α=0（纯标准答案监督，不用教师软标签）
- LoRA：rank 16，alpha 32，dropout 0.05
- 学习率 1e-4，batch size 1 × 梯度累积 8，1 epoch
- 种子：11 / 42 / 8
- 提示：`DISTILL_PROMPT_LANG=zh`（中文牙科 prompt）+ `DISTILL_USE_CHAT_TEMPLATE=1`（Qwen3 自带模板 + enable_thinking=False）

## 执行步骤

```bash
cd 22_chinese_dental_surpass

# 1. 训练（3 seed，每个约 16 分钟）
bash scripts/run_train.sh

# 2. 评估 3 个 seed + 零样本，输出超越幅度
python scripts/eval_dental.py
```

预期输出（与历史结果一致）：

```
s11: 81.6%   s42: 82.4%   s8: 83.2%
零样本 77.6%  训练后 82.40±0.80%  教师 79.2%  Δ +3.20pp → 超越
```

## 关键坑

- Qwen3 是混合思考模型：必须 `DISTILL_USE_CHAT_TEMPLATE=1` + 关 thinking，否则触发推理链被截断，结果全废。
- α=0 时训练数据只需 Question/Options/Answer（标准答案），无需教师标签。
- 中文牙科测试集仅 125 题（±~8pp 噪声），公开数据生态受限所致（详见 `../25_gain_4percent_exploration` 的 4% 研究，或论文 4.5 节）。
