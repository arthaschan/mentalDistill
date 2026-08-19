# 27 英文全科·无印度数据·学生超越教师（弱教师 Qwen3-32B → 学生 Qwen2.5-32B）

## 实验目标

验证：**去掉印度 MedMCQA 数据后，英文全科的"学生超越教师"组合是否依然成立。**

复用的是已成功过的"弱教师"组合：教师 = Qwen3-32B（零样本），学生 = Qwen2.5-32B（蒸馏后）。
在"训练集、教师、学生、评测集都不含印度 MedMCQA"的口径下重跑。

## 组合（复现历史成功组合）

- 教师：Qwen3-32B（零样本，弱教师）
- 学生：Qwen2.5-32B-Instruct + LoRA（α=0 纯 GT 蒸馏，rank16/alpha32，1 epoch，seed 42）
- 训练数据：`data/train_no_india_dentalsplit.jsonl`（10168 题 = 无印度 10488 − 抽出 320 道牙科进测试；**去掉 MedMCQA 印度 10000**）
- 评测集：`data/test_no_india.jsonl`（4110 题 = medqa 1273 + mmlu 2837，无印度）

## 历史结果（含印度，作对照）

| 指标 | 数值 |
|---|---|
| 教师 Qwen3-32B 零样本（无印度 4110） | 80.22% |
| 学生 Qwen2.5-32B 零样本（无印度 4110） | 78.49% |
| headroom（教师−学生零样本） | +1.73pp |
| （历史全量 8293 含印度）学生训练后 | 75.64%，超弱教师 73.84% 达 +1.80pp |

## 预期（无印度）

- 学生零样本 78.49%，蒸馏增益预计 ~3.5~3.8%（无印度后增益下降，因为去掉了增益最高的印度题）。
- 学生训练后预计 ~82%，教师 80.22%，headroom 1.73pp < 增益 → **预计仍能超越**（约 +1.5~2pp）。
- 若实测增益 < headroom（1.73pp），则"无印度后超越失败"——这将是"新语料/新知识=杠杆"的直接证据。

## 目录

- `data/train_no_india.jsonl`：无印度训练集（10488）
- `data/test_no_india.jsonl`：无印度测试集（4110，全科）
- `scripts/run_train_noindia.sh`：训练入口（Qwen2.5-32B，α=0，1 epoch）
- `scripts/eval_noindia_full.py`：评估学生训练后 + 教师零样本（全科口径）
- `runs/32B_noindia_a00_s42`：训练产物（adapter + train.log）

## 执行

```bash
cd 27_english_general_noindia
bash scripts/run_train_noindia.sh        # 训练（约 35 分钟）
$HOME/anaconda3/bin/python3 scripts/eval_noindia_full.py   # 评估（学生+教师）
```

## 关键坑

- Qwen2.5-32B 训练/评估用 `DISTILL_PROMPT_LANG=en` + `DISTILL_USE_CHAT_TEMPLATE=1`。
- 牙科子集（无印度）仅 181 题（medqa 114 + mmlu 67），±~7pp 噪声，见 28 文件夹。
