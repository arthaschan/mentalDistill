# 28 英文牙科·无印度数据·学生超越教师（弱教师 Qwen3-32B → 学生 Qwen2.5-32B）

## 实验目标

验证：**去掉印度 MedMCQA 数据后，英文牙科的"学生超越教师"组合是否依然成立。**

注意：英文牙科测试集原先 980 题中 799 题来自 MedMCQA（印度）；去掉印度后官方测试只剩 181 题。为降低噪声，**从无印度训练集的 968 道牙科题里随机抽 320 道移入测试集**（seed=42），牙科测试集扩大到 **501 题**（±~2.5pp 噪声）。这 320 道已从训练集移除，避免"训练过的题当测试"泄漏。

## 组合（复现历史成功组合，训练与 27 共享）

- 教师：Qwen3-32B（零样本，弱教师）
- 学生：Qwen2.5-32B-Instruct + LoRA（**复用 27 文件夹训练出的 adapter**，不重复训练）
- 评测集：`data/test_no_india_dental.jsonl`（501 题，无印度牙科子集：官方 181 + 训练集抽出 320）

## 历史结果（含印度，作对照）

| 指标 | 数值 |
|---|---|
| 英文牙科·弱教师（含印度 980 题） | 教师 62.76%，学生 65.65%，超 +2.89pp |

## 目录

- `data/test_no_india_dental.jsonl`：无印度牙科子集（501 题 = 官方 181 + 训练集抽出 320）
- `scripts/eval_noindia_dental.py`：评估学生训练后（27 的 adapter）+ 教师零样本（牙科口径）

## 执行

```bash
cd 28_english_dental_noindia
$HOME/anaconda3/bin/python3 scripts/eval_noindia_dental.py   # 评估（学生+教师，牙科口径）
```

## 关键坑

- 501 题（官方 181 + 抽出 320）仍属小样本（±~2.5pp 噪声），结论以 27（全科 4110 题）为准。
- 学生 adapter 路径指向 `../27_english_general_noindia/runs/32B_noindia_a00_s42`。
