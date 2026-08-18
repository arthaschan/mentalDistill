# 4% 增益的研究与探索

## 实验目标

系统研究"蒸馏增益为什么约等于 4 个百分点、以及能否突破"这一核心问题，包括：增益常数的确立、根因定位、以及两次主动突破尝试（加大训练强度、挑题专门训练）的失败。

## 历史结果（全部结论）

### 1. 增益常数（跨语言/跨学生/跨领域）

| 实验 | 学生零样本 | 学生训练后 | 增益 |
|---|---|---|---|
| 判断题（近随机基线） | 48.60% | 59.77% | +11.17 |
| 中文全科 | 83.55% | 88.67% | +5.12 |
| 英文牙科 | 68.09% | 73.05% | +4.96 |
| 中文牙科 | 77.60% | 82.40% | +4.80 |
| 英文全科（32B） | 71.38% | 75.64% | +4.26 |
| 英文全科（14B） | ~67.83% | 72.41% | ~+4.58 |
| 英文全科（Llama-70B） | 72.43% | 76.27% | +3.84 |
| 英文全科（Qwen3-32B） | 73.84% | 77.60% | +3.75 |

增益集中在 3.75~5.12pp，唯一 >6% 的判断题是"学生接近瞎猜"的特例。

### 2. 根因：near-miss × 纠正率

增益 = "差一点答对"(near-miss) 比例(~12%) × 纠正率(~1/3) ≈ 4%。

- near-miss = 学生答错、且正确答案为其次优选项的题（训练集实测 14.2%）。
- 纠正率 ≈ 1/3 由 4%÷12% 反推；判断题（near-miss 比例极大）增益 +11 反向印证。

### 3. 突破尝试 ①：加大训练强度 —— 失败

| 配置 | 验证集准确率 |
|---|---|
| rank 16，1 轮（基准） | 77.28% |
| rank 64，1 轮 | 70.36%（过拟合） |
| rank 128，1 轮 | 发散 |
| rank 16，3 轮 | 73.51%（饱和） |

### 4. 突破尝试 ②：挑题专门训练（PED）—— 彻底失败

只挑 2904 道 near-miss 题训练 → 学生塌到 **15.42%**（比瞎猜 20% 还低）。只刷"错题本"导致灾难性遗忘。

结果文件：`data/eval_results_qwen3_ped_ar.json`（combined_student=15.42）

### 5. 结论

**增益是 α=0 GT SFT 的硬天花板（3.75~5.12pp，最高 +5.12 从未过 6%），无法通过加大训练强度或挑题训练打破。** 想"超老师"只能找 headroom < 增益 的组合，不能指望把增益抬更高。

## 目录内容

- `data/train_head_almostright.jsonl`（2904 道 near-miss 题，PED 选题产物）
- `data/qwen3_train_logprobs.jsonl`（Qwen3-32B 在训练集 20488 题的零样本选项分布，供 near-miss 分析）
- `scripts/select_almostright.py`：near-miss 选题（算 logprobs → 筛 near-miss）
- `scripts/run_ped_train.sh`：PED 训练（只训 near-miss 题）
- `scripts/eval_ped.py`：PED 评估（输出塌到 15.42%）

## 依赖

- 英文全量训练集：`fullEnglish/00_data/out/train.jsonl`（20488 题，选题脚本用）
- 英文测试集：`fullEnglish/00_data/out/test_{medqa,medmcqa,mmlu,pubmedqa}.jsonl`
- 基座模型：`../models/Qwen3-32B`
- 训练代码：`../shared/train_choice_head_distill.py`
- 消融（rank64/128/epoch3）复用 fullEnglish 主实验 `fullEnglish/03_main_distill/runs/32B_ab_*`

## 执行步骤（PED 完整复现）

```bash
cd 24_gain_4percent_exploration

# 1. near-miss 选题（Qwen3-32B 在 20488 题训练集算零样本分布，筛"差点答对"题）
python scripts/select_almostright.py

# 2. PED 训练（只训 2904 道 near-miss 题，α=0）
bash scripts/run_ped_train.sh

# 3. PED 评估（预期：学生塌到 ~15.42%，低于瞎猜 20%）
python scripts/eval_ped.py
```

## 关键坑

- PED 选题脚本要跑 ~2.5 小时（20488 题逐题前向）；选题产物已随文件夹附带（data/ 下），可跳过第 1 步直接训练。
- Qwen3 必须 `DISTILL_USE_CHAT_TEMPLATE=1` + 关 thinking。
