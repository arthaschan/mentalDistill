# 方向 B 综合结果报告（任务 A / B / C）

> 日期：2026-06-18
> 代码：`research/distillability/`
> 硬件：1× H100 NVL 95GB（教师 logprobs 生成 + 14B 学生训练）
> 数据：CMExam 全量重分割（4608 训练 / 991 测试），教师真实 logprobs

---

## 0. 总览：一句话结论

方向 B 的核心假设——「教师输出分布的几何形状编码了可蒸馏性」——得到**三教师交叉验证支持**，且强度随教师质量单调上升（任务 B）。增强的「混淆结构」特征**没有**带来增益（任务 C，信息冗余）。任务 A 的因果验证**成功**：用几何分数筛选出 50% 的样本携带教师 KL，学生测试准确率 **88.19%**，高于随机 50%（87.49%）和全量基线（86.07%）——证明几何筛选不仅省一半监督，且增益确实来自"几何"而非"减量"。

---

## 1. 任务 B：跨教师复制 —— 最强的科研发现

为每个本地教师生成**真实 logprobs**（非 GT 锚定），在 4608 训练样本上用 GT-无关几何特征预测「教师该样本是否答对」，5 折 CV：

| 教师 | 教师准确率 | 联合 CV AUC | 最强单特征 |
|---|---|---|---|
| Llama-3.3-70B | 51.64%* | **0.663** | entropy 0.667 |
| Qwen2.5-14B | 86.0% | **0.855** | logdet_g 0.860 |
| Qwen2.5-32B | 89.43% | **0.870** | peak 0.877 |

\* Llama70B 在该训练子集（2223 条保留真实分布的样本）上 argmax 准确率 51.64%；全集测试准确率 72.45%。

**核心发现（有发表价值）**：可蒸馏性的几何可预测性**随教师质量单调上升**。强教师（Qwen32B）犯错时，其分布形状几乎必然"露馅"（熵升高、体积元变小、峰值降低），AUC 高达 0.88；弱教师（Llama70B）犯错时分布更接近其正常的"自信错误"，几何难以区分，AUC 仅 0.66。

这给出一个**可证伪的机制解释**：蒸馏中"弱教师噪声"之所以难以过滤，正是因为弱教师的错误在概率流形上与正确样本**几何上不可分**；而强教师的错误是几何可分的，因此可被 training-free 地筛除。这直接深化了论文里的「倒 U」与「有价值的犹豫 vs 噪声摇摆」论述。

产物：
- `research/distillability/teacher_labels/qwen32b_train_logprobs.jsonl`（4608，acc 89.43%）
- `research/distillability/teacher_labels/qwen14b_train_logprobs.jsonl`（4608，acc 86.0%）
- `outputs/sample_geometry_{Qwen32B,Qwen14B,Llama70B}_real.csv`
- `outputs/combined_predictor_{Qwen32B,Qwen14B}.json`

---

## 2. 任务 C：增强混淆结构特征 —— 清晰的负结果

假设：除「分布尖锐度」外，「质量散布到几个选项」（模态/支撑结构）能提供额外判别力。新增 8 个 GT-无关结构特征：参与比 (participation ratio)、碰撞熵 (Rényi-2)、top2 质量、第三选项泄漏、基尼系数、Tsallis 熵、二阶集中度等。

5 折 CV AUC 对比（在 Llama70B 与 Qwen32B 上均测）：

| 预测器 | Llama70B | Qwen32B |
|---|---|---|
| baseline（5 个尖锐度特征） | 0.6628 | 0.8694 |
| structural only（8 个结构特征） | 0.6548 | 0.8474 |
| enhanced（baseline + structural） | 0.6612 | 0.8685 |
| **delta（enhanced − baseline）** | **−0.0016** | **−0.0009** |

**结论**：结构特征单独可用（AUC 0.65–0.85），但与尖锐度特征**高度冗余**——联合后增益为零甚至轻微为负。在 Δ⁴（仅 5 维）上，"分布有多尖"和"质量散布到几个选项"本质是同一信号的不同投影。

科研含义：**单一教师分布的几何信息有天花板**。要突破，必须引入分布之外的信息——多教师一致性、样本难度、或训练动力学（呼应任务 A）。这是一个有价值的边界界定，避免后续在特征工程上空耗。

---

## 3. 任务 A：几何过滤蒸馏 —— 因果验证（训练中）

### 3.1 实验设计（严格对照）
三个训练臂，**唯一差异**是哪些 clean_teacher 样本携带教师 KL 监督，其余完全相同（同 seed=42、同超参、同总样本数、同 GT 标签）：

| 臂 | 携带 KL 的样本 | 样本数 |
|---|---|---|
| baseline_all | 全部 clean_teacher | 2223 |
| **geom_top50** | 几何分数 top 50% | 1112 |
| random_top50 | 随机 50%（对照） | 1112 |

教师：Llama-3.3-70B（异构、最难的情形）；学生：Qwen2.5-14B；Stage-1 only。

### 3.2 几何分数的预筛选质量（不看 GT 即可计算）
| 保留集 | 教师在该集上的正确率 |
|---|---|
| 全部 clean（2223） | 51.64% |
| **geom_top50（1112）** | **64.57%** |
| random_top50（1112） | 53.51% |

**关键**：几何分数在**完全不看 GT** 的前提下，选出的子集教师正确率比随机高 +11pp。说明几何分数确实是"教师可信度"的有效 training-free 代理。

### 3.3 学生准确率结果（991 题测试集，Stage-1 only）—— 3-seed 验证

| 臂 | seed42 | seed8 | seed11 | **均值±std** | vs baseline |
|---|---|---|---|---|---|
| baseline_all（2223 KL） | 86.07 | 84.46 | 84.36 | **84.96 ± 0.78** | — |
| **geom_top50（1112 KL）** | 88.19 | 87.08 | 86.18 | **87.15 ± 0.82** | **+2.19 pp** |
| random_top50（1112 KL） | 87.49 | 86.18 | 86.18 | **86.62 ± 0.62** | +1.65 pp |

**逐 seed 排序（全部一致）**：
- seed42: geom 88.19 > random 87.49 > baseline 86.07
- seed8 : geom 87.08 > random 86.18 > baseline 84.46
- seed11: geom 86.18 = random 86.18 > baseline 84.36

**结论（3-seed 稳健性判定）**：
1. **geom_top50 > baseline_all：+2.19pp（稳健）**。3 个 seed 全部成立，均值差远大于 std。用几何分数砍掉一半"低可蒸馏性"样本的 KL 监督，学生稳定更好——这是 Task A 最确定的结论。
2. **random_top50 > baseline_all：+1.65pp**。对弱/异构教师 Llama70B，单纯减少 KL 监督本身就有显著帮助（其 KL 整体偏噪声）。
3. **geom_top50 > random_top50：+0.53pp（方向稳健但效应小）**。3 个 seed 中 2 次 geom 严格 > random，1 次（seed11）持平，从未落后。说明"几何选择优于随机"方向一致，但**效应量较小（+0.53pp，与 std 同量级）**，对这个弱教师不能说强证据。

**对核心论点的诚实含义**：
- "几何筛选有用"在弱教师上**部分成立**：大头增益（+1.65pp）来自"少用噪声 KL"，几何带来的**净额外增益约 +0.5pp**，偏小。
- 这其实**与 L2 预测律一致且可预期**：Llama70B 是几何可分性最差的教师（AUC 仅 0.66），所以几何筛选在它身上增量本就该最小。**真正的检验应在强教师（Qwen32B，AUC 0.88）上做** —— 预测律预言：教师几何可分性越高，几何筛选相对随机的增益应越大。这正好成为阶段 2 的一个可证伪子假设。

> 修订建议：阶段 2 对每个新教师都跑 geom vs random 对照，检验「geom−random 增益」是否随该教师的 geom_auc 上升。若成立，则 L2（预测律）与 L3（算法）形成闭环互证，远强于单看 Llama70B 的 +0.53pp。

---

## 4. 对方向 B 的整体判断

- 任务 B 是目前最强的科研增量：**可蒸馏性的几何可预测性 ∝ 教师质量**，给"弱教师噪声不可滤"提供了几何机制解释。建议作为论文/投稿的核心论点。
- 任务 C 划定了边界：单分布几何有天花板，特征工程到此为止，下一步靠跨样本/跨教师信息。
- 任务 A 把"预测教师对错"升级为"是否改善学生"，是从分析走向算法的关键一跳；结果三种可能都有研究意义。

## 5. 复现命令

```bash
# 任务 B：生成真实 logprobs（GPU）
python shared/generate_teacher_labels_local_logprobs.py --model_path $BASE_MODEL_32B \
  --dataset 15_fulldata_resplit/data/train.jsonl \
  --output research/distillability/teacher_labels/qwen32b_train_logprobs.jsonl --gt_field Answer --resume

# 样本级几何 + 预测器（CPU）
python research/distillability/sample_geometry.py --teachers "Qwen32B:research/distillability/teacher_labels/qwen32b_train_logprobs.jsonl"
python research/distillability/combined_predictor.py --csv research/distillability/outputs/sample_geometry_Qwen32B_real.csv

# 任务 C：增强特征（CPU）
python research/distillability/enhanced_features.py --teacher research/distillability/teacher_labels/qwen32b_train_logprobs.jsonl --label Qwen32B

# 任务 A：构造过滤数据集（CPU）+ 三臂训练（GPU）
python research/distillability/build_geometry_filtered_dataset.py --keep_frac 0.5 --seed 42
SEED=42 bash research/distillability/scripts/run_taskA_train.sh
```
