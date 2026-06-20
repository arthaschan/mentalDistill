# 方向 B 正式 Prior-Art 检索清单 + 两个关键问题的回答

> 日期：2026-06-19
> 用途：(1) 回答「几何输了为什么 B+C 还能发期刊」；(2) 列出可蒸馏性的替代指标；(3) 给出正式投稿前必做的 prior-art 检索清单（查询词 + 数据库 + 判定标准）。
> 前置：已有一轮检索见 `math/方向B_文献查重与科研价值.md`（聚焦"几何"框架）。本轮重心扩到"training-free 教师/数据选择"这个更大的问题，因为论文重心可能从几何转向它。

---

## 第一部分：几何输了，为什么 B+C 还能发期刊？

先厘清一个关键认知误区：**"几何输了" ≠ "研究失败"。**

你的论文真正的科学问题不是"几何这个特定工具好不好用"，而是：

> **能否在不训练学生的前提下，预测一个教师值不值得蒸馏？**

几何只是你**第一个尝试的预测器**。这个问题本身的价值，不取决于几何赢不赢。具体分三种情况，每种都能发：

### 情况 1：几何赢了（DI 相关 > 准确率 & 熵）
- 直接成立："信息几何提供了超越朴素置信度的可蒸馏性信号"。这是最强的版本，B 主线 + C 解释机制 → 中等以上期刊。

### 情况 2：几何打平（DI ≈ 准确率 ≈ 熵）
- **这恰恰是一个有价值的科学结论**，写法变成：
  > "我们系统比较了输出分布几何、准确率、熵等多种 training-free 信号，发现**它们预测力相当**——意味着可蒸馏性的可预测部分主要由'教师整体可靠性'这一个潜变量驱动，复杂几何不带来额外收益。"
- 这是一个**negative-but-informative result**，期刊（尤其重视可复现性的）是接受的。它告诉社区"别浪费时间在复杂几何上，准确率就够了"——有实用价值。
- C 在这里**变得更重要**：用表征层探针解释"为什么这些表面指标都退化到同一个潜变量"，给出机制。这正是 why。

### 情况 3：几何输了（准确率/熵明显更强）
- 写法："**一个极简的 training-free 基线（小校准集准确率）就能预测可蒸馏性，且优于复杂几何方法**"。
- 这是**反直觉 + 实用**的结论，可发表性甚至不低于情况 1——因为"简单方法打败复杂方法"本身是审稿人喜欢的故事（参考 "Simple baselines beat X" 类论文）。
- C 解释："为什么几何这种细粒度信号反而不如粗粒度准确率"——表征层机制。

### 为什么 B+C 组合在三种情况下都成立
关键在于 **B 和 C 回答的是不同层次的问题**：
- **B = what**：什么 training-free 信号能预测可蒸馏性？（几何？准确率？熵？）
- **C = why**：为什么这个信号有效/失效？表征层发生了什么？

无论 B 的具体答案是哪个指标赢，"建立可蒸馏性预测问题 + 系统比较多指标 + 表征层机制解释"这条**完整的科学链条**都成立。期刊看重的是**问题的重要性 + 方法的严谨 + 结论的可靠**，而不是"你押的那个特定指标必须赢"。

> 一句话：把论文的命脉从"几何"挪到"training-free 可蒸馏性预测"这个问题上，几何赢不赢都只是其中一个 finding，不影响整篇可发表。

---

## 第二部分：可蒸馏性还有哪些替代/补充指标？

这是论文的科学核心——把可蒸馏性预测器做成一个**指标家族的系统比较**，而不是只赌几何。按"是否 training-free"和"信号层次"分类：

### A. 输出分布层（你现在用的层次，training-free，最便宜）
| 指标 | 含义 | 与几何的关系 |
|---|---|---|
| **教师准确率**（小校准集） | 教师在一批有 GT 的探针题上的正确率 | 最朴素基线，你必须打赢它 |
| **教师输出熵** | 软标签的 Shannon 熵均值 | 已知与几何 AUC 几乎等价 |
| **置信度/最大概率（peak）** | top-1 概率均值 | 校准类指标 |
| **margin** | top1 − top2 概率差 | 判别裕度 |
| **ECE（期望校准误差）** | 置信度与真实正确率的偏离 | 校准质量，可能比裸准确率更预测 |
| **几何量**（你的）logdet_g / Fisher-Rao / 边界距离 | 单纯形上的流形几何 | 待验证是否独立 |
| **教师-GT 分歧率** | teacher argmax ≠ GT 的比例 | = 1 − 准确率，共线 |

### B. 教师-学生关系层（training-free 或轻量）
| 指标 | 含义 | 价值 |
|---|---|---|
| **教师-学生分布散度**（KL/JS，零样本学生） | 蒸馏前学生与教师输出的距离 | 可能预测"学生能否学到"，呼应 capacity gap |
| **教师-学生 top-k 重叠** | 两者高概率选项的重叠度 | 轻量、直观 |
| **预测熵差** | 教师与零样本学生的熵差 | 信息增益的代理 |

### C. 表征层（方向 C，需要 hidden state，较贵）
| 指标 | 含义 | 价值 |
|---|---|---|
| **线性探针可分性** | 用教师 hidden state 线性预测正确性的 AUC | 比输出分布更早的信号 |
| **CKA / 表征相似度** | 教师-学生表征对齐度 | 迁移性的经典代理 |
| **各向异性 / 有效维度** | 表征空间的几何结构 | 机制解释 |

### D. 迁移性文献的现成指标（最该对标的！）
这是**最危险也最该借鉴**的一类——迁移学习社区已经有成熟的 training-free 打分器：
| 指标 | 出处 | 与你的关系 |
|---|---|---|
| **LogME** | Log Maximum Evidence (ICML 2020) | 经典 training-free 迁移性打分，**你必须对标** |
| **LEEP** | Log Expected Empirical Prediction (ICML 2020) | 用源标签预测目标，training-free |
| **NCE / H-score** | 迁移性度量 | 同上 |
| **TransRate** | 互信息式迁移性 (ICML 2022) | 同上 |

> **关键洞察**：你的"可蒸馏性预测"在数学上非常接近"transferability estimation"。LogME/LEEP 这类指标**就是为"不训练就预测一个源模型值不值得迁移"设计的**。如果你不对标它们，审稿人会直接问"为什么不用 LogME？"。反过来——**把 LogME 类指标搬到"教师选择"场景并比较，本身就是一个有价值的贡献点**（迁移性指标→蒸馏教师选择的桥接）。

### 建议的论文核心实验
把上述指标做成一张大表：**N≥7 教师 × M 个指标 × 真实蒸馏增益的排序相关**。结论无论是"几何赢""准确率赢"还是"LogME 赢"，都是一个干净、系统、可发表的 benchmark 式贡献。这比只赌几何稳健得多。

---

## 第三部分：正式 Prior-Art 检索清单

### 3.1 必查数据库（按优先级）
1. **arXiv API**（已有 harness，可自动）— cs.LG/cs.CL 最新
2. **Semantic Scholar API**（已有，限流）— 引用网络
3. **ACL Anthology** 全文（人工/半自动）— NLP 蒸馏工作集中地
4. **Google Scholar**（人工，需机构访问）— 覆盖最全，查被引
5. **OpenReview**（人工）— ICLR/NeurIPS 投稿+审稿意见，能看到最新未发表工作
6. **DBLP** — 确认作者与版本
7. **中文库**（CNKI/万方）— 避免与中文学位/期刊撞车（上轮未覆盖）

### 3.2 检索查询词（按主题分组，中英对照）

**主题 1：Training-free 教师选择 / 可蒸馏性预测（新重心，最关键）**
- "training-free teacher selection knowledge distillation"
- "predicting distillation performance without training"
- "which teacher to distill from" / "teacher selection distillation"
- "distillation gain prediction" / "distillability metric"
- "estimate knowledge distillation effectiveness a priori"
- 中文："免训练 教师选择 知识蒸馏" / "蒸馏 收益 预测"

**主题 2：迁移性估计（必对标，否则被毙）**
- "transferability estimation LogME LEEP"
- "transferability metric source model selection"
- "TransRate H-score model selection transfer learning"
- "transferability estimation knowledge distillation"（交叉点，查是否有人已做）

**主题 3：教师质量 vs 蒸馏收益（你的 L2 律）**
- "teacher accuracy distillation student performance relationship"
- "stronger teacher worse student capacity gap"
- "does knowledge distillation really work fidelity"
- "calibration error knowledge distillation teacher selection"

**主题 4：置信度/熵预测正确性（朴素基线的来源）**
- "softmax confidence predict correctness training-free"
- "entropy uncertainty model selection distillation"
- "expected calibration error predict transfer accuracy"

**主题 5：几何/信息几何（你原框架，确认仍空白）**
- "Fisher-Rao distance knowledge distillation"（已查，复核新文）
- "information geometry teacher soft label quality"
- "probability simplex geometry distillation"

**主题 6：表征层探针（方向 C 的查重）**
- "linear probe predict model correctness hidden states"
- "CKA representation similarity distillation transfer"
- "representation geometry knowledge distillation layer"

### 3.3 判定标准（每篇近邻论文逐条核对）
对每篇命中的高相关论文，填一张表：
| 字段 | 说明 |
|---|---|
| 问题是否相同 | 是否也做"训练前预测教师/源值不值得用" |
| 信号是否相同 | 用的是几何？准确率？LogME？探针？ |
| 是否 training-free | 还是需要训练/梯度 |
| 任务域 | 分类/生成/MCQ/语音… |
| 与我的差异 | 一句话能否说清"我做了什么他没做" |
| 威胁等级 | 高（几乎相同）/中（部分重叠）/低（仅相关） |

**红线**：如果任何一篇在"问题+信号+training-free+任务域"四项全中，必须重新定位贡献或换角度。

### 3.4 执行方式
- 自动部分：扩展 `litsearch.py` 加入主题 1/2/6 的新查询，后台跑（CPU，不抢 GPU）。
- 人工部分：实验出结果、确定论文重心后，针对最终主张做一轮 Google Scholar + ACL Anthology + OpenReview 精查，人工读 top-10 近邻全文。
- 产出：更新 `math/方向B_文献查重与科研价值.md`，新增"training-free 教师选择 + 迁移性指标对标"两节。

---

## 第四部分：下一步

1. 【自动，现在就起】扩展 litsearch.py，后台跑主题 1/2/6 的新检索（不抢 GPU，与 Yi 蒸馏并行）。
2. 【等实验】Yi/Gemma 蒸馏跑完 → 跑 h1_baseline_comparison.py → 确定哪个指标赢。
3. 【关键加项】把 LogME / LEEP 实现进来，加入指标家族对比——这是从"几何论文"升级到"training-free 可蒸馏性 benchmark"的关键，也是防审稿人"为什么不对标迁移性指标"的必备。
4. 【实验出结果后】人工精查最终主张的 prior-art。
