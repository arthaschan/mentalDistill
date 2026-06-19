# 方向 B 文献查重与科研价值评估

> 日期：2026-06-19
> 检索：arXiv API + Semantic Scholar API，18 个查询覆盖三层声明（原始结果见
> `research/distillability/litsearch_results.txt`）。Semantic Scholar 限流严重，部分查询未返回，
> 但已检索到各层最相关的代表性工作，足以判断新颖性。
> 检索局限：未覆盖中文期刊/学位库；个别经典论文（Cho&Hariharan、Mirzadeh TAKD、
> Menon 统计视角）因限流未直接确认，下文按其公认内容评估。

---

## 0. 一句话结论

三层声明**没有发现完全相同的已有工作**。其中：
- **测量层（L1）**：信息几何用于蒸馏的「软标签质量度量」——**接近空白**，是最干净的新颖点。
- **预测律层（L2）**：「教师错误的几何可分性随教师质量单调上升」——**未见直接对应工作**，是最具原创性的核心论点，但需与"教师不总是越强越好"的已有文献划清边界。
- **算法层（L3）**：「training-free 几何样本筛选」——**部分相关工作存在**（label-free 过滤、logit 不确定性蒸馏），新颖性在于"纯输出分布几何 + 无需训练/无需 GT + 随机对照证明增益来自几何"，需明确增量。

总判断：**有发表价值**，定位为「信息几何视角下的教师/样本可蒸馏性」，L2 是论文的核心卖点。属于 workshop→中等会议级别的增量，若三阶段（含前瞻性验证）都成立，可冲更好的场。

---

## 1. 测量层 L1：软标签的信息几何质量度量

**检索结论：接近空白，最干净的新颖点。**

- 检索 "Fisher-Rao distance knowledge distillation" / "information geometry KD soft labels" / "probability simplex distillation geometry"，**没有任何一篇把 Fisher-Rao 距离 / 体积元 log det g / 单纯形边界距离用于度量教师软标签质量**。
- 命中的几何类论文都在做**别的事**：
  - `On Closed-Form Expressions for the Fisher-Rao Distance` (2023)：纯数学，FR 距离闭式解，与蒸馏无关。
  - `Approximations to the Fisher Information Metric ... for OOD Detection` (TMLR 2024)：FIM 用于 OOD 检测，作用在生成模型似然，不是蒸馏软标签。
  - `Pathological Spectra of the Fisher Information Metric in DNNs` (2019)、`Adversarial Attack under Fisher Information Metric` (AAAI 2018)：FIM 作用在**参数空间/输入空间**，不是输出单纯形上的软标签质量。
- 最接近的蒸馏侧工作是 α-散度蒸馏（你论文已引），但那是**损失函数**，不是**质量度量**。

> 边界声明：你的贡献是「把信息几何当作教师软标签质量的**测量工具**」，而非又一个蒸馏损失。这个 framing 在检索范围内未见。

---

## 2. 预测律层 L2：错误几何可分性随教师质量单调上升（核心卖点）

**检索结论：未见直接对应工作；是三层中最具原创性的。但必须与两类邻近文献严格划界。**

邻近文献 1 —「教师不总是越强越好 / 容量鸿沟」：
- `Does Knowledge Distillation Really Work?` (Stanton et al., NeurIPS 2021, 282 引)：指出学生常无法逼近教师分布，蒸馏的"保真度"有限。**与你不同**：它讲学生-教师分布不匹配，没讲"教师错误的几何可分性随质量变化"。
- `Strong Teacher Not Needed? On Distillation in LLM Pretraining` (2026)：质疑"强教师→强学生"假设。**与你不同**：它比较最终学生效果，未从"教师输出分布几何"给机制解释。
- `Better Teacher Better Student` (ICLR 2022, 66 引)、TAKD（Mirzadeh，教师助教）、Cho&Hariharan：都讲**容量鸿沟**导致更强教师未必更好。**与你不同**：归因于容量差距，而非"错误是否几何可分"。

邻近文献 2 —「从输出分布预测正确性 / 置信度校准」：
- `Understanding Softmax Confidence and Uncertainty` (2021, 120 引)：softmax 置信度作为不确定性代理。**与你不同**：通用校准问题，单模型，没有"可分性随模型质量 scaling"的律。
- `No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes` (2025, 21 引)：用问题表征预测 LLM 答对与否。**与你不同**：用 hidden state 线性探针，不是输出分布几何；且不研究"可分性随教师质量变化"。
- `Predicting with Confidence on Unseen Distributions` (ICCV 2021, 145 引)：用置信度预测模型在分布偏移下的准确率。**与你不同**：预测整体准确率，不是逐样本可分性，也无教师质量 scaling 律。

> **你的独特论点**：把"逐样本错误可被输出分布几何识别的程度（AUC）"作为一个量，并发现它**随教师准确率单调上升**（0.66→0.88），用来机制性解释"弱教师噪声为何难以过滤"。这个**scaling 关系 + 蒸馏机制解释**的组合，在检索范围内未见。这是论文最该主打的发现。

风险：N=3 教师（即将扩到 N=7）样本仍小；审稿人会问"这是不是只是'强模型校准更好'的已知现象的重述"。对策：阶段 2 的前瞻性验证 + 明确区分"校准"（置信度数值准不准）与"可分性 scaling"（错误能否被几何识别且随质量变化）。

---

## 3. 算法层 L3：training-free 几何样本筛选

**检索结论：存在部分相关工作，新颖性在"纯输出几何 + 无训练无 GT + 随机对照"，需明确增量、不宜单独作为主贡献。**

最接近的已有工作：
- `uDistil-Whisper: Label-Free Data Filtering for KD` (NAACL 2024, 5 引)：**最接近的威胁**。label-free 地过滤蒸馏数据。**与你不同**：它用教师-学生一致性/伪标签置信度过滤，针对 Whisper 语音；不是"输出分布的信息几何分数"，也没有"几何 vs 随机"的对照来证明增益来自几何本身。
- `Leveraging logit uncertainty for better knowledge distillation` (Scientific Reports 2024, 14 引)：用 logit 不确定性改进蒸馏，且明确提到"更大教师未必更好"。**与你部分重叠**：都用输出不确定性；**不同**：它是加权损失，不是 training-free 样本筛选，也无单纯形几何 framing。
- `Logitwise Distillation Network` (2025)、`CKD sample-wise` (TIP 2024)：引入"样本置信度/可靠性"做逐样本蒸馏。**与你不同**：需训练中计算、非纯几何、无 training-free 教师预筛选。
- 数据估值（data Shapley、datamodels、influence functions）：通常**需要训练**或梯度，你的是 training-free 几何，方向不同。

> **你的增量**：(a) 选择分数纯由输出分布单纯形几何决定，(b) 完全 training-free 且不需要 GT，(c) 用 random-50% 对照证明增益来自几何而非减量（+0.70pp over random）。这个组合未见完全对应，但因邻近工作较多，L3 建议作为 L1/L2 的**算法落地证据**，而非独立主贡献。

---

## 4. 综合科研价值判断

| 层 | 新颖性 | 邻近最强工作 | 定位建议 |
|---|---|---|---|
| L1 测量 | 高（接近空白） | FIM-for-OOD (TMLR24)；FR 闭式解 (2023) | 方法论 framing，铺垫 |
| **L2 预测律** | **最高（核心）** | Does KD Really Work? (NeurIPS21)；Strong Teacher Not Needed (26) | **论文主打卖点** |
| L3 算法 | 中（有邻近） | uDistil-Whisper (NAACL24)；Leveraging logit uncertainty (2024) | 算法落地证据 |

**结论**：研究有真实科研价值，没有被完全做过。最强、最该主打的是 **L2 的"错误几何可分性随教师质量单调上升"这一 scaling 律 + 蒸馏机制解释**。三层组合（几何测量→预测律→training-free 算法且有因果验证）形成一条完整链条，这种"测量-规律-算法"闭环本身比任何单点更有说服力。

**发表定位**：当前证据（N≤7、单数据集 CMExam、5 选项 MCQ）适合 workshop 或中等会议（如 ACL/EMNLP findings、或 KD/efficient-ML workshop）。若阶段 2 前瞻性验证成立、且能扩到多数据集/多任务，可冲更好的场。

**审稿人最可能的攻击点（须预先防御）**：
1. 「L2 只是'强模型校准更好'的重述」→ 用前瞻性验证 + 校准 vs 可分性的区分回应。
2. 「N 太小、单数据集」→ 阶段 2 扩到 N=7 + 多数据集是必须的。
3. 「Δ⁴ 只有 5 维，几何过简单」→ 坦诚承认是边界条件，但正因低维才可解析、可复现；并说明任务头蒸馏的现实意义。

---

## 5. 检索方法与可复现

- 工具：`research/distillability/scripts/litsearch.py`（arXiv + Semantic Scholar API，带限流退避）
- 原始结果：`research/distillability/litsearch_results.txt`
- 局限：Semantic Scholar 免费 API 限流严重，部分查询未返回；未覆盖中文文献库、专利、Google Scholar 全文。建议正式投稿前用机构访问补一次 Google Scholar + ACL Anthology 全文检索，并人工核对第 2、3 节点名出的 5-6 篇最近邻论文全文。
