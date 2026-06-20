# Prior-Art 第三轮检索结果（D2 容量预测 + 体检工具）

> 日期：2026-06-20
> 方法：arXiv API 直接检索（web_search 后端未配置，改用 arXiv API）。
> 局限：只覆盖 arXiv，未覆盖 Google Scholar/Semantic Scholar 的非 arXiv 论文（如部分会议论文）。
> ⚠️ 完整 prior-art 仍需补充 Scholar 检索（待 web 后端配置或人工补查）。

---

## 1. 最关键发现：Apple《Distillation Scaling Laws》(2025) — D2 的直接 prior-art

- **arXiv: 2502.08606**（2025，真实存在，已核实摘要）。
- **它做了什么**：提出蒸馏缩放定律，根据**计算预算**及其在师生间的分配，估计蒸馏模型**性能（loss）**。给出"计算最优"的师生分配方案，覆盖"教师已存在"和"教师需训练"两种场景。
- **与 D2 的关系（必须正视）**：
  - **重叠**：都在研究"学生规模 vs 蒸馏后表现"的可预测性。这是直接相关工作，**撞车风险：中-高**。
  - **D2 的差异点（需坐实才能成立）**：
    1. Apple 预测的是 **cross-entropy loss / 通用性能**；D2 关注**特定下游任务的 MCQ 正确率**（loss 低 ≠ 选择题正确率达标，二者非线性）。
    2. Apple 用 **compute budget** 做自变量；D2 想用**任务难度**（教师熵/跨模型共识，5d/C 已验证可量化）做容量预测特征——**"难度感知"是 Apple 没有的维度**。
    3. Apple 是大规模预训练式蒸馏；D2 是 **Choice-Head 轻量蒸馏 + 医学领域**的小规模实证。
  - **诚实判断**：纯"容量 vs 正确率曲线"会被 Apple 覆盖，**D2 的新意必须押在"难度感知"上**——即"用任务难度预测达标所需最小容量"，否则是 Apple 的子集。

---

## 2. 其他相关命中（相关性较弱，arXiv 关键词匹配噪声大）

| 论文 | arXiv | 与本研究 |
|---|---|---|
| Capacity Dynamic Distillation for Efficient Image Retrieval | 2303.09230 (2023) | "学生容量"相关，但是图像检索+动态容量，非容量下限预测 |
| A Functional Perspective on KD | 2510.12615 (2025) | KD 理论视角，需核实是否涉及容量 |
| 其余命中（Triplet Loss KD、Feature Maps KD 等） | 多为2018-2020 | 经典 KD 方法，与"容量预测/不确定性筛选"关系弱 |

> 注：方向"不确定性蒸馏筛选"和"难度↔不确定性验证"的 arXiv 检索**命中质量差**（返回大量无关物理/密码学论文），说明 arXiv 全文关键词匹配对这些 query 不灵敏。**这两个方向的 prior-art 必须靠 Google Scholar/Semantic Scholar 补查**，当前不能下"无人做过"的结论。

---

## 3. 撞车风险评级（基于当前不完整检索）

| 方向 | 撞车风险 | 依据 | 差异化策略 |
|---|---|---|---|
| D2 容量下限预测 | **中-高** | Apple 2502.08606 直接相关 | 必须押"难度感知"+"下游MCQ正确率"+"医学领域"，不做纯scaling law |
| 体检工具（不确定性筛选） | **未知（检索不足）** | arXiv 命中差，需Scholar补查 | 押"外部难度金标准验证"角度（5d），但要先确认无人做过 |

---

## 4. 诚实结论与待办

1. **D2 不是空白领域**——Apple 已做通用蒸馏缩放定律。D2 要活，必须聚焦"难度感知的下游正确率容量预测"，并在论文里**明确引用并区分** Apple 2502.08606。否则审稿人会直接拒。
2. **体检工具方向的 prior-art 检索不充分**——arXiv 关键词匹配失败，**必须补 Google Scholar / Semantic Scholar**（需配置 web 搜索后端，或人工检索）。在补齐前，不能宣称"外部难度验证不确定性"是新颖的。
3. **行动项**：
   - 配置 web 搜索后端（Tavily/Brave/Serper API key）→ 让 web_search 可用 → 补 Scholar 检索。
   - 或人工在 Google Scholar 搜 "uncertainty distillation sample selection"、"data difficulty human agreement model confidence" 等，补充本文档。
4. **对价值排序的影响**：D2 的论文价值评分（之前给4）应**下调到 3-3.5**，因为 Apple 已占领"蒸馏缩放定律"高地，D2 只能做其"难度感知"的细分增量。经济价值不变（仍高）。

---

## 5. 一句话

**D2 最大的 prior-art 是 Apple《Distillation Scaling Laws》(2502.08606)——它没做"难度感知"和"下游MCQ正确率"，这是 D2 仅存的差异化空间。
体检工具方向的 arXiv 检索不充分，结论待 Scholar 补查后再定。
诚实提醒：在补齐文献前，不对任何方向宣称"完全新颖"。**
