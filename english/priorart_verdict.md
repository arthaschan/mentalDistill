# Prior-art 新颖性判定（arXiv-only 一轮，2026-07-13）

> ⚠️ 诚实限制：本轮 Semantic Scholar 全程被限速(HTTP 429)，只有 arXiv 命中。
> 这是**不完整**检索，投稿前必须人工补 Google Scholar + ACL Anthology + 精读最近邻全文。
> 结论按"arXiv 未见直接匹配"表述，不宣称绝对新颖。

## §0 一句话判定
三层 claim 在 arXiv 均**未见直接匹配**；最强新颖点是 L3(熵↔跨模型共识难度，英文医学MCQ)，L1(英文牙科MCQ蒸馏)是干净的应用空白，L2(融合负结果)需对最近邻划界。

## L1 — 英文牙科 MCQ 的 choice-head 单教师蒸馏
**判定：应用空白（NOVEL as application）**
- arXiv 精确查询 `蒸馏 AND dental AND multiple choice` → **零命中**。
- 最近邻都是通用/医学影像蒸馏，无一针对牙科文本 MCQ：
  - Multiple Teachers-Meticulous Student (2024)：医学**影像**分类，非文本MCQ。
  - Distilling Calibrated Student from Uncalibrated Teacher (2023)：校准角度，非牙科。
- **划界**：我们=英文牙科**文本** MCQ + choice-head + 学生超教师。牙科专科文本 MCQ 蒸馏在 arXiv 无先例。
- 风险：这是"把已知方法用到新领域"的应用型贡献，非方法首创；单独 L1 只值应用小节。

## L2 — 多教师融合负结果（互补存在但不可廉价捕获）
**判定：需对最近邻划界（PARTIALLY covered 的反面）**
- 最近邻：
  - **One Teacher is Enough? (2021)** — 标题即质疑多教师必要性，最需精读划界。
  - **Confidence-Aware Multi-Teacher KD (2021)** — 置信度加权融合，正面方法；我们是**负结果+诚实上界**，方向相反。
  - Different Teachers, Different Capabilities (2026) — 承认教师能力各异做设备端蒸馏，正面利用差异。
- **我们的 delta**：不是"又提一个融合方法"，而是**用零成本 oracle 上界证明**：英文牙科上教师互补真实存在(+3.3pp oracle)，但任何 label-free/CV 诚实路由都抓不到(≤0pp)——且给出跨语言机制对照(中文无互补 vs 英文有互补但不可用)。这种"融合诚实祛魅"角度 arXiv 未见。
- 风险：reviewer 会说"多教师融合没用早有人讲"→ 必须精读 One Teacher is Enough? 划清"我们量化了互补上界与可捕获性的 gap"。

## L3 — 教师熵 ↔ 跨模型共识难度（英文医学MCQ外部验证）★最强
**判定：arXiv 未见直接匹配（strongest NOVELTY）**
- 精确查询 `model uncertainty AND human difficulty (cs.CL/cs.LG)` → **零命中**。
- `LLM entropy question difficulty cross-model consensus` → 全是无关(金融LLM/宇宙学熵)，无真命中。
- 最近邻仅 "When an LLM is apprehensive about its answers"(2025，模型自我不确定性，非跨模型共识难度金标准)。
- **划界**：我们用**跨模型共识错误数**作难度金标准验证教师熵(ρ=0.69)，并做表面artifact null对照——arXiv 无人在英文医学MCQ+蒸馏教师诊断上做这个组合。
- 这与当前中文论文主线一脉相承(H4/5d)，是"审计型"生态位，撞车风险最低。

## 竞品 Limitations = 下一步弹药（精读时补）
- Confidence-Aware Multi-Teacher KD：假设置信度可靠→我们正好验证/证伪这个假设(外部金标准)。
- Distillation Scaling Laws (2025 ICML)：只预测 loss 不做下游正确率/难度/跨域(与中文论文同款缺口)。

## 待办（投稿前硬性）
1. 人工补 Google Scholar + ACL Anthology(arXiv 漏中文期刊/会议正刊)。
2. 精读全文 3 篇最近邻：One Teacher is Enough?(2021)、Confidence-Aware Multi-Teacher KD(2021)、When an LLM is apprehensive(2025)——逐条对照"是否做了我这一层"。
3. S2 限速补跑一轮(可放 GPU 夜跑时挂)。
