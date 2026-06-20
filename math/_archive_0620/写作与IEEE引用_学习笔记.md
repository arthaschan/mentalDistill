# 学术写作 + IEEE 引用 学习笔记（提炼 skill 前的素材）

> 日期：2026-06-19
> 来源 1：academic-research-skills 项目（方法论，用我们自己的话重述）
> 来源 2：Victoria University IEEE Referencing Guide（IEEE 格式，用户明确需要）
> 用途：学习完整后提炼成 Hermes skill `academic-writing-guidance`。本文件是素材汇总。

---

# 第一部分：写作质量规则（来自 writing_quality_check 全文 A–E）

## A. 高频"塑料词"警告（不是禁用，是提醒"这是最精确的词吗？"）
delve, tapestry, landscape, pivotal, crucial, foster, showcase, testament, navigate,
leverage, realm, embark, underscore, multifaceted, nuanced, comprehensive, robust,
intricate, cornerstone, paradigm, synergy, holistic, streamline, cutting-edge, groundbreaking。
- **例外规则**：若该词是目标学科的标准术语则豁免（统计里的 "robust estimator"、科学哲学的 "paradigm shift"、生态/地理里字面的 "landscape"）。

## B. 标点模式控制
- **破折号 (—)**：全文 ≤3 个，建议 0–1。AI 文本滥用破折号做插入语；学术写作改用逗号/括号/拆句。
- **分号**：≤2 个/千词。别用分号串独立子句，该用句号就分句。
- **冒号-列表**：避免连续 2+ 段都以"冒号+列表"开头，单调。

## C. 清嗓子式开头（删掉，直接进入主题）
"In the realm of…" / "It's important to note that…" / "It is worth mentioning…" /
"In today's rapidly evolving…" / "This serves as a testament to…" / "In order to…"(→"To…") /
"It should be noted that…" / "When it comes to…" / "With that being said…"。
- **元评论**也要避免："This section will discuss…"→直接讨论；"We now turn to…"→直接转。
- **例外**：引言里的路线图句（"Section 2 reviews…; Section 3 describes…"）是标准做法，保留。

## D. 结构模式警告
- **三点强迫症**：不是每个论证都该拆成 3 点。2 个有力的点胜过 3 个注水的。证据需要几点就几点。
- **段落等长**：自然写作长短交错。短段强调，长段展开复杂论证。
- **同义词循环**：一段里给同一概念换 3+ 个同义词（students→learners→participants→subjects）会造成混乱。**学术写作里术语一致是美德**，一个概念一个词，重复即清晰。
- **二元对比滥用**："不是 X，是 Y" 这种修辞全文 ≤2 次，多了变 tic。
- **镜像结构**：别让每节都是"主题句→3 证据→综合句"的模板冲压感，不同节应有不同内部节奏。

## E. Burstiness（句长变化）
- 好写作句长自然变化：短句造冲击，长句展开复杂概念。
- **检测**：若连续 5+ 句都落在窄的字数区间（如都 20–25 词）→ 标记修订。
- **按节的句长变化目标**：摘要中等、引言高、文献中等、方法低（程序性可均匀）、结果中等、讨论最高。

## 使用方式（重要立场）
- **目标是更好的散文，不是规避 AI 检测器**（明确不做 humanizer）。
- 起草时边写边自检最好；评分仅内部用，不报给用户，发现问题静默修复。

---

# 第二部分：写作判断框架（writing_judgment_framework）

## 清晰度测试（每段问一遍）
"删掉这段，论文还成立吗？"
- 成立且无损 → 删
- 成立但失语境 → 保留但精简（支撑段）
- 不成立、论证断裂 → **承重段，值得最仔细打磨**
- 核心洞察：**多数 AI 学术文本失败，是因为每段都被给了相同权重**。要区别投入：承重段多稿，支撑段一稿。

## 读者旅程（任何位置读者都该能答）
1. 我在哪？（结构/路标）2. 为何在此？（与研究问题的连接）3. 带走什么？（本节要点）4. 下一步去哪？（过渡逻辑）
- 任一不清楚就需修订，无论内容多准确。

## 学科语域
- 硬科学：客观、被动、对冲；可信靠测量与方法的精确。
- 社科：半人称、主动、限定；可信靠对局限的透明。
- 人文：人称、论辩、阐释；可信靠对文献的深度engagement。
- 工程：直接、命令式、指标驱动。

---

# 第三部分：摘要 5 部件模型（abstract_writing_guide）

结构化：Background, Purpose/Objective, Method, Results, Conclusion。
非结构化（单段流）：Context→Problem→Purpose→Method→Findings→Implications。
- 字数：标准 150–250；会议 200–500（看 CFP）；学位论文 ≤350。
- 句式：Background "Despite growing interest in X, little is known about…"；Purpose "This study examines…"/"This paper proposes…"。
- **避免**：用 "This paper…" 突兀开头；写 "教育很重要" 这种空泛句；过长历史背景。

---

# 第四部分：论文结构模型（6 种，paper_structure_patterns）

对你（ML/IEEE 方向）最相关的是 **Conference Paper** 和 **IMRaD**：

**Conference Paper（短文 2000–5000 词）**：Title/Authors → Abstract(100–200)+Keywords →
Introduction(问题+RQ) → Related Work(关键先行+本文填的 gap) → Approach/Methodology(设计/数据/分析)
→ Results(关键发现+图表) → Discussion(解释+局限) → Conclusion & Future Work → References。
字数分配（3000 词例）：Intro 15% / Related 20% / Method 20% / Results 25% / Discussion 12% / Conclusion 8%。

**IMRaD（实证研究 5000–8000 词）**：Intro(背景/问题/gap/RQ/意义) → Lit Review → Methodology
→ Results → Discussion(总结/对比文献/理论与实践含义/局限/未来) → Conclusion。

其余 4 种（Thematic Lit Review / Theoretical / Case Study / Policy Brief）备查。

---

# 第五部分：对冲语保护（protected_hedging_phrases）★与我们科研诚实直接相关

- 核心：压缩（如压字数写摘要）时容易丢掉正文的限定词，导致**摘要过度声称 = 发表诚信问题**。
- 规则：**"丢掉它会改变真值声明的短语"必须被预算保护**，优先于任何压缩目标。
- 三类受保护短语：(1) 限定声明范围的认知对冲（may, might, suggests, tentative, preliminary, appears to）；
  (2) 限定适用边界（in this institutional context, within this sample, under these conditions）；
  (3) 统计/方法限定（single-seed, N small, effect size comparable to std）。
- **对我们方向 B 极相关**：写结论时 "单 seed/N=7 偏小/+0.53pp 与 std 同量级/几何可分性最弱的教师增量本就最小" 这类限定，绝不能为简洁而删。

---

# 第六部分：IEEE 引用格式（Victoria University Guide，用户必需）

## 6.1 in-text 规则
- 引用是方括号数字 [X]，对应文末参考列表；从 [1] 起按**出现顺序**升序；重复引用复用原编号。
- 方括号在文字行内、句末标点**之内**、括号前留一个空格：
  "…simplification [13]." 写 "in [1]…" 而非 "in reference [1]…"。
- 多引用：[1], [2], [3] 或连续区间 [1]-[3]。
- 引用可作语法成分：直接 [1] 或 "Ozansoy [7] argued…"。

## 6.2 页码/缩写（用户给的 NumbersAbbreviations 页）
- 引用整篇/转述长段/单页文献时**无需**页码；但定位具体理论/观点时可给。
- 单页 p.，多页 pp.：[5, p. 17]、[5, pp. 6-12]。
- 其它定位：[4, para. 4.2]、[6, Ch. 2, pp. 7-13]、[8, Fig. 33]、[7, Tab. 14]、[6, eq. (8)]、[8, Appendix IV]。

## 6.3 期刊文章格式（电子）
```
[#] A. B. Surname, "Title of the article," Abbrev. Title of Journal, vol. x, no. x,
    pp. xxx-xxx, Abbrev. Month Year, doi:xxxxx.
```
- 文章标题加双引号、小写（句首大写）；期刊名斜体、重要词首字母大写（and/of/on 等小词不大写）；期刊名用标准缩写。
- 有 DOI 优先给 DOI，可省 Access Date。

## 6.4 会议论文格式
- 印刷版：`[#] A. B. Surname, "Title of paper," presented at the Abbrev. Title of Conf., City, State/Country, Year.`
- 在线版例：`[3] J. Roberts and D. Fisher. (14-17 Dec. 2020). pReview: …. Presented at the 19th IEEE Int. Conf. Mach. Learn. Appl. (ICMLA), Miami, FL, USA. [Online]. Available: https://ieeexplore.ieee.org/document/9356281`
- 会议名用标准缩写（Int. Conf., Proc., Mach. Learn. 等）。

## 6.5 图表公式
- **自制**的图表公式无参考文献，但正文必须引用："shown in Fig. 1", "given in Table 1", "Equations (1), (2)"。
- 全文图/表/公式从头到尾连续编号。
- 引用他人的：[5, Fig. 1]、[5, Tab. 3]、[5, eq. (2)]。**不要**写 "in Fig. 1 of reference [5]"，直接 "in [5, Fig. 1]"。

## 6.6 生成式 AI 披露
- 多数机构（如 VU）要求：用 AI 生成内容而不披露 = 抄袭。
- AI 输出可能错误/捏造、无参考文献、要求生成引用时可能伪造 → 必须批判性核验。
- 用前确认是否被允许；用后需按要求披露（很多 IEEE 投稿也有 AI 使用披露政策）。
- **与我们相关**：本项目大量用 Hermes，正式投稿时需按目标会议/期刊的 AI 披露政策声明。

---

# 第七部分：中英双语写作工作流（本项目实操经验，2026-06-19 确立）

> 用户母语为中文。论文写作采用"中文为语义基准 → 翻译英文 → 英文母语化润色"的三段式流程。

## 7.1 工作流（严格按序）
1. **中文为语义基准**：先用中文把论点、逻辑、实验叙述写准、写清。语义、结构、论证的正确性以中文版为准（user 看得懂、能校验）。
2. **翻译为英语**：忠实翻译，不在翻译阶段擅自改写论证或增删内容。
3. **英文母语化润色**：对译文做一次 native-speaker 级润色——
   - 拆长句（中文常见的长句在英文里要断成多句，提升可读性）
   - 补冠词（a/an/the，中文无冠词，是中→英最高频的遗漏）
   - 统一术语（一个概念一个英文词，全文一致；呼应第三部分 D 的"术语一致是美德"）
   - 调整为英文学术惯用语序、搭配、时态

## 7.2 润色阶段的红线（绝对不改）
母语化润色**只动语言表达，不动事实层**。以下五类严禁在润色阶段改动：
1. **实验数字**（准确率、增益、p值、样本量、seed 数等——任何数值）
2. **公式**（符号、下标、变量定义）
3. **引用**（编号、文献条目、引用位置）
4. **IEEE 版式**（章节编号、图表编号、参考文献格式、列表样式）
5. **实验结论的方向与强度**（不能把"几何打平"润色成"几何更优"这类语义漂移）

> 一句话：润色让英文读起来像母语者写的，但论文的科学内容必须和中文基准版逐项一致。润色后应能用第一部分 A–E 的写作质量规则自检（塑料词/破折号/清嗓子开头/句长变化），但任何修改都不得越过上述红线。

## 7.3 与诚实护栏的关系
- 这条流程天然契合 user 重视的"非粉饰结果"：中文基准版怎么写，英文就怎么传达，不借翻译/润色之机美化结论。
- 若润色时发现英文表达会无意中夸大（如 "significant" 在统计语境的歧义），优先保守表达并回到中文基准核对原意。

---

# 下一步：提炼成 skill
把以上七部分用我们自己的话整合成 Hermes skill `academic-writing-guidance`：
- 触发：写论文/报告/摘要/技术写作 + IEEE 引用 + 中英双语写作时加载。
- 模块：写作质量自检(A–E) / 判断框架 / 摘要模型 / 结构模型 / 对冲语保护 / IEEE 引用速查 / 中英双语三段式工作流。
- 立场：清晰精确的散文，非规避检测；诚实护栏优先于压缩；母语化润色不越事实红线。
