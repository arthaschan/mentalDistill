# AIEA 投稿论文 参考文献人工核对指导文档

> 论文：Choice-Head Distillation for Dental Multiple-Choice Question Answering
> 用途：审稿人抽查文献 9 发现信息错误，要求逐条复查。本文件供你（陈天元）一篇一篇人工核对。
> 核对完成的标准：14 篇全部走完 + 正文里每个 [n] 出现的位置都确认引用合理。只有两者都做完，才能说"没问题"。
> 核对源：本地下载的 PDF（thesis/reference 目录）+ arXiv 官方 + Crossref 官方数据库。所有信息均非凭记忆，均已实查。

---

## 第 0 部分：先看这里——本次核对的总结论（最重要）

逐条实查 14 篇后，按严重程度排序：

| 严重度 | 文献 | 问题 | 必须改否 |
|---|---|---|---|
| 🔴 致命（审稿人抽中的就是它） | [9] CMExam | **作者整组写错/虚构**：论文写 "T. Liu, M. Yang, Z. Lu, Q. Chen, S. Zhou, S. Xiang"，真实作者一个都对不上（真实第一作者是 Junling Liu）。会议信息也不完整。 | **必须改** |
| 🟡 中等 | [6] Singhal | 作者缩写链多处拼错/张冠李戴（"L. Gaeabler" 应为 L. Gabler、"S. Natarajan" 应为 V. Natarajan 等）。该文 **32 位作者 > 8，按规则用 et al.**。 | 建议改 |
| 🟡 中等 | [13] Lam | 作者表整组套错（写成 Chau 团队，实为 Lam+Ling+Mao… **9 位 > 8，用 et al.**）。 | 建议改 |
| 🟡 中等 | [12] Chau-Prosthodontic | 作者表套错（漏 3 位、误加 Lo）。官方 **8 位，在 6–8 区间，应全列**。 | 建议改 |
| 🟡 轻微 | [14] HuatuoGPT | 第 5 位 "N. Chen" 应为 Z. Chen；官方 **> 8 位，用 et al.**，并补页码。 | 建议改 |
| 🟢 可选 | [1] Hinton | 会议名/年份写法可优化（NIPS 2014 Workshop）。 | 可选 |
| ✅ 正确 | [2][3][4][5][7][8][10][11] | 经本地 PDF + Crossref 核对，信息准确。 | 不用改 |

> 注：上表"文献编号"用的是 docx 参考文献列表里的实际行号顺序（[1]=Hinton…[9]=CMExam…[14]=HuatuoGPT）。下面逐条展开。

---

## 第 1 部分：怎么读这份文档（核对方法说明）

### 1.1 IEEE 引用语法是什么意思

你的论文用的是 **IEEE 数字引用格式**。规则：

- 正文里用方括号数字指代文献，例如 `[1]`、`[5]`。这个数字 = 文末"References"列表里的第几条。
- 多篇连引写成 `[1], [2], [3]`（每个数字各自带方括号、逗号分隔），**不是** `[1][2][3]`，也不是 `[1-3]`（范围式 IEEE 也允许 `[1]–[3]`，但本文统一用逗号式）。
- 一条参考文献的标准字段顺序：
  `作者. "标题," 期刊/会议名, 卷 vol., 期 no., 页码 pp., 月 年.`
  - 期刊文章：要有 vol./no./pp./年。
  - arXiv 预印本：写 `arXiv preprint arXiv:XXXX.XXXXX, 年`。
  - 会议论文：写 `in Proc. 会议简称, 地点, 年`。

### 1.2 每条核对要回答的 5 个问题（你人工核对时照这个清单走）

对每一篇被引文献，确认：
1. **这个 [n] 在我正文哪里出现？**（定位）
2. **它指向的真实论文是哪一篇？去哪看？**（本地 PDF 路径 / arXiv 链接）
3. **被引论文的哪个位置支持我的引用？**（页码/章节）
4. **我为什么可以引用它？**（我正文那句话的主张，能不能被这篇论文支撑）
5. **被引论文那段的中文意思是什么？**（翻译，确认你没有曲解）

### 1.3 作者列表写多少个？（你导师/IEEE 的规则）

**规则：作者 1–8 位 → 全部列出；超过 8 位（即 ≥9 位）→ 只写第一作者 + "et al."。**

- 这条规则决定了本文档每条"建议改法"是"全列"还是"et al."。各篇按此判定：
  - [9] CMExam 11 位 → **et al.**（关键：第一作者必须是 J. Liu，不是 T. Liu）
  - [6] Singhal 32 位 → **et al.**
  - [13] Lam 9 位 → **et al.**
  - [14] HuatuoGPT >8 位 → **et al.**
  - [12] Chau-Prosthodontic 8 位 → **全列**（在 6–8 区间内，不要用 et al.）
  - 其余 [1][2][3][8] 等 ≤8 位 → 全列（现稿已正确）。
- ⚠️ 用 et al. 时唯一要盯的就是 **et al. 前那个第一作者的姓名缩写有没有写对**——审稿人抓 [9] 抓的正是这个。

### 1.4 哪些字段"看论文 PDF 确认不了"，必须另查（重要）

你人工核对时直接打开被引论文的 PDF，**只能确认一部分字段**。下表告诉你哪些能、哪些不能，以及不能时怎么查：

| 字段 | 看论文 PDF 能否确认 | 不能时怎么查 |
|---|---|---|
| 作者姓名、顺序 | ✅ 能（看首页署名） | — |
| 标题 | ✅ 能 | — |
| 期刊/会议名 | ✅ 能（首页或页眉页脚） | — |
| arXiv 编号 | ✅ 能（首页右上或页脚） | 或上 arxiv.org 搜标题 |
| **DOI 号** | ⚠️ 多数 PDF **首页没有**（尤其 arXiv 预印本完全没有） | **见下方"DOI 查询法"** |
| **卷 vol. / 期 no.** | ⚠️ 期刊正式版首页常有，arXiv 版**没有** | 用 DOI 查询，或期刊官网 |
| **页码 pp.** | ⚠️ 同上，arXiv 版没有正式页码 | 用 DOI 查询，或会议论文集官网 |
| **出版年/月** | 🔶 部分能 | 用 DOI 查询确认正式出版日期 |

#### 📌 DOI / 卷期页 查询法（你逐条核对时照做）

你不必自己记 DOI。**两种最快的人工核对方式，任选其一：**

1. **用论文标题反查（最简单，推荐）**
   - 打开 https://search.crossref.org
   - 把论文**英文标题**粘进去搜索 → 第一条结果点进去
   - 页面会显示：作者全名、期刊名、**卷 vol. / 期 no. / 页码 pp. / 年 / DOI**
   - 拿这些和 docx 里那条参考文献逐字段比对。

2. **用 DOI 直接查（本文档已给出每条的 DOI，你想验证 DOI 本身对不对时用）**
   - 在浏览器地址栏输入 `https://doi.org/` + 本文档列出的 DOI
     例如 [10] Chau：访问 `https://doi.org/10.1016/j.identj.2023.12.007`
   - 能正确跳转到那篇论文的官方页面 = DOI 正确；跳错或 404 = DOI 有问题。
   - 注意：本论文参考文献里**不写 DOI**（IEEE 会议格式不强制），所以 DOI 只是你**核对卷期页的工具**，不需要写进 docx。

3. **arXiv 类（[1][2][4][5][8]）**：直接访问 `https://arxiv.org/abs/编号`（如 https://arxiv.org/abs/2106.09685），核对标题、作者、年份即可。这类没有期刊卷期页，写法到 arXiv 号为止就对。

4. **CMExam [9] / NeurIPS 类会议**：去 https://papers.nips.cc 或 https://proceedings.neurips.cc 搜标题，核对卷次（NeurIPS 2023 = vol. 36）和页码（pp. 52430-52452）。

> 一句话：**作者/标题/会议名看 PDF 首页；DOI/卷/期/页 用标题在 search.crossref.org 反查。** 本文档每条已把这些值替你查好填上，你只需复核"docx 写的 == 本文档给的"即可。

---

## 第 2 部分：14 篇逐条核对卡

> 格式说明：每张卡片对应一条参考文献。
> 「正文出现位置」指 docx 里 [n] 实际出现的句子；「去哪看」给本地 PDF 路径或 arXiv 号；「为何可引」解释这句话为何站得住；「原文翻译」是被引论文关键句的中文。

---

### 📕 [1] Hinton — 知识蒸馏开山之作

- **docx 参考文献原文**：G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," in Proc. NIPS Deep Learn. Representation Learn. Workshop, Montreal, Canada, 2015, arXiv:1503.02531.
- **正文出现位置**：
  - 摘要后 Introduction 第 1 句 `[1], [2], [3]`（"Knowledge distillation compresses large models by transferring soft targets to smaller students"）。
  - 第 29 段 `[1], [2]`（"Current distillation pipelines use full-vocabulary logits"）。
  - Method 提到全词表蒸馏对照。
- **去哪看**：本地 `thesis/reference/05_Distilling_Knowledge_1503.02531.pdf`，第 1 页摘要 + 第 2 节 "Distillation"。
- **为何可引**：你这句话讲"知识蒸馏=把软目标从大模型迁移到小模型"，正是 Hinton 这篇提出的核心思想（soft targets / temperature）。引用合理。
- **被引原文关键句（英→中）**：
  - 原文 Abstract："A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions."
  - 中文："改善几乎任何机器学习算法性能的一个非常简单的方法，是在同一数据上训练多个不同模型并对它们的预测取平均。"（接着引出把集成知识蒸馏进单个小模型）
- **核对要点 / 建议**：
  - ✅ 作者（Geoffrey Hinton, Oriol Vinyals, Jeff Dean → "G. Hinton, O. Vinyals, and J. Dean"）、标题、arXiv 号 1503.02531 全部正确。
  - ⚠️ **年份内部矛盾（你人工核对已发现）**：现稿写 "…NIPS Deep Learn. Workshop, Montreal, Canada, **2015**, arXiv:1503.02531"。但这篇宣读于 **NIPS 2014 Deep Learning Workshop（2014 年 12 月，Montreal）**，arXiv 才是 2015。出处标成 workshop 却配 2015，自相矛盾。**两种改法任选其一：**
    - **改法 A（推荐，保留 workshop 出处）**：年份 2015→**2014**：
      `G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," in Proc. NIPS Deep Learn. Workshop, Montreal, Canada, 2014, arXiv:1503.02531.`
    - **改法 B（当 arXiv 预印本引）**：去掉 workshop，保留 2015：
      `G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," arXiv preprint arXiv:1503.02531, 2015.`
    - 推荐 A：信息更完整，且与导师把其他条目都写成正式出处的风格一致。属轻微级，不致命，但既然在逐条核，顺手改对最干净。

---

### 📗 [2] DistilBERT — NLP 蒸馏代表

- **docx 原文**：V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv preprint arXiv:1910.01108, 2019.
- **正文出现位置**：Introduction `[1], [2], [3]`；第 27 段（"DistilBERT [2] was introduced to perform distillation in the pre-training phase"）；第 29 段 `[1], [2]`。
- **去哪看**：本地 `thesis/reference/20_DistilBERT_1910.01108.pdf`，第 1 页标题与摘要。
- **为何可引**：你用它说明"蒸馏可在预训练阶段进行、产出更小的通用模型"，与该文主旨完全一致。
- **被引原文关键句（英→中）**：
  - 原文："we propose a method to pre-train a smaller general-purpose language representation model, called DistilBERT, which can then be fine-tuned..."
  - 中文："我们提出一种方法，预训练一个更小的通用语言表示模型 DistilBERT，它随后可被微调用于多种下游任务。"
- **核对要点**：✅ 作者（Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf）、arXiv 号 1910.01108 均正确。无需改。

---

### 📘 [3] Gou — 知识蒸馏综述

- **docx 原文**：J. Gou, B. Yu, S. J. Maybank, and D. Tao, "Knowledge distillation: A survey," Int. J. Comput. Vis., vol. 129, no. 6, pp. 1789-1819, 2021.
- **正文出现位置**：Introduction `[1], [2], [3]`（蒸馏作为一类方法的总括引用）。
- **去哪看**：本地 `thesis/reference/24_Gou_Knowledge_Distillation_A_Survey_IJCV_2021.pdf`（本地是 arXiv 全文版 2006.05525）；卷期页以期刊正式版为准。
- **为何可引**：综述类文献，用来支撑"蒸馏是一类成熟的师生学习/模型压缩方法"的背景陈述，标准用法。
- **被引原文关键句（英→中）**：
  - 原文："Knowledge distillation effectively learns a small student model from a large teacher model."
  - 中文："知识蒸馏能有效地从大型教师模型学习出一个小型学生模型。"
- **核对要点（Crossref 官方核对）**：
  - ✅ 作者 Jianping Gou, Baosheng Yu, Stephen J. Maybank, Dacheng Tao — 正确。
  - ✅ 期刊 International Journal of Computer Vision, **vol. 129, no. 6, pp. 1789-1819, 2021**，DOI 10.1007/s11263-021-01453-z — 卷期页全部正确。无需改。

---

### 📙 [4] Qwen2.5 — 学生模型技术报告

- **docx 原文**：Qwen Team, "Qwen2.5 technical report," arXiv preprint arXiv:2412.15115, 2024.
- **正文出现位置**：Experimental Setup（"Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct [4]"）。
- **去哪看**：本地 `thesis/reference/04_Qwen2.5_Technical_Report_2412.15115.pdf`。
- **为何可引**：你的学生模型就是 Qwen2.5 系列，引用其官方技术报告说明模型来源，必引且正确。
- **被引原文关键句（英→中）**：
  - 原文："we introduce Qwen2.5, a comprehensive series of large language models (LLMs) designed to meet diverse needs."
  - 中文："我们推出 Qwen2.5，一个为满足多样化需求而设计的、覆盖面全面的大语言模型系列。"
- **核对要点**：✅ 团队作者、arXiv 号正确。无需改。

---

### 📕 [5] DeepSeek-V3 — 教师模型技术报告

- **docx 原文**：DeepSeek-AI, "DeepSeek-V3 technical report," arXiv preprint arXiv:2412.19437, 2024.
- **正文出现位置**：Experimental Setup（"The teacher is DeepSeek-V3 [5]"）。
- **去哪看**：本地 `thesis/reference/16_DeepSeek_V3_2412.19437.pdf`。
- **为何可引**：你的教师模型就是 DeepSeek-V3，引用其官方报告，必引且正确。
- **被引原文关键句（英→中）**：
  - 原文："We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token."
  - 中文："我们提出 DeepSeek-V3，一个强大的混合专家（MoE）语言模型，总参数 671B，每个 token 激活 37B 参数。"（与你正文"671B/37B"描述一致）
- **核对要点**：✅ 正确。无需改。

---

### 📗 [6] Singhal — 大模型编码临床知识（Nature）🟡 需修

- **docx 原文（当前，有错）**：K. Singhal, S. Azizi, T. Tu, S. S. Mahdavi, J. Wei, H. W. Chung, N. Scales, A. Tanwani, H. Cole-Lewis, S. Pfohl, P. Payne, M. Seneviratne, **L. Gaeabler, J. Liu, Z. Dai, C. Mclean, D. S. Webster, P. Balaguer, A. S. Chen, G. S. Corrado, Y. Matias, S. Natarajan, Y. Liu, V. Rajpurkar, A. Karton, A. Shetty,** "Large language models encode clinical knowledge," Nature, vol. 620, pp. 172-180, 2023.
- **正文出现位置**：Introduction `[4], [5], [6], [7]`（医疗/通用大模型背景）；第 28 段 `[5], [6], [7], ...`。
- **去哪看**：本地 `thesis/reference/25_Singhal_Large_Language_Models_Encode_Clinical_Knowledge_Nature_2023.pdf`（本地是 arXiv 2212.13138 全文版，作者表在首页）；Nature 正式版 DOI 10.1038/s41586-023-06291-2。
- **为何可引**：用于"医疗大模型在基准上强但部署成本高"的背景，且该文是 MultiMedQA/医学问答评测的奠基工作，合理。
- **被引原文关键句（英→中）**：
  - 原文："Large language models (LLMs) have demonstrated impressive capabilities... we present MultiMedQA, a benchmark combining six existing medical question answering datasets."
  - 中文："大语言模型已展现出令人印象深刻的能力……我们提出 MultiMedQA，一个整合六个已有医学问答数据集的基准。"
- **核对要点 / 必须修正**：
  - 🟡 这篇是 **32 位作者**的大型合作论文。docx 当前把后半段作者缩写写得很乱，经核对至少这些是**错的**：
    - "L. Gaeabler" → 应为 **L. Gabler**（实际通讯/作者列里并无此拼写，疑似 hallucination，建议直接删）
    - "S. Natarajan" → 真实是 **V. Natarajan**（Vivek Natarajan，通讯作者之一）
    - 后段 "A. S. Chen, A. Karton, A. Shetty, P. Balaguer" 等在原文作者表中找不到对应，疑似虚构。
  - ✅ 期刊卷页正确：**Nature, vol. 620, no. 7972, pp. 172-180, 2023**。
  - **建议改法（最稳妥）**：IEEE 允许长作者表用 et al.。改为：
    `K. Singhal et al., "Large language models encode clinical knowledge," Nature, vol. 620, no. 7972, pp. 172-180, 2023.`
    这样既准确又规避了逐个缩写出错的风险。

---

### 📘 [7] Thirunavukarasu — 医学中的大模型（Nature Medicine）

- **docx 原文**：A. J. Thirunavukarasu, D. S. J. Ting, K. Elangovan, L. Gutierrez, T. F. Tan, and D. S. W. Ting, "Large language models in medicine," Nat. Med., vol. 29, pp. 1930-1940, 2023.
- **正文出现位置**：Introduction `[4], [5], [6], [7]`；第 28 段。
- **去哪看**：本地 `thesis/reference/26_Thirunavukarasu_Large_Language_Models_in_Medicine_NatMed_2023.pdf`，首页含 DOI 10.1038/s41591-023-02448-8。
- **为何可引**：综述医学场景下 LLM 的应用、局限与风险，支撑你"医疗任务对可靠性/评测有要求"的背景。
- **被引原文关键句（英→中）**：
  - 原文："Large language models (LLMs) can respond to free-text queries without being specifically trained in the task in question, causing excitement and concern about their use in healthcare settings."
  - 中文："大语言模型无需针对具体任务专门训练即可回答自由文本查询，这既令人振奋，也引发对其在医疗场景中应用的担忧。"
- **核对要点（Crossref 官方核对）**：
  - ✅ 作者 6 位全部正确（Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, Daniel Shu Wei Ting）。
  - ✅ **Nature Medicine, vol. 29, pp. 1930-1940, 2023**（更完整可加 no. 8）。卷页正确。无需改。

---

### 📙 [8] LoRA — 参数高效微调

- **docx 原文**：E. J. Hu, Y. Shen, P. Wallis, J. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," arXiv preprint arXiv:2106.09685, 2021.
- **正文出现位置**：第 27 段（"LoRA ... adapts large models using low-rank incremental parameters [8]"）；Experimental Setup（"LoRA with rank 16 and LoRA alpha 32 [8]"）。
- **去哪看**：本地 `thesis/reference/15_LoRA_2106.09685.pdf`，首页作者表。
- **为何可引**：你的训练用 LoRA，引用原始 LoRA 论文，必引且正确。
- **被引原文关键句（英→中）**：
  - 原文："We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture."
  - 中文："我们提出低秩适配（LoRA），冻结预训练权重，并在 Transformer 每层注入可训练的低秩分解矩阵。"
- **核对要点**：✅ 8 位作者（Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen）、arXiv 号正确。无需改。

---

### 🔴 [9] CMExam — 审稿人抽中、作者整组错误，**必须改**

- **docx 原文（当前，严重错误）**：
  T. Liu, M. Yang, Z. Lu, Q. Chen, S. Zhou, and S. Xiang, "Benchmarking Large Language Models on CMExam: A Comprehensive Chinese Medical Exam Dataset," in Adv. Neural Inf. Process. Syst., 2023.
- **正文出现位置**：
  - 摘要（"datasets such as the Chinese Medical Exam dataset (CMExam)"）。
  - 第 28 段 `[9], [10], [11], [12], [13]`。
  - 第 29 段 `[9], [14]`（"existing studies [9], [14] often use small medical QA test sets"）。
  - Experimental Setup（"a CMExam-based resplit ... [9]"）—— 你的数据集就来自这篇，是硬核引用。
- **去哪看（权威源，三处一致）**：
  - arXiv：https://arxiv.org/abs/2306.03030 （v3，2023-06-05）
  - NeurIPS 2023 proceedings：Advances in Neural Information Processing Systems 36，New Orleans，2023-12-10～16，**pp. 52430-52452**，DOI 10.52202/075280-2283。
- **真实作者（11 位，按 arXiv 顺序）**：
  **Junling Liu, Peilin Zhou, Yining Hua, Dading Chong, Zhongyu Tian, Andrew Liu, Helin Wang, Chenyu You, Zhenhua Guo, Lei Zhu, Michael Lingzhi Li**。
- **为何可引**：你的整个评测数据集（CMExam 重分割、991 题、7 学科）就来自这篇，必引。问题不在"该不该引"，而在"作者写错了"。
- **被引原文关键句（英→中）**：
  - 原文标题："Benchmarking Large Language Models on CMExam — A Comprehensive Chinese Medical Exam Dataset"
  - 中文："在 CMExam 上对大语言模型进行基准测试——一个全面的中文医学考试数据集。"
  - 摘要中文大意："CMExam 源自中国国家医师资格考试，包含 60K+ 多选题，并提供疾病组、临床科室、医学学科、能力领域、难度等五类人工标注，用于细粒度模型评估。"（正好支撑你"7 学科、按难度分层"的描述）
- **错在哪（逐项对照）**：
  | 字段 | docx 当前（错） | 正确 |
  |---|---|---|
  | 第一作者 | T. Liu | **J. Liu**（Junling Liu） |
  | 第 2-6 作者 | M. Yang, Z. Lu, Q. Chen, S. Zhou, S. Xiang | **P. Zhou, Y. Hua, D. Chong, Z. Tian, A. Liu**（全错） |
  | 会议信息 | "in Adv. Neural Inf. Process. Syst., 2023" | 应补全卷次/页码 |
- **建议改法（红色标注整条替换）**：
  CMExam 共 **11 位作者**。按你导师/IEEE 规则（6–8 位全列、**超过 8 位用 et al.**），11 > 8，**应当用 et al.**：
  `J. Liu et al., "Benchmarking large language models on CMExam: A comprehensive Chinese medical exam dataset," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, New Orleans, LA, USA, 2023, pp. 52430-52452.`
  - ⚠️ 唯一要保证的是 **et al. 前的第一作者必须是 `J. Liu`（Junling Liu），不是原稿的 `T. Liu`**。审稿人抓的就是这个错；改成 `J. Liu et al.` 既符合作者人数规则，又改正了错误，最稳。
  - （若导师坚持"无论多少都全列"，再用 11 位全名版本，见上 `真实作者` 一行。但默认按 >8→et al. 规则走。）

---

### 📗 [10] Chau — 生成式 AI 在牙科执照考试中的表现

- **docx 原文**：R. C. W. Chau et al., "Performance of Generative Artificial Intelligence in Dental Licensing Examinations," Int. Dent. J., vol. 74, no. 3, pp. 616-621, Jun. 2024.
- **正文出现位置**：第 28 段 `[9], [10], [11], [12], [13]`（牙科/医疗聊天机器人评测背景）。
- **去哪看（Crossref 官方）**：DOI 10.1016/j.identj.2023.12.007。
- **为何可引**：直接相关的牙科 AI 答题评测工作，且作者团队与你们有关联（Hsung、Lam 等），合理。
- **真实作者**：Reinhard Chun Wang Chau, Khaing Myat Thu, Ollie Yiru Yu, Richard Tai-Chiu Hsung, Edward Chin Man Lo, Walter Yu Hang Lam。
- **被引原文中文大意**："评估生成式 AI（如 ChatGPT）在牙科执照考试题上的作答表现。"
- **核对要点**：✅ 卷期页 **vol. 74, no. 3, pp. 616-621, Jun. 2024** 完全正确（已 Crossref 实查）。用 et al. 也合规。无需改。

---

### 📘 [11] Chau — 对前文的回应（Comment 回复）

- **docx 原文**：R. C. W. Chau, K. M. Thu, O. Y. Yu, E. C. M. Lo, T.-C. Hsung, and W. Y. H. Lam, "Response to Generative AI in Dental Licensing Examinations: Comment," Int. Dent. J., vol. 74, no. 4, pp. 897-898, Aug. 2024.
- **正文出现位置**：第 28 段连引中。
- **去哪看（Crossref）**：DOI 10.1016/j.identj.2024.02.002。
- **核对要点**：✅ **vol. 74, no. 4, pp. 897-898, Aug. 2024** 正确。作者顺序与官方一致。无需改。
- **提示**：注意别和另一篇 **Daungsupawong & Wiwanitkit 的 "...: Comment"（vol.74, no.2, p.361）混淆——那是别人写的评论，不是你引的这篇 Chau 团队的"回复"。** 你引的是 "Response to ... Comment"，正确。

---

### 📙 [12] Chau — 修复/义齿牙科多选题聊天机器人评估

- **docx 原文**：R. C. W. Chau, K. M. Thu, O. Y. Yu, E. C. M. Lo, T.-C. Hsung, and W. Y. H. Lam, "Evaluation of Chatbot Responses to Text-Based Multiple-Choice Questions in Prosthodontic and Restorative Dentistry," Dent. J., vol. 13, no. 7, p. 279, Jul. 2025.
- **正文出现位置**：第 28 段连引中。
- **去哪看（Crossref）**：DOI 10.3390/dj13070279。
- **核对要点**：
  - ✅ 期刊 **Dentistry Journal**（缩写 Dent. J.）、**vol. 13, no. 7, art. 279, 2025** 正确。
  - 🟡 官方作者共 **8 位**：Chau, Thu, Yu, **Hsung, Denny Chon Pei Wang, Manuel Wing Ho Man, John Junwen Wang**, Lam（docx 错把 [11] 的作者表套上来，漏了中间三位 Wang/Man/Wang，且误加了 Lo——Lo 不在此篇）。
  - **建议改法**：8 位在"6–8 全列"区间内，**应全列**（不要用 et al.）：
    `R. C. W. Chau, K. M. Thu, O. Y. Yu, R. T.-C. Hsung, D. C. P. Wang, M. W. H. Man, J. J. Wang, and W. Y. H. Lam, "Evaluation of chatbot responses to text-based multiple-choice questions in prosthodontic and restorative dentistry," Dent. J., vol. 13, no. 7, art. no. 279, Jul. 2025.`

---

### 📕 [13] Lam — 数字牙科临床实践综述

- **docx 原文**：W. Y. H. Lam, R. C. W. Chau, K. M. Thu, O. Y. Yu, E. C. M. Lo, and T.-C. Hsung, "Digital Dentistry in Clinical Practice: A Scoping Review of Current Capabilities and Future Directions," Int. Dent. J., vol. 76, no. 1, Feb. 2026, Art. no. 109296.
- **正文出现位置**：第 28 段连引中。
- **去哪看（Crossref）**：DOI 10.1016/j.identj.2025.109296。
- **核对要点**：
  - ✅ 期刊、**vol. 76, no. 1, 2026, Art. no. 109296** 正确。
  - 🟡 官方作者共 **9 位**：Walter Yu Hang Lam, **Zhaoting Ling, Kaijing Mao, Ji-Man Park, Amirali Zandinejad, Adriana da Fonte Porto Carreiro, Francesco Guido Mangano, Jeffrey A. Platt, Falk Schwendicke**（与 docx 写的 Chau/Thu/Yu/Lo/Hsung **完全不同**——docx 又错套了 Chau 团队作者表）。这是除 [9] 外作者表偏差最大的一条。
  - **建议改法**：9 位 > 8，按规则**用 et al.**（第一作者保留 Lam）：
    `W. Y. H. Lam et al., "Digital dentistry in clinical practice: A scoping review of current capabilities and future directions," Int. Dent. J., vol. 76, no. 1, art. no. 109296, Feb. 2026.`

---

### 📗 [14] HuatuoGPT — 医疗对话模型

- **docx 原文**：H. Zhang, J. Chen, F. Jiang, F. Yu, N. Chen, J. Li, G. Chen, and S. Cui, "HuatuoGPT, towards taming language models to be a doctor," in Findings Assoc. Comput. Linguistics: EMNLP 2023, 2023.
- **正文出现位置**：第 29 段 `[9], [14]`（free-form targets / 小测试集）；Introduction free-form 蒸馏目标 `[14]`。
- **去哪看**：本地 `thesis/reference/08_HuatuoGPT_2023.findings-emnlp.725.pdf`；Crossref DOI 10.18653/v1/2023.findings-emnlp.725，**pp. 10859-10885**。
- **为何可引**：作为"自由文本/生成式医疗模型"代表，与你的"选项级 vs 自由文本目标"对比，合理。
- **被引原文关键句（英→中）**：
  - 原文："we present HuatuoGPT, a large language model (LLM) for medical consultation."
  - 中文："我们提出 HuatuoGPT，一个用于医疗咨询的大语言模型。"
- **核对要点**：
  - ✅ 题名、会议（Findings of ACL: EMNLP 2023）正确。
  - 🟡 docx 把作者截到第 8 位（"...and S. Cui"）当成完整列表，但 HuatuoGPT 官方作者**多于 8 位**（Hongbo Zhang, Junying Chen, Feng Jiang, Fei Yu, **Zhihong Chen**, Guiming Chen, Jianquan Li, Xiangbo Wu, ... 等），且 docx 中段第 5 位写成 "N. Chen" 是**错的**——应为 **Z. Chen（Zhihong Chen）**。
  - **建议改法**：作者 > 8 位，按规则**用 et al.**（第一作者 H. Zhang 正确，保留），同时补页码：
    `H. Zhang et al., "HuatuoGPT, towards taming language models to be a doctor," in Findings Assoc. Comput. Linguistics: EMNLP 2023, 2023, pp. 10859-10885.`
    这样既符合 >8→et al. 规则，又规避了 "N. Chen" 这类中段缩写错误。

---

## 第 3 部分：alpha 值修改（你说的第 2 件事）

> 你说"刚好有修改机会，修改论文里的 alpha 值和对应描述"。这部分我已查清数据，但**最终用哪个值要你拍板**，所以单独列出，等你确认后我再动 docx。

### 3.1 现状
- 论文当前：主设置 **α = 0.35**（Stage 1 损失 = α·KL + (1−α)·CE，即 0.35·KL + 0.65·CE）。
- 正文第 44 段、51 段都写了 α=0.35 与 CE 权重 0.65。

### 3.2 消融实验的真实结论（来自 alpha_ablation_results.md，canonical 标准化评测）
- α=0（纯 CE / 纯 SFT）：full 991 题 = **89.14%**（三种子 89.40/88.50/89.51）
- α=0.35（当前主结果）：= 88.67%
- 随 KL 权重升高，性能**单调下降**到 α=1.0 时 86.21%。
- 即：**KL 权重越大越差，α=0 最好。**

### 3.3 这带来一个"诚实性"问题（必须你决定怎么处理）
论文现在的叙事暗示"靠蒸馏 KL 让学生超过老师"。但消融显示：**真正起作用的是决策空间的 CE 监督（Choice-Head 结构），KL 反而轻微有害。**

三个选项（我建议 B）：
- **A. 不动 alpha，只修文献**：风险最低，但放弃了"借返修机会修正归因"的机会，且 α=0.35 并非最优值这件事仍埋着。
- **B.（推荐）保留 α=0.35 作为主报告值，但在描述里诚实补一句**：消融表明性能对 KL 权重敏感、α→0 时最优，超越教师主要来自决策空间监督而非 dark-knowledge 蒸馏。改动小、红字标出、不推翻已定稿的 89.10% 主结果。
- **C. 把主设置直接改成 α=0**：最忠于数据，但 89.10%/88.67% 这些已被评审看过的主数字会变（变成 89.14% 体系），改动大，风险高。

👉 **请你回复选 A / B / C（或你的想法）。** 选定后我再在 docx 里红字修改对应段落，并写中文修改说明。

---

## 第 4 部分：人工核对进度勾选表（你边核边打勾）

参考文献（14 篇）：
- [ ] [1] Hinton  　- [ ] [2] DistilBERT  　- [ ] [3] Gou  　- [ ] [4] Qwen2.5
- [ ] [5] DeepSeek-V3  　- [ ] [6] Singhal（**改 et al.**）  　- [ ] [7] Thirunavukarasu
- [ ] [8] LoRA  　- [ ] **[9] CMExam（🔴 整条替换）**  　- [ ] [10] Chau-Licensing
- [ ] [11] Chau-Response  　- [ ] [12] Chau-Prosthodontic（建议 et al.）
- [ ] [13] Lam-ScopingReview（建议 et al./补正确作者）  　- [ ] [14] HuatuoGPT

正文引用位置（每处确认 [n] 指向合理）：
- [ ] 摘要 CMExam 提及
- [ ] Introduction 第 26 段 `[1],[2],[3]` / `[4],[5],[6],[7]`
- [ ] 第 27 段 DistilBERT [2] / LoRA [8]
- [ ] 第 28 段 `[5],[6],[7],[9],[10],[11],[12],[13]`
- [ ] 第 29 段 `[1],[2]` / `[14]` / `[9],[14]`
- [ ] Experimental Setup CMExam [9] / DeepSeek [5] / Qwen [4] / LoRA [8]

全部打勾后，方可定稿重投。

---

## 第 5 部分：导师返修版（1015）复核结果 ⭐ 以这版为准

> 2026-06-22，导师直接改了一版：`aiea/aiea_DentalMCQ_Distillation_2026-06-22_EN - 1015.docx`（及同名 PDF）。
> 我已逐条对照前面查好的权威源复核。**结论：导师改得很到位，上面第 0 部分标的问题基本全修好了，拒稿级错误已消除。**
> 导师用的两招：① 容易出错的长作者表统一改成 et al.；② 把 [9] 第一作者改对。
> **你现在人工核对，就以这一版（1015）为对象**，前面 0030 版的"建议改法"已被导师采纳，仅作背景参考。

### 5.1 逐条复核（导师 1015 版实际写法 vs 权威源）

| 编号 | 导师 1015 版现在写的 | 复核结论 |
|---|---|---|
| [1] Hinton | G. Hinton, O. Vinyals, and J. Dean, … NIPS Deep Learn. Workshop, Montreal, **2015**, arXiv:1503.02531 | ⚠️ 作者✅；年份矛盾：workshop 实为 **2014**，2015 是 arXiv 年（见 [1] 卡详解，建议 2015→2014） |
| [2] DistilBERT | V. Sanh, L. Debut, J. Chaumond, and T. Wolf, … arXiv:1910.01108 | ✅ 正确 |
| [3] Gou | J. Gou, B. Yu, S. J. Maybank, and D. Tao, … IJCV vol.129 no.6 pp.1789-1819, 2021 | ✅ 正确 |
| [4] Qwen2.5 | Qwen Team, … arXiv:2412.15115 | ✅ 正确 |
| [5] DeepSeek-V3 | DeepSeek-AI, … arXiv:2412.19437 | ✅ 正确 |
| [6] Singhal | **K. Singhal et al.**, Nature vol.620 pp.172-180, 2023 | ✅ 已修好（et al. 规避了原拼写错） |
| [7] Thirunavukarasu | **A. J. Thirunavukarasu et al.**, Nat. Med. vol.29 pp.1930-1940, 2023 | ✅ 正确 |
| [8] LoRA | E. J. Hu, Y. Shen, … and W. Chen, arXiv:2106.09685 | ✅ 正确（8 位全列，合规） |
| **[9] CMExam** | **J. Liu et al.**, … in Adv. Neural Inf. Process. Syst. (NeurIPS), **pp. 52430-52452, 2023** | ✅ **致命错误已修复**（T.→J. Liu，补了页码） |
| [10] Chau-Licensing | R. C. W. Chau et al., Int. Dent. J. vol.74 no.3 pp.616-621, Jun.2024 | ✅ 标题/卷期页/DOI 全对（Crossref 核实）；🔶 作者实为 **6 位**（在 6–8 区间），按规则应**全列**而非 et al.，见 5.2 第④点 |
| [11] Chau-Response | R. C. W. Chau et al., Int. Dent. J. vol.74 no.4 pp.897-898, Aug.2024 | ✅ 正确 |
| [12] Chau-Prosthodontic | R. C. W. Chau et al., Dent. J. vol.13 no.7 **pp. 279**, 2025 | ✅ 作者已修好；🔶 见 5.2 第②点（pp.→Art. no.） |
| [13] Lam | **W. Y. H. Lam, et al.**, Int. Dent. J. vol.76 no.1, Feb.2026, Art. no.109296 | ✅ 作者已修好；🔶 见 5.2 第③点（多了个逗号） |
| [14] HuatuoGPT | **H. Zhang et al.**, … EMNLP 2023, **pp. 10859-10885** | ✅ 已修好（et al.+页码，规避了 N. Chen 错） |

### 5.2 仅剩几个吹毛求疵级小格式（不影响投稿，看你/导师要不要顺手清）

1. **[9] 缺卷次**：现写 "…(NeurIPS), pp. 52430-52452, 2023"，有页码已够用；IEEE 更完整可补成 "…(NeurIPS), **vol. 36**, pp. 52430-52452, 2023"。**非必须。**
2. **[12] 页码写法**：现写 "pp. 279"。该文是文章编号 279（不是页范围），严格应写 "**Art. no. 279**" 或 "**p. 279**"（单数）。**小瑕疵。**
3. **[13] 多一个逗号**：现写 "W. Y. H. Lam, et al."，IEEE 规范是 "**W. Y. H. Lam et al.**"（Lam 和 et al. 之间不加逗号）。**小瑕疵。**
4. **[10] 作者用了 et al. 但只有 6 位**：Crossref 核实 [10] 实为 6 位作者（Chau, Thu, Yu, Hsung, Lo, Lam），在 6–8 区间内，按你导师规则应**全列**：
   `R. C. W. Chau, K. M. Thu, O. Y. Yu, R. T.-C. Hsung, E. C. M. Lo, and W. Y. H. Lam, …`
   导师版写成 "R. C. W. Chau et al." 是偏保守的写法，不算错，但若严格执行 6–8 全列规则可展开。**小瑕疵，按导师口味定。** （[11] 同为 Chau 团队、人数相近，若改 [10] 建议一并核对 [11]。）
5. **[1] Hinton 年份**：workshop 实为 2014，现写 2015（=arXiv 年），建议 2015→2014。详见 [1] 卡。**轻微。**

> 这些都不致命。你可以：(a) 自己在 docx 里顺手改；(b) 或直接把这几条告诉导师由他定。**即使都不改，这版参考文献也已达到可投状态。**

---

## 第 6 部分：你人工核对的操作流程（照着做一遍）

> 目标：你不依赖我，自己独立把导师 1015 版的引用复核一遍，确认无误后定稿。预计 30–45 分钟。

**准备**：打开两样东西——
- A. 导师版 PDF：`aiea/aiea_DentalMCQ_Distillation_2026-06-22_EN - 1015.pdf`（翻到最后 REFERENCES 那页）
- B. 本文档（你正在看的这份）

**对每一条 [1]–[14]，走 4 步：**

1. **看作者**：PDF 里这条的作者，和本文档第 5.1 表"导师 1015 版现在写的"对得上吗？
   - 若是 et al. → 只需确认 **et al. 前第一作者缩写**对（特别盯 [9] 是不是 J. Liu）。
   - 若是全列 → 数一下是不是 ≤8 位、且姓名顺序对。

2. **看标题/会议名**：和 PDF 能对上即可（这些字段看论文首页就能确认）。

3. **看卷/期/页/年**（DOI 类）：
   - 打开 https://search.crossref.org → 粘论文标题搜 → 比对卷 vol./期 no./页 pp./年。
   - 或用本文档每条给的 DOI，访问 `https://doi.org/<DOI>` 跳转核对。
   - arXiv 类（[1][2][4][5][8]）：访问 `https://arxiv.org/abs/<编号>` 核对即可，无卷期页。

4. **看引用合理性**：回到正文那个 [n] 出现的句子（位置见第 2 部分各卡"正文出现位置"），确认你那句话的主张确实能被这篇论文支撑（"为何可引"已替你写好理由）。

**全部 14 条走完后**，回到下面这张总表打勾。14 条 + 正文位置全打勾 = 核对完成，可定稿重投。

### 核对进度勾选表（导师 1015 版）

参考文献（14 篇，导师版）：
- [ ] [1] Hinton  - [ ] [2] DistilBERT  - [ ] [3] Gou  - [ ] [4] Qwen2.5  - [ ] [5] DeepSeek-V3
- [ ] [6] Singhal  - [ ] [7] Thirunavukarasu  - [ ] [8] LoRA
- [ ] **[9] CMExam（重点确认第一作者 = J. Liu）**  - [ ] [10] Chau-Licensing  - [ ] [11] Chau-Response
- [ ] [12] Chau-Prosthodontic（顺带看 pp.279 要不要改 Art. no.）
- [ ] [13] Lam（顺带看 "Lam, et al." 多余逗号）  - [ ] [14] HuatuoGPT

正文引用位置：
- [ ] 摘要 CMExam 提及
- [ ] Introduction `[1],[2],[3]` / `[4],[5],[6],[7]`
- [ ] 第 27 段 DistilBERT [2] / LoRA [8]
- [ ] 第 28 段 `[5],[6],[7],[9],[10],[11],[12],[13]`
- [ ] 第 29 段 `[1],[2]` / `[14]` / `[9],[14]`
- [ ] Experimental Setup CMExam [9] / DeepSeek [5] / Qwen [4] / LoRA [8]

三个小格式（可选）：
- [ ] [9] 是否补 vol. 36
- [ ] [12] pp. 279 → Art. no. 279
- [ ] [13] 删 "Lam," 后的逗号
