# IEEE 论文真实范例量化（基于已评审通过的 AIEA 论文）

> 来源：math/aiea_DentalMCQ_Distillation_2026-06-19_EN.docx（经论文委员会评审 + 老师审核的标准 IEEE 格式论文）
> 用途：把抽象 IEEE 规则量化成可对照的具体格式，补充进学习笔记，再提炼 skill。

---

## 1. 作者信息块（IEEE 多机构格式）

实际写法（多机构分块，每块独立）：
```
Title（论文标题，居中，每个主要词首字母大写）

第一作者, 第二作者*, 第三…, …, 末作者   （* 标通讯作者）
Dept. Computer Science
Hong Kong Chu Hai College
Hong Kong SAR, China
0009-0004-0137-8360, {richardhsung, harristsang, wllo, ylzhu, xxyang}@chuhai.edu.hk

Billy Chiu                          ← 不同机构另起一块
School of Data Science
Lingnan University
Hong Kong SAR, China
billychiu@ln.edu.hk

Walter Lam                          ← 第三机构再一块
Faculty of Dentistry
The University of Hong Kong
Hong Kong SAR, China
retlaw@hku.hk
```
量化要点：
- 每个机构块四行：作者名 / 系(Dept./School/Faculty) / 机构 / 地区+国家。
- 同机构多作者共用一块，邮箱可用 `{user1, user2}@domain` 合并写法。
- ORCID（0009-…）可与邮箱同行。
- 通讯作者用 `*` 标注。

## 2. 摘要 + 关键词（量化）
- Abstract 用 `Abstract—`（破折号）起头，**单段**，本文约 250 词。
- 流程严格对应 5 部件：背景(医疗 LLM 部署贵)→问题(全词表 vs 五选项错配)→方法(Choice-Head)→实验设置(DeepSeek 教师/Qwen 学生/991 题)→结果(89.10% > 87.18% 教师，3-seed 88.67%)→结论。
- **数字必须具体**：89.10%、87.18%、88.67% 直接进摘要——评审通过的论文摘要里结果是量化的，不是 "significant improvement"。
- Keywords 用 `Keywords—`，分号分隔，5 个：`knowledge distillation; medical question answering; large language models; multiple-choice reasoning; decision-space supervision`。

## 3. 章节结构（IEEE 会议论文实际节序）
Introduction → Method（含 Problem Setting / Choice-Head Distillation / Stage 1 / Stage 2 子节）→ Experimental Setup → Results and Discussion（含带小标题的分析段）→ Practical Implications and Limitations → Conclusion → Acknowledgment → References。
- 贡献点在 Introduction 末尾用 "The main contributions of this work are as follows:" 列出（本文 2 条）。
- Results 里用**加粗 run-in 小标题**组织讨论：`The Role of Stage 2 SFT Calibration:`、`Student-Over-Teacher Dynamics:`、`Scope of Applicability:`。

## 4. 参考文献格式（按源类型量化，13 条真实样本）

**期刊文章**（卷/期/页/年，期刊名缩写斜体）：
```
[8] J. Gou, B. Yu, S. J. Maybank, and D. Tao, "Knowledge distillation: A survey,"
    Int. J. Comput. Vis., vol. 129, no. 6, pp. 1789-1819, 2021.
[9] K. Singhal, ... and A. Shetty, "Large language models encode clinical knowledge,"
    Nature, vol. 620, pp. 172-180, 2023.
[11] R. C. W. Chau et al., "Performance of Generative AI in Dental Licensing Examinations,"
    Int. Dent. J., vol. 74, no. 3, pp. 616-621, Jun. 2024.
```
**会议论文**（in Proc. … 缩写会议名）：
```
[1] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network,"
    in Proc. NIPS Deep Learn. Representation Learn. Workshop, Montreal, Canada, 2015, arXiv:1503.02531.
[6] T. Liu, ... and S. Xiang, "Benchmarking Large Language Models on CMExam: ...,"
    in Adv. Neural Inf. Process. Syst., 2023.
[7] H. Zhang, ..., "HuatuoGPT, towards taming language models to be a doctor,"
    in Findings Assoc. Comput. Linguistics: EMNLP 2023, 2023.
```
**arXiv 预印本**（两种都出现）：
```
[2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, ...,"
    arXiv preprint arXiv:1910.01108, 2019.
[5] DeepSeek-AI, "DeepSeek-V3 technical report," arXiv preprint arXiv:2412.19437, 2024.
```
量化要点（从样本提炼）：
- 作者：`First-Initial(s). Surname`，逗号分隔，最后一个前用 and；**6 作者以上用 `et al.`**（见 [11]）。
- 文章标题：双引号、句式大小写（仅首词与专有名词大写，多数样本如此）。
- 期刊/会议名：**标准缩写 + 斜体**（Int. J. Comput. Vis. / Nat. Med. / Int. Dent. J. / Adv. Neural Inf. Process. Syst.）。
- 卷期页：`vol. x, no. x, pp. xx-xx`；月份缩写 `Jun. 2024`。
- 文章编号（无页码的新式期刊）：`Art. no. 109296`（见 [14]）。
- 机构作者直接写名（`Qwen Team`、`DeepSeek-AI`）。
- 参考列表按**正文出现顺序**编号。

## 5. 图（Figure）题注格式
- 位置：图**下方**。标签 `Fig. N.`（缩写 + 句点 + 空格），后接说明句。
- 实例：
  - `Fig. 1. Main results on the 991-question full test set. The distilled 14B student improves over the zero-shot baseline and exceeds the teacher; the best run reaches 89.10%, while the three-seed mean remains 88.67%.`
  - `Fig. 2. Accuracy gains over zero-shot baselines on full and dental tests.`
  - `Fig. 3. Teacher-gold agreement and Stage 1 loss weights. CE dominates disagreement cases while KL preserves option preferences.`
- 量化：图题注是**完整句**，常含关键数字，不只是标签词组。正文引用写 `Fig. 2 visualizes these gains.`

## 6. 表（Table）题注格式
- 位置：表**上方**。标签全大写罗马数字 `TABLE I.`，后接标题。
- 实例：`TABLE I. Results on the CMExam full set and subset. Full = 991-question full test set; Dental Test = 125-question dental subset.`
- 量化：表号用罗马数字（I, II…）非阿拉伯；题注里可加缩写定义（Full = …）。
- 表体列：Model / Setting / Full Test Accuracy / Dental Test Accuracy，准确率带 %。

## 7. 公式
- 编号公式在正文用 `Stage 1 KL loss`、`softmax over the five candidate options` 描述，公式本身右侧编号 (1)(2)(3)。
- 正文引用写 `Equations (1), (2), and (3)`，不写 "equation (2) of …"。

## 8. 数字/百分比写法（从范例量化）
- 准确率两位小数 + %：`89.10%`、`87.18%`、`88.67%`。
- 百分点差用 "percentage points"：`a gain of 9.11 percentage points`。
- 区间用 en-dash：`88.40%–89.10%`；样本数带逗号：`6,590`、`4,608`。
- 学习率：`1 × 10⁻⁴`；超参直接给值：`rank 16 and LoRA alpha 32`、`α is set to 0.35`。

## 9. 诚实/对冲语（范例里如何写局限——印证我们的诚实护栏）
范例 Limitations 段的真实写法值得照搬语气：
- "this work does not include a direct full-vocabulary distillation control, so it cannot strictly prove that option-level distillation necessarily outperforms…"
- "evaluates a single strong teacher and a narrow family of student architectures. That is enough to establish the central result, but not enough to show that every teacher-student pair will behave the same way."
- "these conclusions should not be broadly generalized to open-ended clinical dialogue…"
→ 即：明确声明边界、不过度外推、点明缺对照——这正是受保护的对冲语，评审通过的论文就是这么写的。

## 10. 致谢
- `Acknowledgment`（IEEE 用单数拼写）段：`The work described in this paper was supported by Hong Kong Chu Hai College.`
