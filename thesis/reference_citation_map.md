# 论文引用位置与参考文献对应说明

说明：
- 文内引用位置以 [thesis/thesis.md](thesis/thesis.md) 为准。
- 当前参考文献文件统一以 [thesis/reference2](thesis/reference2) 为准。
- [thesis/thesis_v2.md](thesis/thesis_v2.md) 的参考文献表已按本次核验结果做过一轮修正。

## 总体状态

| 类型 | 数量 | 说明 |
|---|---:|---|
| 已有正确 PDF | 19 | 可直接打开核验正文 |
| 官方 landing page | 4 | 出版方页面已保存 |
| 检索证据 | 2 | 原条目仍待人工核准 |

## 编号对应关系

| 编号 | thesis.md 中的主要引用位置 | 引用目的 | reference2 中的对应文件 | 当前状态 |
|---|---|---|---|---|
| [1] | thesis.md:130, 324 | Transformer 与 self-attention 基础 | reference2/01_Attention_Is_All_You_Need_1706.03762.pdf | PDF |
| [2] | thesis.md:130, 219 | GPT 系列作为通用 LLM 代表 | reference2/02_GPT4_Technical_Report_2303.08774.pdf | PDF |
| [3] | thesis.md:130, 219, 245 | LLaMA 系列与跨架构背景 | reference2/03_Llama3_Herd_of_Models_2407.21783.pdf | PDF |
| [4] | thesis.md:130, 241 | Qwen 系列模型背景 | reference2/04_Qwen2.5_Technical_Report_2412.15115.pdf | PDF |
| [5] | thesis.md:144, 227, 229, 281 | 软标签蒸馏、温度缩放、暗知识 | reference2/05_Distilling_Knowledge_1503.02531.pdf | PDF |
| [6] | thesis.md:219 | ChatGLM 家族背景 | reference2/06_ChatGLM_2406.12793.pdf | PDF |
| [7] | thesis.md:219 | 中文医疗对话模型代表工作 | reference2/07_DISC-MedLLM_search_results.json | 检索证据 |
| [8] | thesis.md:219 | HuatuoGPT 的医疗推理与训练方法 | reference2/08_HuatuoGPT_2023.findings-emnlp.725.pdf | PDF |
| [9] | thesis.md:221, 271 | Huatuo 系列医学 QA 数据集背景 | reference2/09_Huatuo-26M_2025.findings-naacl.211.pdf | PDF |
| [10] | thesis.md:223 | CMExam 评测基准 | reference2/10_CMExam_Benchmark_landing.html | Landing page |
| [11] | thesis.md:223, 271 | MedQA 基准与数据集来源 | reference2/11_MedQA_10.3390_app11146421_landing.html | Landing page |
| [12] | thesis.md:223, 271 | PubMedQA 基准 | reference2/12_PubMedQA_1909.06146.pdf | PDF |
| [13] | thesis.md:229 | FitNets 中间层提示蒸馏 | reference2/13_FitNets_1412.6550.pdf | PDF |
| [14] | thesis.md:229 | Attention transfer 蒸馏 | reference2/14_Attention_Transfer_1612.03928.pdf | PDF |
| [15] | thesis.md:233 | 黑盒/输出层蒸馏相关讨论 | reference2/15_Scaling_Laws_for_KD_search_results.json | 检索证据 |
| [16] | thesis.md:233 | Symbolic KD 思路 | reference2/16_Symbolic_KD_2110.07178.pdf | PDF |
| [17] | thesis.md:235, 285, 356 | LoRA 低秩适配 | reference2/17_LoRA_2106.09685.pdf | PDF |
| [18] | thesis.md:243 | DeepSeek-V3 教师模型背景 | reference2/18_DeepSeek_V3_2412.19437.pdf | PDF |
| [19] | thesis.md:253, 259, 366 | 信息几何与 α-散度理论 | reference2/19_Information_Geometry_and_Its_Applications_landing.html | Landing page |
| [20] | thesis.md:255 | Fisher-Rao 距离的经典来源 | reference2/20_Rao_Information_and_Accuracy_reprint_landing.html | Landing page |
| [21] | thesis.md:281 | TextBrewer 工具包 | reference2/21_TextBrewer_2002.12620.pdf | PDF |
| [22] | thesis.md:281 | DistilBERT 作为 NLP 蒸馏代表例子 | reference2/22_DistilBERT_1910.01108.pdf | PDF |
| [23] | thesis.md:283 | AWQ 量化 | reference2/23_AWQ_2306.00978.pdf | PDF |
| [24] | thesis.md:285 | QLoRA 量化微调 | reference2/24_QLoRA_2305.14314.pdf | PDF |
| [25] | thesis.md:332 | RoPE / RoFormer | reference2/25_RoFormer_2104.09864.pdf | PDF |

## 当前仍需人工确认的条目

1. [7] DISC-MedLLM：原 bibliography 中曾带错误 arXiv 编号，当前仅保存检索证据。
2. [15] Scaling Laws for Knowledge Distillation：当前题名与可检索结果不稳定，尚未确认正式来源。

## 已修正的 bibliography 重点条目

以下条目已在 [thesis/thesis_v2.md](thesis/thesis_v2.md) 中修正为更接近正式来源的写法：

1. [8] HuatuoGPT
2. [9] Huatuo-26M
3. [10] CMExam
4. [11] MedQA
5. [16] Symbolic Knowledge Distillation
6. [21] TextBrewer
7. [25] RoFormer 年份