# 论文引用位置与参考文献对应说明

说明：
- 当前以 thesis_submission.md 为主稿。
- 当前以 thesis/reference 为唯一参考文献目录。
- thesis_submission.md 正文大多是叙述式引用，因此本文件采用“章节位置 + 论断用途 + 对应原文部分”的方式映射，而不再依赖 thesis.md 的旧行号。

## 核心映射表

| 编号 | thesis_submission.md 中的主要使用位置 | 引用目的 | 对应原文部分 | 当前状态 |
|---|---|---|---|---|
| [1] | 2.3 医疗大语言模型背景 | 说明 Transformer 架构是后续大模型的共同基础 | Abstract；Introduction | PDF |
| [2] | 2.1 通用大模型背景 | 作为 GPT 系列代表，支撑“通用大模型进入医疗场景” | Abstract；Introduction；Scope and Limitations | PDF |
| [3] | 2.1、2.3、4.9 | 说明 Llama 系列的基础能力与跨架构蒸馏背景 | Introduction；Benchmark；post-training/data mix 相关章节 | PDF |
| [4] | 2.2、2.3 | 支撑 Qwen2.5 的规模、训练改进与词表/架构描述 | Abstract；Introduction；Architecture & Tokenizer；training methodology 相关章节 | PDF |
| [5] | 2.2、3.3 | 支撑软标签蒸馏、温度缩放与暗知识的基本思想 | Abstract；1 Introduction；2 Distillation；3 Preliminary Experiments | PDF |
| [6] | 2.1 | 说明 ChatGLM/GLM-4 在中文医疗与通用中文场景中的代表性 | Abstract；Introduction | PDF |
| [7] | 参考文献表保留，正文直接依赖已移除 | 原本拟用作中文医疗对话模型代表条目；现已确认 arXiv 正式来源存在，旧题名误写问题已修复，并已补本地 PDF 与快照，因此当前仅作补充背景，不承载正文核心论证 | arXiv Abstract；数据构建说明；实验结论 | arXiv + local PDF/snapshot |
| [8] | 2.1 | 支撑 HuatuoGPT 的训练配方与医疗咨询能力 | Abstract；Introduction；Methodology；2 RLMF | PDF |
| [9] | 2.1、3.2 | 支撑 Huatuo-26M 作为大规模中文医疗 QA 数据集 | Abstract；Introduction；Dataset；Data Sources/Data Processing；Benchmarking | PDF |
| [10] | 2.1、3.2 | 支撑 CMExam 为中国医学考试标准化评测基准 | 页面标题与出版信息 | Landing page |
| [11] | 2.1、3.2 | 支撑 MedQA 为医学考试来源的英文基准 | Abstract；1 Introduction；3 Data；5 Experiments；6 Conclusions；Share and Cite 元数据 | PDF + article page |
| [12] | 2.1、3.2 | 支撑 PubMedQA 为生物医学 QA 基准 | Abstract；Introduction；Data Collection；Evaluation Settings | PDF |
| [13] | 2.2 | 支撑中间层 hint 蒸馏思路 | student-teacher framework 与 hints 训练流程；MNIST/CIFAR/SVHN/AFLW 基准实验 | PDF |
| [14] | 2.2 | 支撑注意力迁移蒸馏思路 | 3 Activation-based Attention Transfer；3 Gradient-based Attention Transfer；4 Experimental Results | PDF |
| [15] | 2.2、3.7、5.2、5.3 | 支撑 LoRA 低秩适配与参数高效训练载体 | LoRA design and practical benefits；rank-deficiency 讨论；GLUE/语言模型适配实验 | PDF |
| [16] | 2.3、4.4、4.12、5.4 | 支撑 DeepSeek-V3 的 MoE 规模与教师背景 | Abstract；Introduction；2 Architecture；性能图表；Conclusion | PDF |
| [17] | 2.4、3.6、5.1 | 支撑信息几何框架、散度/对偶结构、统计推断与机器学习应用背景 | Springer 图书简介、Part I/III/IV 概述、关键词 | Landing page + DOI metadata |
| [18] | 2.4 | 支撑 Fisher 信息与统计参数估计精度的经典历史来源 | 章节题名、DOI、作者、页码与摘要片段 | Chapter metadata |
| [19] | 2.2 | 用于补充 NLP 蒸馏工具链背景与工程化蒸馏流程 | Abstract；Introduction；工具工作流步骤；Experiments | PDF |
| [20] | 2.2 | 用于补充 NLP 蒸馏代表案例，说明预训练阶段也可蒸馏 | Abstract；Introduction；Experiments | PDF |
| [21] | 缩写表、3.1 硬件部署、4.9 | 支撑 AWQ 量化与 Llama-70B-AWQ 的部署背景 | activation-aware scaling/search 方法；TinyChat 推理系统；Experiments | PDF |
| [22] | 3.7 | 支撑量化微调与低显存训练背景 | Abstract；Introduction；4-bit/quantization 方法相关章节 | PDF |
| [23] | 2.3 | 支撑 RoPE/RoFormer 的位置编码背景 | Introduction；Preliminary；RoPE 理论与实验讨论部分 | PDF |

## 当前仍需补强的地方

1. [7] 已能细化到 arXiv 摘要与实验层级，并已补本地 PDF 与快照证据；当前剩余较弱来源主要是 [10]、[17]、[18] 这类 landing page 条目。
2. [19]、[20]、[22]、[23] 已分别落到 2.2、2.2、3.7、2.3，但若后续要提交更严格的引用核验表，仍可补成显式编号型引文。
3. [17] / [18] 当前虽然不是全文 PDF 核验，但已经具备“替代证明链”：前者证明现代信息几何框架与应用范围，后者证明 Fisher 信息/估计精度的经典来源地位。
4. 如果后续需要提交更严格的引用核验表，建议把 thesis_submission.md 中对应章节再补上更显式的编号型引文。