# 医学选择题数据集（fullEnglish）

面向医学 MCQ（Multiple Choice Question）研究 / 蒸馏实验的英文数据集集合。所有数据已转换成**统一格式**，存放在 `data/` 目录。

## 一、数据总览

| 数据集 | 题数 | 切分 | 选项数 | 说明 |
|---|---|---|---|---|
| **MedQA** | 12,723 | train / dev / test | 5 | 美国 USMLE 医师执照考试题 |
| **MedMCQA** | 187,005 | train / validation | 4 | 印度医学考试题（AIIMS/NEET） |
| **MMLU（医学相关 12 科目）** | 3,207 | test / dev / validation | 4 | 大规模多任务语言理解的医学科目 |
| **PubMedQA** | 1,000 | labeled | 3 | 生物医学 yes/no/maybe 问答（独立评测集，不用于蒸馏训练） |

**总计：203,935 条**

---

## 二、统一格式

所有 `data/*.jsonl` 文件的每一行是一个 JSON 对象，字段如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 唯一标识，格式 `{source}_{subject}_{split}_{序号}` |
| `source` | string | 数据集来源：`medqa` / `medmcqa` / `mmlu` / `pubmedqa` |
| `split` | string | 切分：`train` / `dev` / `test` / `validation` / `labeled` |
| `subject` | string | 科目名（仅 MMLU 有，其他为空字符串） |
| `question` | string | 题目文本 |
| `context` | string | 上下文（仅 PubMedQA 有 CONTEXTS，其他为空） |
| `options` | list[str] | 选项文本列表 |
| `answer_idx` | int | 正确答案在 `options` 中的索引（0-based） |
| `answer` | string | 正确答案文本 |

### 示例

```json
{
  "id": "medqa_train_000000",
  "source": "medqa",
  "split": "train",
  "subject": "",
  "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination...",
  "context": "",
  "options": ["Ampicillin", "Ceftriaxone", "Ciprofloxacin", "Doxycycline", "Nitrofurantoin"],
  "answer_idx": 4,
  "answer": "Nitrofurantoin"
}
```

---

## 三、各数据集详细说明

### 1. MedQA（USMLE 英文版）

美国医师执照考试（USMLE）题目，是最经典的医学问答基准。

- **文件**：`data/medqa_train.jsonl`、`data/medqa_dev.jsonl`、`data/medqa_test.jsonl`
- **切分**：train 10,178 / dev 1,272 / test 1,273（与论文原始切分完全一致）
- **选项**：5 个（A–E）
- **字段**：question / options / answer_idx / answer
- **论文**：Jin et al., "What Disease does this Patient Have?" (arXiv:2009.13081, 2020)
- **来源**：官方 Google Drive（另含 4 选项版、中国大陆/台湾版本、教科书，见原始数据目录 `medqa/`）

### 2. MedMCQA

印度医学入学考试（AIIMS/NEET）题目，规模最大（约 19 万题）。

- **文件**：`data/medmcqa_train.jsonl`、`data/medmcqa_validation.jsonl`
- **切分**：train 182,822 / validation 4,183（无独立 test，官方 test 未公开）
- **选项**：4 个
- **字段**：question / options / answer_idx / answer
- **来源**：HuggingFace `medmcqa/medmcqa`（经 ModelScope 镜像）

### 3. MMLU（医学相关 12 科目）

MMLU（Massive Multitask Language Understanding）中医学/生物医学相关的 12 个科目。

**核心医学 6 科目**（test 合计 1,089）：

| 科目 | test 题数 |
|---|---|
| anatomy（解剖学） | 135 |
| clinical_knowledge（临床知识） | 265 |
| college_biology（大学生物） | 144 |
| college_medicine（大学医学） | 173 |
| medical_genetics（医学遗传学） | 100 |
| professional_medicine（职业医学） | 272 |

**扩展生物医学/健康 6 科目**（test 合计 1,748）：

| 科目 | test 题数 |
|---|---|
| high_school_biology（高中生物） | 310 |
| nutrition（营养学） | 306 |
| virology（病毒学） | 166 |
| human_aging（人类衰老） | 223 |
| human_sexuality（人类性学） | 131 |
| professional_psychology（职业心理学） | 612 |

- **文件**：`data/mmlu_{科目}_{split}.jsonl`（12 科目 × 3 切分 = 36 个文件）
- **切分**：每科目 test / dev(5) / validation
- **选项**：4 个
- **字段**：question / options / answer_idx / answer（多一个 `subject`）

### 4. PubMedQA

生物医学文献问答，答案是 yes/no/maybe 三选一（非传统 4 选 1）。

- **文件**：`data/pubmedqa_labeled.jsonl`
- **题数**：1,000（专家标注），yes 552 / no 338 / maybe 110
- **选项**：3 个（`["yes", "no", "maybe"]`）
- **字段**：question / context（文献上下文）/ options / answer_idx / answer
- **来源**：官方 GitHub `pubmedqa/pubmedqa`
- **说明**：另有 61.2K 未标注数据未收录

> **实验定位：PubMedQA 是「独立评测集」，不混入蒸馏训练数据。**
> 蒸馏训练只用纯 MCQ（MedQA / MedMCQA / MMLU）。训练完成后，再用 PubMedQA 做一次「泛化到判断题」的额外评测——如果蒸馏后的学生模型在没训练过的判断题上也接近/超过老师，说明蒸馏迁移的是医学推理能力，而不是背 MCQ 应试。

---

## 实验用法（蒸馏）

### 训练数据（纯 MCQ，不混判断题）
- **MedQA**：`medqa_train.jsonl` + `medqa_dev.jsonl`
- **MedMCQA**：`medmcqa_train.jsonl` + `medmcqa_validation.jsonl`
- **MMLU**：12 科目的 `test` 与 `validation`（`dev` 每科仅 5 条，通常不用）

### 评测集
- **主指标（MCQ）**：MedQA `test`、MedMCQA `validation`、MMLU 12 科目 `test` —— 用于判定「学生是否超越老师」
- **泛化指标（判断题）**：PubMedQA `labeled`（held-out，蒸馏时未使用）—— 训练完成后作为额外一步，测「蒸馏是否泛化到判断题」

> 提示：中英对照实验（中文已实现「学生全科超越老师」）时，Teacher 模型、蒸馏流程、评测集三处需与中文实验对齐，否则「超越」不可比。

---

## 四、data 目录结构

```
data/
├── medqa_train.jsonl            # 10,178 条
├── medqa_dev.jsonl              # 1,272 条
├── medqa_test.jsonl             # 1,273 条
├── medmcqa_train.jsonl          # 182,822 条
├── medmcqa_validation.jsonl     # 4,183 条
├── mmlu_anatomy_test.jsonl      # 135 条
├── mmlu_anatomy_dev.jsonl
├── mmlu_anatomy_validation.jsonl
├── mmlu_clinical_knowledge_*.jsonl
├── mmlu_college_biology_*.jsonl
├── mmlu_college_medicine_*.jsonl
├── mmlu_medical_genetics_*.jsonl
├── mmlu_professional_medicine_*.jsonl
├── mmlu_high_school_biology_*.jsonl
├── mmlu_nutrition_*.jsonl
├── mmlu_virology_*.jsonl
├── mmlu_human_aging_*.jsonl
├── mmlu_human_sexuality_*.jsonl
├── mmlu_professional_psychology_*.jsonl
└── pubmedqa_labeled.jsonl       # 1,000 条
```

---

## 五、下载方法

> 国内访问 HuggingFace 极慢且新版 Xet 协议易报错，**推荐用 ModelScope 国内镜像**。

### 1. MedQA（Google Drive）

官方完整数据在 Google Drive（含 US/Mainland/Taiwan + textbooks）：

```
https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing
```

大文件会返回「病毒扫描警告」确认页，需提取 `confirm` 参数。**多线程下载（走代理提速）**：

```bash
# 走 Clash 代理 + 多线程（aria2c 直连 Google Drive 会 SSL 握手失败，需关闭证书校验）
aria2c --all-proxy=http://127.0.0.1:7890 --check-certificate=false \
  -x 16 -s 16 -k 1M \
  -o data_clean.zip \
  "https://drive.usercontent.google.com/download?id=1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw&export=download&authuser=0&confirm=t"
```

解压后英文题在 `data_clean/questions/US/`（train/dev/test.jsonl + 4_options/）。

### 2. MedMCQA / MMLU（ModelScope 镜像）

ModelScope 国内 CDN 快，直接下载：

```bash
# 安装 modelscope
pip install modelscope

# MedMCQA
aria2c -x 8 -s 8 "https://modelscope.cn/datasets/extraordinarylab/medmcqa/resolve/master/data/train-00000-of-00001.parquet"

# MMLU（以 anatomy 为例，6 核心 + 6 扩展共 12 科目）
aria2c -x 8 -s 8 "https://modelscope.cn/datasets/cais/mmlu/resolve/master/anatomy/test-00000-of-00001.parquet"
```

通用 URL 格式：`https://modelscope.cn/datasets/{owner}/{repo}/resolve/master/{path}`

### 3. PubMedQA

- 官方标注版（1,000 条）：GitHub 仓库

```bash
curl -L -o pqa_labeled.json \
  "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
```

- 扩展版（instruction 格式 11,000 条）：ModelScope `hiyouga/PubMedQA`

---

## 六、复现脚本

转换脚本（原始数据 → 统一格式）在 `scripts/convert_to_unified.py`（未推送到远端，仅本地保留）。
