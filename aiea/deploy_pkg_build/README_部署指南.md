# Choice-Head 蒸馏 14B 学生模型 —— 最小可用部署指南

本包是论文 *Choice-Head Distillation for Dental Multiple-Choice Question Answering*（AIEA 2026）中
**全量测试集 89.10% 最佳成绩**对应的学生模型（Qwen2.5-14B + Stage-1 Choice-Head LoRA，seed=8）。

> 关键认识：本包**不含 14B 基座模型权重**，只含约 50MB 的 LoRA adapter（增量补丁）。
> adapter 必须挂在 Qwen2.5-14B-Instruct 基座上才能运行。基座需你自行下载（见第 2 步）。

---

## 0. 包内容

```
choice_head_14b_s8_deploy/
├── README_部署指南.md                 # 本文件
├── adapter/                          # Stage-1 best LoRA（seed=8，89.10%）
│   ├── adapter_model.safetensors     # ~50MB，LoRA 增量权重（真正的“学生模型”）
│   ├── adapter_config.json           # 指向基座路径（需改，见第 3 步）
│   ├── tokenizer.json / vocab.json / merges.txt / *.json  # 分词器（自包含）
│   └── chat_template.jinja
├── scripts/
│   ├── evaluate_model.py             # 评估脚本（独立，无外部相对依赖）
│   ├── run_eval.sh                   # 一键评估封装
│   └── infer_demo.py                 # 单题推理示例
├── data/
│   ├── test_full_991.jsonl           # 991 题全量测试集（复现 89.10%）
│   └── test_dental_125.jsonl         # 125 题牙科子集（复现 78.40%）
└── requirements.txt                  # Python 依赖
```

---

## 1. 环境配置

硬件：单卡 GPU，显存 ≥ 32GB（14B BF16 + LoRA 推理约需 30–32GB）。论文使用 NVIDIA H100 NVL 95GB。

```bash
# 建议新建独立 conda 环境
conda create -n choicehead python=3.10 -y
conda activate choicehead

# 安装依赖（先按官方指引装好匹配 CUDA 的 PyTorch）
# 例：CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 其余依赖
pip install -r requirements.txt
```

`requirements.txt` 关键版本：
```
torch>=2.0.0
transformers>=4.51.0
peft>=0.7.0
accelerate>=0.20.0
numpy>=1.24.0
```

---

## 2. 下载 14B 基座模型

学生基座是公开模型 `Qwen/Qwen2.5-14B-Instruct`（约 28GB），自行下载：

```bash
# 方式一：huggingface-cli
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
    --local-dir ./Qwen2.5-14B-Instruct

# 方式二（国内更快）：ModelScope
pip install modelscope
python -c "from modelscope import snapshot_download; \
snapshot_download('Qwen/Qwen2.5-14B-Instruct', local_dir='./Qwen2.5-14B-Instruct')"
```

记下基座的**绝对路径**，例如 `/data/models/Qwen2.5-14B-Instruct`，下一步要用。

---

## 3. 修改路径

`adapter/adapter_config.json` 里 `base_model_name_or_path` 仍是训练机的旧路径，需改成你的基座路径。
**两种方式任选其一：**

方式 A（推荐，不改文件）：运行时用 `--base_model` 参数显式指定基座，脚本会覆盖配置里的路径。无需改任何文件。

方式 B（改文件）：把 `adapter/adapter_config.json` 第 4 行改为你的基座绝对路径：
```json
"base_model_name_or_path": "/data/models/Qwen2.5-14B-Instruct",
```

---

## 4. 运行评估（复现 89.10%）

```bash
cd choice_head_14b_s8_deploy

# 全量 991 题 —— 预期 ≈ 89.10%
bash scripts/run_eval.sh /data/models/Qwen2.5-14B-Instruct data/test_full_991.jsonl

# 牙科 125 题 —— 预期 ≈ 78.40%
bash scripts/run_eval.sh /data/models/Qwen2.5-14B-Instruct data/test_dental_125.jsonl
```

或直接调脚本：
```bash
python scripts/evaluate_model.py \
    --base_model /data/models/Qwen2.5-14B-Instruct \
    --adapter_dir ./adapter \
    --test_data ./data/test_full_991.jsonl \
    --wrong_log ./test_wrong.jsonl
```
输出示例：`测试集准确率: 89.10% (883/991)`

> 评估口径说明：本脚本使用论文 Table I 的 **canonical 评估**（system prompt = “你是一位专业的
> 牙科医生…只输出一个大写字母”，确定性贪婪解码 `do_sample=False`，`max_new_tokens=4`）。
> 换 prompt 或开采样会得到不同数字，复现请勿改这些设置。

---

## 5. 单题推理示例

```bash
python scripts/infer_demo.py --base_model /data/models/Qwen2.5-14B-Instruct
```
`infer_demo.py` 内置一道示例题，演示如何加载 base+adapter 并对单题预测 A/B/C/D/E。
改成你自己的题目即可集成到应用里。

---

## 6. 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| `OSError: ... is not a valid model identifier` | 基座路径错。用 `--base_model` 指向第 2 步下载目录的绝对路径 |
| `CUDA out of memory` | 显存不足。需 ≥32GB；或改用多卡 `device_map="auto"`（需改脚本）|
| 加载 adapter 报 base_model 路径找不到 | adapter_config.json 旧路径未改。用第 3 步方式 A 传 `--base_model` 即可绕过 |
| 准确率明显偏低（如 <85%） | 多半改了 prompt 或开了采样。务必用 `do_sample=False` + 本包 system prompt |
| transformers 版本报 Qwen2 不识别 | 升级 `transformers>=4.51.0` |

---

## 7. 关于本模型的诚实说明（供使用者知情）

- 89.10% 是 **seed=8、仅 Stage-1（Choice-Head KL 蒸馏）** 的最佳单次结果；3-seed 均值为 88.67%，
  均高于 DeepSeek-V3 教师 87.18%。本包提供的就是该最佳 seed。
- 论文中 **Stage-2（GT SFT 精校）并不稳定提升**性能，故最小部署只用 Stage-1 best。
- 训练教师为 DeepSeek-V3（仅训练期通过 API 提供软标签）。**部署后的本模型不含、也不调用 DeepSeek**，
  纯本地推理。
- 测试数据源自 CMExam（中文医学考试）。若再分发测试集，请遵守 CMExam 的数据许可。
