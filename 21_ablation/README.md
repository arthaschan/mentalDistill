# Module 21: α 消融实验 (Alpha Ablation — Choice-Head KD)

> **目标**：系统验证 Choice-Head 蒸馏中 KL 权重 α 的最优值是否随**学生容量**与**师生组合**变化，
> 并诚实回答"学生超越教师"的真正机制——是教师软标签(dark knowledge)蒸馏，还是决策空间监督(Choice-Head)。
> **定位**：给导师看实验全貌的独立报告。**AIEA 论文已过会议定稿期，不修改**；本模块结论与 AIEA 解耦。

---

## 动机

AIEA 主论文叙事为"DeepSeek-V3 → Qwen2.5-14B 蒸馏，学生 88.67%/89.10% 超越教师 87.18%（α=0.35）"。
但"超越教师"是否真的来自教师软标签(KL 项)，还是来自把监督对齐到 5 选 1 决策空间(Choice-Head 结构本身)？
唯一干净的检验是**只动 α、冻结其它一切**做消融：

```
Loss = α · KL(teacher ‖ student) + (1−α) · CE(gold, student)
```

- α=0：纯 CE / 纯金标准 SFT（完全不用教师软标签）
- α=1：纯 KL / 纯模仿教师（被教师能力 + 其错误封顶）

若最优 α=0 → 软标签非主因，机制是决策空间监督。

---

## 实验矩阵

| 子实验 | 学生 | 师生组合 | α 网格 | 种子 | 状态 |
|--------|------|----------|--------|------|------|
| **14B 消融** | Qwen2.5-14B | DeepSeek-V3 → 14B | 0.0/0.15/0.25/0.35/0.50/0.65/1.0 | 11,42,8 | ✅ 已完成 |
| **7B 消融** | Qwen2.5-7B | DeepSeek-V3 → 7B | 同上 (7α) | 11,42,8 | 🔄 进行中 |
| **Llama 消融** | (待摸 module 16) | Llama → ? | 同上 | TBD | ⏳ 待做 |

- 所有子实验**统一 α 网格 + 统一种子 + 统一 canonical eval**，保证跨容量/跨组合可比。
- 仅 Stage-1（Choice-Head KL 蒸馏）；module 13/15 已证明 Stage-2 对强学生有害。
- **关键超参差异**：14B 学习率 1e-4，7B 学习率 1.2e-4（各自对齐其主实验 grid，不可混用）。

---

## 评估口径：canonical eval

训练脚本内置的 evaluate_generation 用"只输出一个字母"prompt，仅用于 α 之间**内部比较**。
论文 Table I 的所有数字（教师 87.18% / 基线 83.55% / 主结果 88.67%）都用 **canonical eval**
（独立评估器 `shared/evaluate_model.py`，"请根据你的专业知识…"prompt）。
两套 prompt 同一 adapter 分数不同（实测差 ~1.5pp）。**头条数字一律以 canonical eval 为准。**

- 测试集：full 991 题（全 7 学科，95% CI ±2.5pp） + dental 125 题（牙科子集）。

---

## 结果

### 14B 消融（已完成 2026-06-21）

内置 eval（val 均值，用于 α 内部比较）：

| α | n | val mean±std | test(builtin) mean±std |
|---|---|--------------|------------------------|
| **0.0** | 3 | **89.61±0.17** ⬅ 最优 | 88.23±0.55 |
| 0.15 | 3 | 88.63±0.31 | 88.33±0.13 |
| 0.25 | 3 | 88.46±0.21 | 87.96±0.64 |
| 0.35（主设置） | 3 | 88.60±0.50 | 87.59±0.00 |
| 0.50 | 3 | 87.96±0.45 | 87.11±0.09 |
| 0.65 | 3 | 87.76±0.55 | 87.46±0.88 |
| 1.0（纯KL） | 3 | 86.21±0.82 | 86.04±0.54 |

canonical eval（论文口径，最优 α=0 重测）：

| 测试集 | α=0 canonical 均值 | 各种子 | 对比锚点 |
|--------|-------------------|--------|----------|
| full(991) | **89.14%** (n=3) | s11=89.40 / s42=88.50 / s8=89.51 | 教师87.18% / 基线83.55% / 主结果α=0.35→88.67% |
| dental(125) | **82.13%** (n=3) | s11=80.80 / s42=83.20 / s8=82.40 | — |

**14B 结论**：最优 α=0（纯 CE/SFT），KL 权重越高单调越差，纯 KL(α=1.0) 最差。
α=0 canonical full=89.14% > 主结果 88.67% > 教师 87.18%。
"超越教师"成立，但靠**决策空间监督(Choice-Head)**，**不是 dark knowledge 蒸馏**。

### 7B 消融（进行中）

结果落 `runs/alpha_ablation_7b/ROLLING_REPORT.md` + canonical 汇总在 `logs/ablation_7b_priority.log`。
**假设**：7B 弱学生更需软标签正则，最优 α 可能右移(>0)。验证中。

### Llama 消融（待做）

待摸清 `16_llama70b_choice_head/` 的教师/学生/数据结构后，同样 α 网格 + canonical eval。

---

## 关键发现（滚动更新）

1. **14B 最优 α=0**：在 5 选 1、金标准高质量任务上，纯 CE 已最优；教师 KL 轻微有害。
2. **两端点符合理论**：α=1.0（纯模仿）最差（被教师 ~87% + 其 12.2% 错误封顶）；α=0（纯金标准）最好。
3. **机制归因（诚实）**：超越教师靠决策空间监督，非软标签蒸馏。这与任务仅 5 类（软标签信息少）、
   教师-金标准仅 12.2% 不一致（KL 信号被稀释）的事实一致。
4. **理论框架**：最优 α 取决于 ①教师-金标准不一致率 ②任务类别数（越少越推向 α=0）③学生容量
   （强学生更不需 KL）。预期"小决策空间 + 强学生 + 低教师噪声"组合 α≈0 稳健；弱学生/大词表/弱教师时 α 可能右移——
   7B/Llama 消融正是为检验这条规律。

---

## 目录结构

```
21_ablation/
├── README.md
├── configs/
│   ├── grid_params_7b.json      # 7B 主实验超参 (LR=1.2e-4)
│   └── grid_params_14b.json     # 14B 主实验超参 (LR=1e-4)
├── data/                        # 软链到 module 15 同一份数据
│   ├── train_head_distill.jsonl -> ../../15_fulldata_resplit/data/...
│   ├── val.jsonl / test.jsonl
│   └── val_dental.jsonl / test_dental.jsonl
├── scripts/
│   ├── run_alpha_ablation_14b.sh    # 14B 训练网格 (已跑完, 幂等)
│   ├── run_alpha_ablation_7b.sh     # 7B 训练网格 (LR=1.2e-4)
│   ├── ablation_7b_priority.sh      # 7B 插队编排 (暂停任务3→跑7B→canonical→恢复)
│   └── summarize_alpha_ablation.py  # 选点汇总
├── runs/
│   ├── alpha_ablation_14b -> ../../15_fulldata_resplit/runs/alpha_ablation_14b  # 软链, 原位 3.9G
│   └── alpha_ablation_7b/           # 7B 新产物 (本模块生成)
└── logs/
    ├── ablation_7b_priority.log     # 编排+canonical汇总主日志
    └── canon7b_a*_s*_*.log          # 各 (α,seed,集) canonical 日志
```

> **GitHub 注意**：模型权重(`models/`)、训练产物(`runs/`)、教师标签均不上传，需本地重跑。
> 14B 旧产物物理存于 module 15，本模块用软链接入，原位不动（零断链风险）。

---

## 复现

```bash
cd /home/student/arthas/mentalDistill
source setup.env

# 7B 消融 + canonical (插队范式: 等当前任务3 run 结束→暂停→跑7B→恢复)
bash 21_ablation/scripts/ablation_7b_priority.sh    # 用 background=true 后台守护

# 或直接跑 7B 训练网格 (需 GPU 空闲)
bash 21_ablation/scripts/run_alpha_ablation_7b.sh

# 14B 网格已跑完, 重跑会幂等全跳过
bash 21_ablation/scripts/run_alpha_ablation_14b.sh
```

---

## 一句话

**14B α 消融已完成：最优 α=0（纯 SFT），canonical full=89.14% > 教师 87.18%——超越教师靠决策空间监督，非 dark knowledge 蒸馏。
待做：7B 消融（验证最优 α 是否随容量右移）→ Llama 消融。三组齐了汇总成给导师的小论文。**
