# fullEnglish — 全医科英文数据「学生超越老师」完整方案

> 目标：在**全医科英文选择题**（MedQA + MedMCQA + MMLU-med，共 20 万题）下，探索
> 学生（Qwen2.5-32B）能否**超越**教师（DeepSeek / Llama / Gemini 等）。
> 中文全科实验已实现「学生超越老师」（14B 89.14% > 教师 87.18%），本方案检验该结论能否跨语言复制。
> 本文档自包含；所有数字与配置直接对应 `fullEnglish/` 下的可执行脚本。

---

## 0. 一句话定位

把中文 CMExam 上已验证的 **Choice-Head 蒸馏（α=0 纯 GT SFT / 决策空间监督）** 迁移到**英文全科医学 MCQ**，
先做**教师预评估**（筛选 DeepSeek / Gemini / Llama 里谁最强、headroom 多大），
再用最强教师做基准，检验 32B 学生能否在宽领域英文医学上复现「学生超越老师」。

核心赌注：中文规律是「**宽领域能超越教师、专科难超越**」（全科 89.14%>87.18% 超越 / 牙科子集 82.13% 未超越；
英文牙科 75.18% vs 88.30% 未超越）。fullEnglish 是**宽领域**，落在「能超越」一侧；
且学生从 14B 升级到 **32B**，天花板更高，超越概率更大。

---

## 1. 已有结论（不是从零开始，数字来自本项目既有实验）

| 结论 | 出处 |
|---|---|
| 中文全科：学生 14B **89.14%** > 教师 DeepSeek-V3 **87.18%**（headroom≈3.6pp，**超越**） | aiea/alpha_ablation_results.md §4.9 |
| 中文牙科子集(125)：学生 82.13%，**未超越** | 同上 |
| 英文牙科：学生 14B **75.18%** vs 教师 **88.30%**（headroom≈20pp，**未超越**） | english/03_main_distill/CP3 |
| **α=0（纯 CE / 决策空间监督）最优**，KL 权重越高越差（中文+英文牙科双重复制） | aiea §0 / english CP3 §3 |
| 「超越教师」靠**决策空间监督**，不靠 dark-knowledge 蒸馏（教师-GT 仅 12.2% 不一致，任务仅 4–5 类） | aiea §2 |
| 教师预评估（zero-shot 先验）能提前判定 headroom：headroom 小→可超越，headroom 大→难超越 | english/01_teacher_screening |

**迁移策略**：教师阵容（DeepSeek/Gemini/Llama）+ 教师预评估 + α=0 头条臂，三处与中文实验对齐，保证「超越」可比。

---

## 2. 数据（fullEnglish/data，已全部统一格式）

| 数据集 | 题数 | 选项 | 角色 |
|---|---|---|---|
| MedQA (USMLE) | 12,723 | 5 选 1 | train 10,178 / dev 1,272 / test 1,273 |
| MedMCQA (印度 AIIMS/NEET) | 187,005 | 4 选 1 | train 182,822 / validation 4,183（无公开 test） |
| MMLU 医学 12 科目 | 3,207 | 4 选 1 | test ~2,837 / validation ~310 / dev 每科 5 |
| PubMedQA | 1,000 | 3 选 1（yes/no/maybe） | **独立评测集，蒸馏全程不用** |

### 切分决策（Phase 0 已产出，数字为真实运行结果）

| 集合 | 题数 | 说明 |
|---|---|---|
| **train** | **20,488** | MedQA train 10,178 + MedMCQA train 抽样 10,000 + MMLU validation 310 |
| **val**（选点） | **1,272** | MedQA dev（干净 hold-out） |
| **test_medqa** | 1,273 | 主测试（5 选 1，最干净） |
| **test_medmcqa** | 4,183 | 跨来源测试（4 选 1） |
| **test_mmlu** | 2,837 | 12 科目测试（4 选 1） |
| **test_pubmedqa** | 1,000 | 判断题泛化评测（held-out） |
| **screen_input** | 600 | 每测试集抽 200，教师预评估 + 学生零样本地板 |

> MedMCQA train 抽样量可调（`--medmcqa_sample`，默认 1 万；0=全部 18 万）。
> 训练数据**纯 MCQ**，不混判断题（PubMedQA），保证「泛化到判断题」是干净的额外评测。

---

## 3. 教师与学生

| 角色 | 模型 | 类型 | 说明 |
|---|---|---|---|
| **学生** | Qwen2.5-32B-Instruct | 本地 | LoRA r=16，用户指定「继续用 32B」 |
| 教师 | DeepSeek-V3 (deepseek-chat) | API | 中文/英文牙科均冠军，默认主教师（key 已配，实测有效） |
| 教师 | Gemini (flash-latest / pro-latest) | API | 免费额度；**需开 Clash 代理**，否则地区受限（实测） |
| 教师 | Llama-3.3-70B | 本地 AWQ(vLLM) 或 硅基流动 API | 官方 API 无法注册（国家原因），本机 vLLM 首选 |
| 参考 | Qwen2.5-14B / Gemma-27B / Phi-4 / Yi-34B | 本地 logprobs | 免费补充教师 + 学生零样本地板 |

API key 在 `setup.env`（`DEEPSEEK_API_KEY` / `GEMINI_API_KEY` / `SILICONFLOW_API_KEY` / `OPENROUTER_API_KEY` / `GROQ_API_KEY`），本地教师无需 key。

---

## 4. 实验流程（四个阶段，checkpoint 审批制）

### Phase 0 数据装配（✅ 已完成，零 GPU）
`00_data/build_data.py` → 统一格式转 trainer 格式 + 切分 + 审计。真实数字见 §2。

### Phase 1 教师预评估（CP1 停下报数字）
`01_teacher_screening/run_screening.sh` → 在 screen_input(600) 上 zero-shot 跑全部教师，
`aggregate_screening.py` 产出**教师能力先验表**（总分 + 分源 + 学生零样本地板 + headroom）。

**GO/NO-GO 判据（预注册）**：
- headroom = 最强教师 acc − 学生零样本 acc。
- headroom ≤ 8pp：参照中文（≈3.6pp 超越），直接进主实验。
- headroom > 8pp：参照英文牙科（≈20pp 未超越），仍跑主实验但如实报告，走机制分析。
- 淘汰线：acc < 学生零样本 或 < 55% 的教师不进主实验（弱教师蒸馏历史为负）。

### Phase 2 多教师融合上界（可选，零训练）
`02_fusion_oracle/fusion_oracle.py` → 复用 Phase 1 标签，算 majority_vote / domain_route_CV /
prob_avg 等可实现上界。**≥2pp 走融合，<0.5pp 走单教师**（中文=死、英文牙科=有互补但不可捕获，看 fullEnglish 落在哪）。

### Phase 3 主蒸馏（CP2 后启动，头条 α=0）
`03_main_distill/run_main_distill.sh`：
1. 主教师标签：train 软标签 + 三个测试集同集准确率（学生同题对标锚）。
2. 构造 `train_head_distill.jsonl`（AIEA 配方，smooth_eps=0.25）。
3. 学生零样本地板（32B，无 adapter，4 个测试集）。
4. **头条 α=0 × 3 seed**（11/42/8）；可选 `RUN_ALPHA_SWEEP=1` 加 α∈{0.35,1.0} 复现跨语言消融。
5. 评估 + 聚合。

**训练配置**（对齐中文 14B 冠军配，32B 按显存调 batch）：
`num_epochs=1, batch=1×grad_accum=8, lr=1e-4, LoRA r=16/α=32, DISTILL_PROMPT_LANG=en, deterministic`。

### Phase 4 评估与判定（CP3 报结果）
`04_eval` + `aggregate_results.py` → 分测试集「教师 vs 学生 α=0」同题集对比 + 组合 MCQ 加权 Δ + PubMedQA 泛化。

**头条判定**：组合 MCQ（MedQA+MedMCQA+MMLU 加权）上，学生 α=0 mean > 教师 → **超越**。

---

## 5. 里程碑 / checkpoint（分阶段审批）

- **CP1**｜Phase 0 + Phase 1 完成：数据审计 + 教师先验表 + headroom → **在此停下，报数字请示**。
- **CP2**｜Phase 2 融合上界（可选）→ 决定单教师 or 融合。
- **CP3**｜Phase 3+4 完成：学生 vs 教师同题集对比 → 判定是否超越。
- **CP4**｜写作交接（prior-art 查重 + 自审 + 成文，复用 english/ 阶段技能）。

---

## 6. 风险清单（投稿前逐条回应）

1. **32B 显存/训练时长**：32B LoRA 单 H100 95GB 可行（batch1×8），但 α 扫描×3seed 全套约 20–40 GPU 时。默认只跑 α=0×3seed（≈6–12 GPU 时），α 扫描显式开启。
2. **MedMCQA 印度分布 vs MedQA 美国分布**：英文牙科曾因「训练 82% 印度 / 测试英美」产生分布错配 confound。本方案 train 里 MedQA(10,178) 与 MedMCQA(10,000) 约 1:1，且主测试 test_medqa 有同源 MedQA train 覆盖，已规避。跨来源 test_medmcqa / test_mmlu 单独报告，作分布泛化证据。
3. **教师太强 headroom 过大**：若 Gemini/DeepSeek 在英文医学上断层领先（如英文牙科 88% vs 学生 68%），单靠决策空间监督可能填不平。对策：32B 学生零样本本身强（预计 MedQA ~80%+），headroom 大概率落在可超越区间；若真过大，如实报告并升级到机制分析（同英文牙科）。
4. **α=0 最优是否推翻「蒸馏」叙事**：否。α=0 即「决策空间监督」本身是方法核心，Teacher 仍是基准与可选软标签源；措辞按 aiea §3 改写。
5. **判断题泛化**：PubMedQA 3 选 1，Choice-Head 只训过 4–5 选 1；若学生在 PubMedQA 接近教师，说明迁移的是医学推理而非背题。若泛化失败，如实报告为边界。
6. **数据版权**：MedQA/MedMCQA/MMLU/PubMedQA 均为公开学术数据集，可引用来源；不发布原始题目，发布时脱敏。

---

## 7. 目录结构与执行

```
fullEnglish/
├── PLAN.md                      # 本文件（完整方案）
├── README.md                    # 数据说明（已有）
├── data/                        # 统一格式数据（已有）
├── 00_data/build_data.py        # ✅ Phase 0 数据装配
├── 01_teacher_screening/        # Phase 1 教师预评估
│   ├── run_screening.sh         #   API+本地教师 zero-shot 筛选
│   ├── aggregate_screening.py   #   先验表 + headroom
│   ├── candidates/*.json        #   DeepSeek/Gemini/Llama 候选
│   └── system_prompt_en.txt / trailing_instruction_en.txt
├── 02_fusion_oracle/fusion_oracle.py   # Phase 2 融合上界（可选）
├── 03_main_distill/             # Phase 3 主蒸馏
│   ├── generate_teacher_labels.sh
│   ├── build_train_head.py
│   ├── run_main_distill.sh
│   ├── evaluate_all.py
│   └── aggregate_results.py
└── 04_eval/eval_mcq.py          # Phase 4 统一英文评估器
```

执行（从仓库根）：
```bash
source setup.env                       # 填好 API keys
python3 fullEnglish/00_data/build_data.py                 # Phase 0 (已跑)
bash   fullEnglish/01_teacher_screening/run_screening.sh  # Phase 1 → CP1 停下
python3 fullEnglish/02_fusion_oracle/fusion_oracle.py    # Phase 2 (可选)
bash   fullEnglish/03_main_distill/run_main_distill.sh    # Phase 3+4 → CP3 报结果
```

---

## 8. 立即执行

Phase 0 已跑通（真实数字见 §2）。下一步是 **Phase 1 教师预评估**（需 API key + 单 H100，
本地教师部分无需 key 可先跑）。跑完在 CP1 停下，报「教师先验表 + headroom」，据此定主教师与预期。
