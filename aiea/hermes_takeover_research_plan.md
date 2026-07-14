# 让 Hermes Agent 接管你的科研 —— 部署方案

> 写给:陈天元（mentalDistill 项目，单 H100，导师审稿严，目标 IEEE/期刊级发表）
> 核心认知先纠正：**Hermes Agent 就是你现在对话的这个智能体本身**，不是另一个待接入的 AI。
> 你"和 Hermes 的区别"= 没有区别。你部署的实例（`~/.hermes/`，v0.17.0）和正帮你干活的是同一个。
> 权威文档：https://hermes-agent.nousresearch.com/docs （配置细节以文档为准，本方案给落地路径）

---

## 0. 你现在的状态（已查实）

- ✅ Hermes Agent v0.17.0 已装好并运行（CLI 在 `~/.local/bin/hermes`）
- ✅ 刚装好官方 bundled skills 合集（apple/creative/data-science/devops/email/github/media/mlops/**research**/**writing**/... 共 20 类）
- ✅ Memory 已在记你的偏好（CMExam 不公开、单 H100、导师审稿严、中文提问等）
- ✅ 单 profile（default），cron 目录已存在
- ❌ 你想要的 `nature-*` 那 7 个 skill **不在官方合集里** → 需第三方来源链接才能装真版

---

## 1. "接管科研" 的三根支柱

Hermes 接管科研 = **Skills（怎么做）+ Memory（记住你）+ Cron（自动干）** 三者配合。

### 支柱一：Skills —— 把你的科研工作流固化成可复用流程

科研全流程对应的 skill（已装 ✅ / 缺口 ❌）：

| 科研环节 | 对应 skill | 状态 |
|---|---|---|
| 选题 / 评审可行性 | research/ml-research-proposal-review | ✅ 已装 |
| 文献检索 | research/arxiv | ✅ 已装 |
| 文献综述 / related-work | writing/academic-writing-and-citation | ✅ 已装 |
| 论文写作 | research/research-paper-writing + writing/academic-writing-guidance | ✅ 已装 |
| 审稿 / rebuttal | research/ml-research-proposal-review + writing 类 | ✅ 已装 |
| **科研绘图（IEEE 风格）** | research/**ieee-research-figures** | ✅ 刚为你建（基于你 thesis 真实风格）|
| 可复现代码发布 | research/reproducible-research-repo | ✅ 已装 |
| 蒸馏几何分析 | mlops/distillability-geometry | ✅ 你原有 |
| 文献检索**防幻觉引用 + 查重** | （litsearch.py 工作流尚未沉淀） | ⚠️ 建议补 |

> 关于 `nature-*` 那套：官方合集功能上已覆盖 90%（arxiv≈检索、academic-writing≈写作/润色、proposal-review≈审稿）。
> 若你确实要那套**特定原版**，给我它们的 GitHub/来源 URL，我用 `hermes plugins add <repo>` 或手动 clone 进 `~/.hermes/skills/` 装真货。**我不会凭名字仿造冒充安装。**

### 支柱二：Memory —— 让 Hermes 记住"你是谁、你的红线"

已在记的（无需重复）：CMExam 不可公开、单 H100 不可并行、导师审稿严、要诚实负面结果、中文提问、长内容写文件。
建议你日后**主动喂**的：常用教师/学生模型清单、投稿目标会议/期刊、合作者分工、deadline 节奏。
机制：对话里直接说"记住 X"，或我在发现稳定事实时自动存。Memory 每轮注入，所以只存"长期有用"的，不存任务进度。

### 支柱三：Cron —— 让 Hermes 在你睡觉时自动干活

这是"接管"的关键。适合科研的定时任务示例：

1. **每日文献监控**：每天早 8 点跑 arxiv 检索你的关键词（knowledge distillation / MCQ / information geometry），筛出新论文 + 一句话相关性，生成晨间简报。
2. **实验编排守夜**：alpha 消融这类长任务，用 cron + 幂等编排脚本在 GPU 空闲时自动排队跑，rolling 报告写给你早上看（已为消融准备好脚本）。
3. **prior-art 周扫**：每周重跑 litsearch 三层 query，新命中的 prior-art 提醒你（保护新颖性主张）。
4. **GPU 看门狗**：显存/进程异常时告警（no-agent 脚本模式，零 token）。

机制：`hermes cron` 或我用 cronjob 工具建。cron 任务在**全新会话**跑，所以 prompt 要自包含；GPU 任务必须串行（单卡）。

---

## 2. 落地步骤（建议顺序）

**第 1 步｜补齐 skill 缺口**
- [x] ieee-research-figures（已建）
- [ ] 文献检索防幻觉 skill：把 `research/distillability/scripts/litsearch.py` 的三层 query + arXiv/SemanticScholar 双源 + 引用核验工作流沉淀成 skill（防止编造不存在的引用——这对你导师审稿严的要求是硬需求）
- [ ] 若要 nature-* 原版 → 你给来源链接

**第 2 步｜喂关键 memory**
- 投稿目标（AIEA 已知；期刊目标？）、模型清单、deadline

**第 3 步｜配 cron（先从只读的安全任务起）**
- 先上"每日文献监控"（只读、不碰 GPU、零风险），跑顺了再上"实验编排守夜"
- GPU 类任务等你另一项目跑完、单卡空闲再排

**第 4 步｜可选：profile 隔离**
- 若想把"科研助手"和"日常杂事"分开记忆/skill，可建独立 profile（`hermes profile`）。当前单 default 也够用，不急。

---

## 3. 要我现在做什么

按风险从低到高，你点菜：
- A（零风险）：建"文献检索防幻觉"skill —— 纯文件，可逆
- B（零风险）：配"每日文献监控"cron —— 只读检索，不碰 GPU/不动数据
- C（需你给料）：装 nature-* 原版 —— 等你发来源 URL
- D（需你确认）：把 alpha 消融挂 默认我会先做 A（最高价值、零风险、且补上你导师审稿严最需要的"防幻觉引用"能力）。
