# H100 环境重建方案（mentalDistill 项目）

> 目的：H100 上的项目被删后，未来重新登录能一键重建整个环境（含 Hermes）。
> 本文档 + 同目录脚本自包含。按顺序执行 00→05 即可。
> 编写日期 2026-07-16，基于当时真实环境快照（下方“环境快照”为事实来源）。

---

## 0. 环境快照（重建目标，来自被删前的真实机器）

| 项 | 值 |
|---|---|
| 操作系统 | Linux（内核 6.8.0），主机名 user-RS720A-E12-RS12 |
| GPU | **单张 NVIDIA H100 NVL, 95GB**（注意：只有1张物理卡，不是2张） |
| GPU 驱动 | 575.57.08（CUDA 12.8 运行时兼容） |
| 项目路径 | /home/student/arthas/mentalDistill |
| 代码仓库 | https://github.com/arthaschan/mentalDistill.git |
| 项目 Python | Anaconda3 的 base：/home/student/anaconda3/bin/python3（Python 3.13.5） |
| 关键训练依赖 | torch 2.9.1+cu128, transformers 4.57.6, peft 0.7.1, datasets 3.6.0, accelerate, sentence-transformers |
| vLLM 环境 | 独立 conda 环境 `vllm`（vllm 0.16.0，用于 Llama-3.3-70B-AWQ 等大教师） |
| Hermes | 装在 ~/.hermes/hermes-agent（git 仓库 + uv 建的 venv，Python 3.11.15，版本 0.17.0） |
| 模型 | models/ 下 12 个模型，共 **382GB**（清单见 §4） |
| 环境激活 | `source setup.env`（定义模型路径变量 + API key + 快捷命令） |

**关键坑（务必记住）**：
- 这台机器**只有 1 张 H100**。单卡 95GB 不能同时并发跑两个大任务（14B LoRA 训练约占 40-50GB）。GPU 作业必须**顺序排队**，勿并行。
- vLLM 必须用**独立 conda 环境**：autoawq/transformers4.57 在 base 环境有 PytorchGELUTanh 崩溃问题，隔离到 `vllm` 环境可避开。
- Hermes 用 **uv** 装（不是 conda/pip 直装），Python 3.11。

---

## 1. 重建步骤总览（脚本在本目录，按序执行）

```bash
cd ~/arthas/mentalDistill/rebuild_h100    # 重建后仓库会在这，先临时放别处也行
bash 00_check_system.sh        # 检查 GPU/驱动/磁盘，确认机器就绪
bash 01_install_base.sh        # 装 Anaconda(若无) + 系统工具 + git-lfs
bash 02_clone_repo.sh          # 克隆项目仓库
bash 03_python_env.sh          # 建 base 训练依赖 + 独立 vllm 环境
bash 04_download_models.sh     # 从 HuggingFace 下载 12 个模型（382GB，最久）
bash 05_install_hermes.sh      # 装 Hermes（uv + venv）
# 最后：手动填 setup.env 里的 API keys（见 §6），source setup.env
```

预计耗时：模型下载是大头（382GB，取决于带宽，几小时）；其余 10-30 分钟。

---

## 2. 前置要求（重建前确认）

- 能 SSH 登录 H100，有 sudo（装系统包）或已有 Anaconda。
- 磁盘：模型 382GB + 训练产物预留，**至少 600GB 可用空间**。
- 网络：能访问 github.com、huggingface.co、api.astral.sh（uv）、pypi。
- HuggingFace token（下部分模型需登录）：https://huggingface.co/settings/tokens
- 各 API key（DeepSeek/豆包/Moonshot/DashScope）——从你自己的密码管理器取，**不在本仓库**。

---

## 3. 各脚本做什么（详解）

- **00_check_system.sh**：打印 GPU 型号/显存/驱动、磁盘剩余、Python/conda 是否在。仅检查不改动，先跑它确认机器状态。
- **01_install_base.sh**：若无 Anaconda 则下载安装；装 git、git-lfs、wget、tmux 等。幂等（已装则跳过）。
- **02_clone_repo.sh**：git clone 项目到 ~/arthas/mentalDistill。
- **03_python_env.sh**：在 anaconda base 装训练依赖（requirements.txt + sentence-transformers）；另建独立 `vllm` conda 环境装 vllm。
- **04_download_models.sh**：用 huggingface-cli 下载 §4 清单里的模型到 models/。可断点续传，可注释掉已有的。
- **05_install_hermes.sh**：装 uv → clone hermes-agent → uv 建 venv → 装依赖 → 软链到 ~/.local/bin/hermes。

---

## 4. 模型清单（models/ 下，共 382GB）

从 HuggingFace 下载（04 脚本自动化）：

| 本地目录 | HuggingFace repo | 用途 |
|---|---|---|
| Qwen2.5-0.5B-Instruct | Qwen/Qwen2.5-0.5B-Instruct | 小student/调试 |
| Qwen2.5-1.5B-Instruct | Qwen/Qwen2.5-1.5B-Instruct | 小student |
| Qwen2.5-3B-Instruct | Qwen/Qwen2.5-3B-Instruct | student |
| Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | **主student 7B** |
| Qwen2.5-14B-Instruct | Qwen/Qwen2.5-14B-Instruct | **主student 14B** |
| Qwen2.5-32B-Instruct | Qwen/Qwen2.5-32B-Instruct | 教师/写作 |
| Qwen3-14B | Qwen/Qwen3-14B | student(Qwen3) |
| GLM-4-32B-0414 | THUDM/GLM-4-32B-0414 | 教师候选 |
| gemma-2-27b-it | google/gemma-2-27b-it | 教师候选 |
| phi-4 | microsoft/phi-4 | 教师候选 |
| Yi-1.5-34B-Chat | 01-ai/Yi-1.5-34B-Chat | 教师候选 |
| Llama-3.3-70B-Instruct-AWQ | casperhansen/llama-3.3-70b-instruct-awq | 大教师(需vLLM) |

> repo 名以官方为准；下载前用 `huggingface-cli` 搜索确认。AWQ 量化版仓库名可能变，04 脚本里可改。

---

## 5. 环境激活与快捷命令（setup.env）

项目根目录 `setup.env` 定义（重建后需自己补 API keys）：
- 模型路径变量：`BASE_MODEL_7B/14B/32B/QWEN3_14B`、`EASYEDIT_PY`
- API keys：`DEEPSEEK_API_KEY`、`DOUBAO_API_KEY`、`MOONSHOT_API_KEY`、`DASHSCOPE_API_KEY`
- 快捷命令（在被删机器上曾定义）：`hermes_env`（激活Hermes环境）、`vllm_env`（激活vLLM环境）、`start_vllm/stop_vllm/vllm_log`（vLLM服务）、`check_env`（环境诊断）、`test_all`

每次登录后：`cd ~/arthas/mentalDistill && source setup.env`

同目录提供 `setup.env.template`（不含真实密钥），重建后复制为 setup.env 再填。

---

## 6. API Keys（安全）

- **绝不提交到仓库**。setup.env 应在 .gitignore 里。
- 重建后从你自己的密码管理器/邮件里取，填进 setup.env：
  - DeepSeek（主教师）：https://platform.deepseek.com
  - 豆包 Doubao：火山引擎
  - Moonshot Kimi：https://platform.moonshot.cn
  - DashScope（通义千问 API）：阿里云

---

## 7. 验证重建成功

```bash
cd ~/arthas/mentalDistill && source setup.env
bash rebuild_h100/06_verify.sh   # 检查 GPU 可见、torch CUDA、模型在、hermes 能启动
```

逐项应为 ✓：nvidia-smi 出卡、torch.cuda.is_available()=True、models/ 齐、`hermes --version` 有输出。

---

## 8. 重建后如何继续研究

- 训练/评估脚本在各模块目录（english/、shared/ 等），复用 `$EASYEDIT_PY`。
- 大教师（Llama-70B-AWQ）：先 `vllm_env` 再 `start_vllm` 起服务。
- 数据：仓库只含代码+文档（大数据/模型不入库）；题目数据需用 english/00_data/ 下抽取脚本从原始源重建（PDF 教材需自备，版权原因不入库）。
