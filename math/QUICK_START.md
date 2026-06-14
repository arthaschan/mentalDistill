# 快速开始：Codex 统筹 H100 研究

**你的架构**：

```
Mac (ChatGPT/Claude) → 生成脚本 → 复制到 H100
                        ↓
                   H100 (Codex 编排)
                   ├─ 调用本地 GPU (Qwen 分析)
                   ├─ 调用 Gauss (数学求解)
                   └─ 调用 DeepSeek (推理)
```

---

## 5 分钟快速开始

### 1. H100 上配置 API Key

```bash
# SSH 到 H100（或通过 VS Code tunnel）
export OPENAI_API_KEY="sk-proj-xxx..."
```

### 2. 在 ChatGPT 中生成辅助脚本

复制以下 prompt 到 **ChatGPT（选择 GPT-4.5）** 或 **Claude Opus**：

```
我需要 3 个 Python 脚本来分析 Qwen2.5-14B 的几何结构。

1. research/local_analyzer.py
   - run_tda()：用 Gudhi 计算持久同调，返回 {"collapsed_layers": [12, 15, ...]}
   - run_fisher_rank()：计算 Fisher 秩，返回 {"layer_i": rank_value}
   - run_anisotropy()：计算各向异性，返回 {"anisotropy_scores": {...}}

2. research/codex_client.py
   - 与 OpenAI API 通信
   - 支持发送分析结果，接收 Codex 决策

3. research/codex_orchestrator.py
   - 主编排脚本
   - 调用 local_analyzer 和 codex_client
   - 执行完整流程：TDA → Fisher → 策略 → 报告
   - 所有结果保存为 JSON

要求：
- 使用 bfloat16
- 支持环境变量配置
- 代码有详细注释
- 返回结果格式化为 JSON
```

### 3. 将生成的脚本复制到 H100

```bash
# 在 H100 上创建目录
mkdir -p /home/student/arthas/mentalDistill/research/outputs
mkdir -p /home/student/arthas/mentalDistill/research/logs

# 从 ChatGPT 复制脚本内容，粘贴到：
# - research/local_analyzer.py
# - research/codex_client.py
# - research/codex_orchestrator.py
```

### 4. 运行编排脚本

```bash
cd /home/student/arthas/mentalDistill

# 设置 API Key
export OPENAI_API_KEY="sk-proj-xxx..."
export OPENAI_MODEL="gpt-4.5"

# 执行
python research/codex_orchestrator.py
```

### 5. 查看结果

```bash
cat research/outputs/orchestration_report.json | jq .
```

---

## 详细指南

更多配置和故障排除：[CODEX_ORCHESTRATION_GUIDE.md](CODEX_ORCHESTRATION_GUIDE.md)

---

## 文件结构

```
mentalDistill/
├── math/
│   ├── ai数学研究.md                    # 研究方案
│   ├── CODEX_ORCHESTRATION_GUIDE.md    # 详细指南（本文件）
│   └── QUICK_START.md                  # 快速开始（你在这里）
├── research/
│   ├── local_analyzer.py               # 本地 GPU 分析（GPT 生成）
│   ├── codex_client.py                 # Codex 客户端（GPT 生成）
│   ├── codex_orchestrator.py           # 编排脚本（GPT 生成）
│   ├── outputs/                        # 实验结果
│   └── logs/                           # 日志
├── models/
│   └── Qwen2.5-14B-Instruct/           # 模型（需下载）
└── setup.env                           # 环境变量（你创建）
```

---

## 环境变量配置

创建 `setup.env`：

```bash
cat > /home/student/arthas/mentalDistill/setup.env << 'EOF'
export OPENAI_API_KEY="sk-proj-xxx..."
export OPENAI_MODEL="gpt-4.5"
export BASE_MODEL_14B="/home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct"
export TORCH_DTYPE="bfloat16"
export RESEARCH_DIR="/home/student/arthas/mentalDistill/research"
EOF

# 使用时
source setup.env
```

---

## API Key 获取

1. **OpenAI API Key**
   - 访问 https://platform.openai.com/api/keys
   - 创建 New Secret Key
   - 选择 gpt-4.5 或 gpt-4-turbo

2. **Hugging Face Token**（如需下载 Qwen）
   - 访问 https://huggingface.co/settings/tokens
   - 创建 User Access Token
   - `huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models/Qwen2.5-14B-Instruct`

---

## 常见问题

**Q：H100 能出网吗？**  
A：运行 `curl https://api.openai.com/v1/status` 测试。

**Q：Codex 返回的决策是字符串吗？**  
A：是的。编排脚本根据字符串（如 "run_anisotropy"、"generate_report"）执行不同步骤。

**Q：脚本生成后怎样迭代？**  
A：在 ChatGPT 中继续对话，描述问题，让它修复。或者直接编辑本地脚本。

---

## 下一步

✅ 配置 OPENAI_API_KEY  
✅ 在 ChatGPT 中生成 3 个脚本  
✅ 复制脚本到 `research/`  
✅ 运行 `python research/codex_orchestrator.py`  
✅ 查看 `research/outputs/orchestration_report.json`  

成功！🎉
