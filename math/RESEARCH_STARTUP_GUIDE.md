# 🚀 数学 AI 研究启动指南

**文档日期**：2026-06-14  
**研究项目**：Qwen 2.5-14B 几何空间结构分析与优化  
**硬件环境**：H100 GPU（直接本地使用）  
**参考文档**：[ai数学研究.md](./ai数学研究.md)

---

## 📋 目录

- [第 0 步：准备 API Key](#第-0-步准备-api-key)
- [第 1 步：在 H100 上安装依赖](#第-1-步在-h100-上安装依赖)
- [第 2 步：配置 setup.env](#第-2-步配置-setupenv)
- [第 3 步：下载 Qwen 2.5-14B 模型](#第-3-步下载-qwen-25-14b-模型)
- [第 4 步：启动 aider 开始开发](#第-4-步启动-aider-开始开发)
- [第 5 步：运行第一个实验](#第-5-步运行第一个实验)
- [第 6 步：迭代优化](#第-6-步迭代优化)
- [快速参考](#快速参考)
- [监控与故障排除](#监控与故障排除)

---

## 第 0 步：准备 API Key

### 1️⃣ Hugging Face Token（用于下载 Qwen 模型）

```bash
# 步骤：
# 1. 访问 https://huggingface.co/
# 2. 注册账号（GitHub/Google/邮箱均可）
# 3. 进入 Settings → Access Tokens
# 4. 点击 "New token" → 勾选 read 权限
# 5. 复制 token

# 在 H100 上配置（一次性）
export HF_TOKEN="hf_your_token_here"
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc

# 验证
echo $HF_TOKEN
```

### 2️⃣ OpenAI API Key（用于 aider 代码生成）

```bash
# 步骤：
# 1. 访问 https://platform.openai.com/api/keys
# 2. 登录 OpenAI 账号
# 3. Settings → API keys → Create new secret key
# 4. 复制 key

# 在 H100 上配置（一次性）
export OPENAI_API_KEY="sk-your_key_here"
echo 'export OPENAI_API_KEY="sk-your_key_here"' >> ~/.bashrc
source ~/.bashrc

# 验证
echo $OPENAI_API_KEY
```

---

## 第 1 步：在 H100 上安装依赖

```bash
# 进入项目目录
cd /home/student/arthas/mentalDistill

# 1. 安装 aider（AI 代码生成工具）
pip install aider-chat

# 2. 安装研究需要的库
pip install torch transformers accelerate gudhi matplotlib scikit-learn peft

# 3. 验证 CUDA 和 GPU 可用
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 预期输出示例：
# PyTorch version: 2.1.0
# CUDA available: True
# GPU count: 1
# GPU name: NVIDIA H100 NVL
# GPU memory: 94.5 GB
```

---

## 第 2 步：配置 setup.env

### 创建或编辑 setup.env

```bash
# 创建 setup.env（如果不存在）
cat > /home/student/arthas/mentalDistill/setup.env << 'EOF'
# ========== Python 解释器 ==========
EASYEDIT_PY="/home/student/anaconda3/bin/python"

# ========== 模型路径 ==========
# Qwen 2.5-14B 模型（下一步会下载）
BASE_MODEL_14B="/home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct"

# ========== 研究目录 ==========
RESEARCH_DIR="/home/student/arthas/mentalDistill/research"

# ========== Hugging Face 配置 ==========
HF_TOKEN="hf_your_token_here"
HF_HOME="/home/student/.cache/huggingface"

# ========== OpenAI 配置 ==========
OPENAI_API_KEY="sk-your_key_here"

# ========== 日志与输出 ==========
LOG_DIR="/home/student/arthas/mentalDistill/research/logs"
OUTPUT_DIR="/home/student/arthas/mentalDistill/research/outputs"

# ========== PyTorch 配置 ==========
# 使用 bfloat16 以节省显存
TORCH_DTYPE="bfloat16"
CUDA_VISIBLE_DEVICES="0"
EOF

# 验证文件
cat /home/student/arthas/mentalDistill/setup.env
```

### 创建研究目录

```bash
mkdir -p /home/student/arthas/mentalDistill/research/{logs,outputs}
```

---

## 第 3 步：下载 Qwen 2.5-14B 模型

这一步需要约 15-30 分钟，具体时间取决于网络速度。

```bash
# 登录 Hugging Face
huggingface-cli login
# 粘贴你的 HF_TOKEN，然后按 Enter

# 下载模型到本地（约 28GB）
huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
  --local-dir /home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct

# 验证下载完成
ls -lh /home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct/
# 应该能看到 config.json、model-*.safetensors 等文件
```

---

## 第 4 步：启动 aider 开始开发

aider 是一个 AI 驱动的命令行编程助手。它会根据你的描述自动生成代码。

### 启动 aider

```bash
# 进入项目
cd /home/student/arthas/mentalDistill

# 加载环境变量
source setup.env

# 启动 aider（指向 research 目录）
aider research/
```

### 在 aider 中生成第一个实验脚本

```bash
# 以下命令在 aider 交互式界面中输入

# 1. 添加研究文档作为上下文
/add math/ai数学研究.md

# 2. 生成 TDA 拓扑分析脚本
> Based on Section 2.1 (思路一：TDA 拓扑分析) in the math document, 
> implement `research/tda_topology_analysis.py` with these features:
> 
> 1. Load Qwen 2.5-14B from BASE_MODEL_14B environment variable
> 2. Use bfloat16 dtype to fit in H100 memory
> 3. Hook into layers 12, 24, 36, 47 to extract hidden states
> 4. Compute persistent homology using gudhi (limit to 100 points per layer)
> 5. Generate persistence diagram showing H0, H1, H2 features
> 6. Detect topology collapse (short-lived features indicate pruning candidates)
> 7. Save results to outputs/tda_results.json
> 8. Save plots to outputs/tda_topology_collapse_detect.png
> 9. Add CLI arguments: --model-path, --layers, --output-dir, --input-text
> 10. Include error handling and logging

# 3. aider 会生成完整脚本，按 y 确认应用
```

### aider 的常用命令

```bash
# 在 aider 交互式界面中

/add <file>          # 把文件加入上下文
/remove <file>       # 移除文件
/help                # 查看帮助
/exit                # 退出 aider

# 示例
/add math/ai数学研究.md
/add shared/fisher_rao_analysis.py
> Your task description here
```

---

## 第 5 步：运行第一个实验

### 运行 TDA 拓扑分析（思路一）

```bash
# 加载环境变量
source /home/student/arthas/mentalDistill/setup.env

# 运行脚本
python research/tda_topology_analysis.py \
  --model-path "$BASE_MODEL_14B" \
  --layers 12,24,36,47 \
  --output-dir research/outputs \
  --input-text "如果把一个正方体的表面涂成红色，然后切成27个大小相同的小正方体。那么有且仅有2面是红色的小正方体有多少个？请一步步推导。"

# 预期输出：
# - research/outputs/tda_results.json （拓扑分析数据）
# - research/outputs/tda_topology_collapse_detect.png （拓扑图表）
# - research/logs/tda_experiment.log （详细日志）
```

### 实时查看日志

```bash
# 在另一个终端中
tail -f research/logs/tda_experiment.log
```

### 查看结果

```bash
# 查看输出文件
ls -lh research/outputs/

# 查看 JSON 结果（简单统计）
cat research/outputs/tda_results.json | python -m json.tool | head -50

# 查看图表（需要本地显示）
# 通过 VS Code Remote Tunnels 浏览 research/outputs/tda_topology_collapse_detect.png
```

---

## 第 6 步：迭代优化

实验完成后，回到 aider 生成更多分析脚本。

### 生成 Fisher 有效秩分析（思路二）

```bash
# 重新启动 aider
aider research/

# 在 aider 中
/add math/ai数学研究.md
/add shared/fisher_rao_analysis.py

> Based on Section 2.2 (思路二：信息几何) in the math document,
> implement `research/fisher_effective_rank.py` that:
> 
> 1. Compute Fisher information matrix diagonal for all 48 layers
> 2. Focus on q_proj, k_proj, v_proj, o_proj in attention mechanism
> 3. Calculate effective rank using entropy formula from the document
> 4. Rank all layers by effective rank
> 5. Output a CSV table: layer_id | effective_rank | recommended_lora_rank
> 6. Visualize with bar charts
> 7. Can be integrated with existing Module 17/20 LoRA training
> 
> Handle memory efficiently (Qwen has 48 layers × 4 projections = 192 matrices)
```

### 生成各向异性分析（思路三）

```bash
# 在 aider 中
> Based on Section 2.3 (思路三：群论对称性) in the math document,
> implement `research/anisotropy_analysis.py` that:
> 
> 1. Fix the bug in lines 187-188 of the math document
> 2. Test anisotropy scores in layers 20-48
> 3. Compute cosine similarity matrix for normalized hidden states
> 4. Calculate anisotropy score (0=isotropic, 1=collapsed)
> 5. Identify layers with score > 0.85 (severe collapse)
> 6. Suggest orthogonal re-centering for affected layers
> 7. Generate visualizations showing anisotropy progression across layers
```

---

## 快速参考

### 一键启动流程（今天到明天）

```bash
# ===== 第一天（配置 + 模型下载，约 1 小时）=====

# 1. 配置 API Keys（按上面的指引获取）
export HF_TOKEN="hf_..."
export OPENAI_API_KEY="sk-..."

# 2. 安装依赖
pip install aider-chat torch transformers accelerate gudhi matplotlib scikit-learn peft

# 3. 创建 setup.env 和目录结构
mkdir -p research/{logs,outputs}
cat > setup.env << 'EOF'
...（见上面第 2 步）
EOF

# 4. 下载模型（后台运行）
huggingface-cli login
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models/Qwen2.5-14B-Instruct &

# ===== 第二天（开发 + 实验，约 4-6 小时）=====

# 5. 启动 aider
aider research/

# 6. 在 aider 中生成 TDA 脚本（见第 4 步）
/add math/ai数学研究.md
> Based on Section 2.1, implement tda_topology_analysis.py...

# 7. 运行第一个实验
source setup.env
python research/tda_topology_analysis.py \
  --model-path "$BASE_MODEL_14B" \
  --layers 12,24,36,47 \
  --output-dir research/outputs \
  --input-text "数学推理文本"

# 8. 查看结果
ls -lh research/outputs/
# 期望看到 tda_results.json 和 tda_topology_collapse_detect.png
```

### 文件位置速查

| 项目 | 位置 |
|------|------|
| **环境变量** | `~/.bashrc` 或 `~/.zshrc` |
| **项目配置** | `/home/student/arthas/mentalDistill/setup.env` |
| **Qwen 模型** | `/home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct/` |
| **研究代码** | `/home/student/arthas/mentalDistill/research/*.py` |
| **实验结果** | `/home/student/arthas/mentalDistill/research/outputs/` |
| **实验日志** | `/home/student/arthas/mentalDistill/research/logs/` |
| **研究文档** | `/home/student/arthas/mentalDistill/math/ai数学研究.md` |

---

## 监控与故障排除

### 实时监控 GPU 使用

```bash
# 在 H100 上另开一个终端
watch -n 1 nvidia-smi

# 查看详细显存使用
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

# 限定仅查看 GPU 0
nvidia-smi -i 0 -l 1
```

### 常见问题与解决方案

#### Q: 导入 gudhi 出错

```bash
# 症状：ImportError: cannot import name 'RipsComplex' from gudhi

# 解决：重新安装
pip uninstall gudhi -y
pip install gudhi
```

#### Q: 模型加载 OOM（显存溢出）

```bash
# 症状：CUDA out of memory

# 检查：H100 应该有 80GB 或 95GB 显存，足够装 14B@bfloat16

# 解决：
# 1. 确保使用 bfloat16
torch_dtype=torch.bfloat16

# 2. 使用 device_map="auto"（自动分配）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

#### Q: aider 生成的代码有 bug

```bash
# 在 aider 中告诉它问题
> Line 42 raises KeyError. Fix by adding proper error handling.
> The hook output format is (batch, seq_len, hidden_dim), not just hidden_dim.

# aider 会修复并重新生成
```

#### Q: Hugging Face 下载速度慢

```bash
# 解决：
# 1. 换个时间段（避免高峰）
# 2. 使用镜像源（如 mirrors.tsinghua）
export HF_ENDPOINT="https://huggingface.co"  # 默认官方
# 或国内镜像（如有）

# 3. 直接在云存储上下载后转移
```

---

## 后续推进

### 一周实验计划

| 时间 | 任务 | 预期产出 |
|------|------|---------|
| **Day 1** | 环境配置 + 模型下载 | setup.env, 28GB 模型文件 |
| **Day 2-3** | 思路一（TDA）| tda_topology_collapse_detect.png，识别平庸层 |
| **Day 4-5** | 思路二（Fisher）| fisher_effective_rank.csv，LoRA 秩分配建议 |
| **Day 6-7** | 思路三（各向异性）| anisotropy_scores.json，正交校准建议 |
| **Day 8+** | 论文撰写 + 优化 | 扩展 AIEA 论文，集成到 mentalDistill |

### 与现有项目的联系

- **Module 18**（Fisher-Rao 分析）：已有标签空间的几何分析
- **Module 17/20**（LoRA 蒸馏）：可直接对接 Fisher 有效秩的秩分配
- **AIEA 论文**：新增「表征空间与参数空间的几何分析」章节

---

## 技术支持

如果遇到问题，可以：

1. 重新启动 aider，让它根据错误信息修复代码
2. 查看详细日志：`tail -f research/logs/tda_experiment.log`
3. 逐步调试：修改脚本中的参数，重新运行
4. 参考原始文档：[ai数学研究.md](./ai数学研究.md)

---

## 最后一步：开始

```bash
# 当你准备好时，运行这行命令即可开始
aider /home/student/arthas/mentalDistill/research/
```

祝你的 AI 数学研究顺利！🎉
