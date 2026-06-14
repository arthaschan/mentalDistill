# Codex 统筹与 H100 直接调用指南

**目标**：用 Codex（GPT-4.5）作为宏观统筹者，直接从 H100 调用外部 AI 模型（Gauss、DeepSeek 等）和本地 GPU 计算。

---

## 架构

```
┌──────────────────────────────────────────────────┐
│ Codex/GPT-4.5（H100 上直接调用）                  │
│ - 调用决策：用哪个模型、什么参数、怎样综合结果    │
│ - 编排流程：TDA → Fisher → Anisotropy → 报告     │
└──────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
    ┌────────────┐    ┌──────────────┐    ┌─────────┐
    │   Gauss    │    │  DeepSeek    │    │ H100 GPU│
    │(数学求解)  │    │  (推理决策)  │    │(拓扑/秩)│
    └────────────┘    └──────────────┘    └─────────┘
```

**关键点**：H100 能直接访问外网，无需 Mac 网关。

---

## Step 0：获取 OpenAI API Key

1. 访问 https://platform.openai.com/api/keys
2. 创建 New Secret Key（GPT-4.5 或 gpt-4-turbo）
3. 复制 Key，保妥善保管

---

## Step 1：H100 环境配置

### 登录 H100

```bash
# 从 Mac 通过 VS Code tunnel 连接（已有）
# 或直接 SSH
ssh student@h100.local
```

### 配置环境变量

```bash
# 编辑 ~/.bashrc 或 ~/.zshrc
export OPENAI_API_KEY="sk-proj-xxx..."
export OPENAI_MODEL="gpt-4.5"  # 或 gpt-4-turbo
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选，默认已正确

# 加载配置
source ~/.bashrc
```

或创建 H100 项目级配置：

```bash
# 在 /home/student/arthas/mentalDistill/ 下创建 setup.env
cat > setup.env << 'EOF'
export OPENAI_API_KEY="sk-proj-xxx..."
export OPENAI_MODEL="gpt-4.5"
export BASE_MODEL_14B="/home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct"
export TORCH_DTYPE="bfloat16"
export RESEARCH_DIR="/home/student/arthas/mentalDistill/research"
EOF

# 使用时
source setup.env
```

### 验证 API 连接

```bash
python3 << 'EOF'
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-4.5",
    messages=[{"role": "user", "content": "Hello, test message."}],
    max_tokens=10
)

print("✓ API 连接成功")
print(f"回复：{response.choices[0].message.content}")
EOF
```

---

## Step 2：创建本地辅助脚本（用 GPT 生成）

### 在 ChatGPT/Claude 中生成脚本

在 **ChatGPT（GPT-4.5）** 或 **Claude** 中输入以下 prompt：

```
我需要编写 3 个 Python 脚本，用于分析 Qwen2.5-14B 的几何结构：

1. research/codex_client.py
   - 与 Codex/GPT-4.5 通信的客户端
   - 接收数据，返回决策（字符串形式的 Python 命令）

2. research/local_analyzer.py
   - 计算本地 GPU 上的 TDA 拓扑特征（使用 Gudhi）
   - 计算 Fisher 信息矩阵秩
   - 计算各向异性指标
   - 返回 dict 格式结果

3. research/gauss_integration.py
   - 调用 Gauss API（或命令行）求解特征值、秩等
   - 接收 numpy 矩阵，返回结果

要求：
- 所有代码都要支持 bfloat16
- 使用环境变量读取配置
- 返回结果格式化为 JSON
- 带详细日志

请生成这三个脚本的完整代码。
```

复制生成的脚本到 H100 的 `/home/student/arthas/mentalDistill/research/` 目录。

---

## Step 3：编写编排脚本（由你或 Codex 生成）

### 核心编排逻辑

```python
# research/codex_orchestrator.py
"""
Codex 驱动的实验编排脚本
"""
import os
import json
import openai
from local_analyzer import LocalAnalyzer
from gauss_integration import GaussClient

class CodexOrchestrator:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.5")
        self.analyzer = LocalAnalyzer()
        self.gauss = GaussClient()
        openai.api_key = self.openai_key
    
    def request_next_action(self, context: dict) -> str:
        """
        向 Codex 请求下一步行动
        
        Args:
            context: 当前实验状态 {"tda_results": {...}, "fisher_results": {...}}
        
        Returns:
            Codex 的决策字符串（如 "run_anisotropy" 或 "generate_report"）
        """
        prompt = f"""
        你是一个几何分析的统筹者。根据目前的分析结果，决定下一步行动。
        
        当前分析结果：
        {json.dumps(context, indent=2)}
        
        可选行动：
        1. "run_anisotropy" - 计算各向异性指标（如果还没做）
        2. "call_gauss_rank" - 调用 Gauss 计算 Fisher 秩
        3. "generate_pruning_strategy" - 生成修剪建议
        4. "generate_report" - 生成最终报告
        
        返回单个行动名称，不要其他文字。
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=20
        )
        
        action = response.choices[0].message.content.strip()
        return action
    
    def run_full_pipeline(self):
        """执行完整实验流程"""
        context = {}
        
        # Step 1: TDA 拓扑分析
        print("[1/4] 运行 TDA 拓扑分析...")
        context["tda_results"] = self.analyzer.run_tda()
        print(f"✓ TDA 完成，检测到 {len(context['tda_results']['collapsed_layers'])} 个坍缺层")
        
        # Step 2: 询问 Codex 下一步
        action = self.request_next_action(context)
        print(f"[Codex 决策] 下一步：{action}")
        
        # Step 3: Fisher 秩分析
        if action in ["call_gauss_rank", "generate_pruning_strategy"]:
            print("[2/4] 运行 Fisher 有效秩分析...")
            context["fisher_results"] = self.analyzer.run_fisher_rank()
            print(f"✓ Fisher 完成")
        
        # Step 4: 再次询问 Codex
        action = self.request_next_action(context)
        print(f"[Codex 决策] 下一步：{action}")
        
        # Step 5: 生成修剪策略
        if action == "generate_pruning_strategy":
            print("[3/4] 生成修剪策略...")
            strategy = self._codex_generate_strategy(context)
            context["pruning_strategy"] = strategy
            print(f"✓ 修剪策略：{strategy}")
        
        # Step 6: 生成最终报告
        print("[4/4] 生成最终报告...")
        report = self._codex_generate_report(context)
        
        # 保存结果
        output_file = f"{os.getenv('RESEARCH_DIR', '.')}/outputs/orchestration_report.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(context, f, indent=2)
        
        print(f"\n✓ 完成！结果保存到 {output_file}")
        print(f"\n最终报告：\n{report}")
    
    def _codex_generate_strategy(self, context: dict) -> str:
        """让 Codex 生成修剪策略"""
        prompt = f"""
        根据以下分析结果，为 Qwen2.5-14B 的修剪生成策略：
        
        TDA 检测结果：{context.get('tda_results', {})}
        Fisher 秩结果：{context.get('fisher_results', {})}
        
        返回 JSON 格式的修剪建议：
        {{
            "layers_to_prune": [列表],
            "lora_ranks": {{"layer_i": rank_value, ...}},
            "reasoning": "解释这个策略的原因"
        }}
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _codex_generate_report(self, context: dict) -> str:
        """让 Codex 生成最终报告"""
        prompt = f"""
        基于以下几何分析结果，生成一份简明的研究报告（200 字以内）：
        
        {json.dumps(context, indent=2)}
        
        包含：
        1. 主要发现
        2. 几何特征
        3. 修剪建议
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content


if __name__ == "__main__":
    orchestrator = CodexOrchestrator()
    orchestrator.run_full_pipeline()
```

---

## Step 4：运行编排脚本

```bash
# 进入项目目录
cd /home/student/arthas/mentalDistill

# 加载环境变量
source setup.env

# 运行编排脚本
python research/codex_orchestrator.py

# 监控 GPU
nvidia-smi  # 另一个终端
```

---

## Step 5：查看结果

```bash
# 查看输出
cat research/outputs/orchestration_report.json | jq .

# 查看日志
tail -f research/logs/orchestration.log
```

---

## 关键文件清单

| 文件 | 作用 | 来源 |
|------|------|------|
| `setup.env` | 环境变量配置 | 手动创建 |
| `research/codex_client.py` | Codex API 包装 | GPT 生成 |
| `research/local_analyzer.py` | 本地 GPU 分析 | GPT 生成 |
| `research/gauss_integration.py` | Gauss 集成层 | GPT 生成 |
| `research/codex_orchestrator.py` | 主编排脚本 | 本指南或 GPT 生成 |
| `math/ai数学研究.md` | 研究方案参考 | 已有 |

---

## 故障排除

### 问题 1：OPENAI_API_KEY 不被识别

```bash
# 验证环境变量
echo $OPENAI_API_KEY

# 如果为空，手动设置
export OPENAI_API_KEY="sk-proj-xxx..."

# 或写入 ~/.bashrc
echo 'export OPENAI_API_KEY="sk-proj-xxx..."' >> ~/.bashrc
source ~/.bashrc
```

### 问题 2：API 连接超时

```bash
# 检查网络连接
curl -I https://api.openai.com/v1/models

# 如果超时，可能 H100 无法出网（改用方案 2）
```

### 问题 3：显存不足

```bash
# 检查 GPU 状态
nvidia-smi

# 使用 bfloat16 并减少 batch size
export TORCH_DTYPE="bfloat16"
```

---

## 下一步

1. 从 https://platform.openai.com/api/keys 获取 OPENAI_API_KEY
2. 在 H100 上配置 `setup.env`
3. 在 ChatGPT/Claude 中生成三个辅助脚本
4. 复制脚本到 `research/`
5. 运行 `python research/codex_orchestrator.py`
6. 查看结果并迭代

---

## 参考

- OpenAI API 文档：https://platform.openai.com/docs/api-reference
- 研究方案：[ai数学研究.md](ai数学研究.md)
- Qwen 模型：https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
