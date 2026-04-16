# Qwen2.5-14B 学生 + DeepSeek-V3 教师 两阶段蒸馏

## 实验目标

将学生模型从 Qwen2.5-7B 升级为 Qwen2.5-14B，利用 DeepSeek-V3 的教师标签进行两阶段蒸馏。

**核心假设**：7B 学生的天花板 (79.52%) 源于模型容量不足。14B 基线已达 83.13%，超过 7B 蒸馏天花板，蒸馏有望进一步提升至接近 DeepSeek 的 87.95%。

## 硬件需求

- GPU: NVIDIA H100 NVL 95GB
- 14B BF16 + LoRA rank=16: 约 35GB 显存，余量充足

## 训练配置

- 学生模型: Qwen2.5-14B-Instruct (via `$BASE_MODEL_14B`)
- 教师模型: DeepSeek-V3 (87.95% test acc)
- 两阶段蒸馏: Stage1 choice-head (α=0.35, 1 epoch) → Stage2 GT SFT (5 epoch)
- LoRA: rank=16, lora_alpha=32
- 3 seeds: 42, 11, 8
- Batch: 2 × 4 grad_accum = effective 8

## 数据来源

教师标签复用自 `02_deepseek_v3_choice_head/data/`，无需重新 API 调用。

## 执行

```bash
source setup.env
bash 13_14b_deepseek_distill/scripts/run_pipeline.sh
```

## 目录内容

- `configs/teacher_candidate.json`: DeepSeek API 配置
- `configs/grid_params_best.json`: 训练超参 (3 seeds)
- `data/`: 训练/验证/测试数据 + 教师标签
- `scripts/`: 完整流程脚本
- `runs/`: 训练产物 (gitignored)
- `outputs/`: 评估结果

## 模型准备（GitHub 不含模型文件）

本仓库上传到 GitHub 时 **不包含模型权重**。克隆代码后需自行下载：

```bash
# 下载 14B 学生模型
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models/Qwen2.5-14B-Instruct
```

在 `setup.env` 中配置：
```bash
export BASE_MODEL_14B="/your/path/to/models/Qwen2.5-14B-Instruct"
```

教师标签复用 Module 02，还需配置：
```bash
export DEEPSEEK_API_KEY="your-api-key"  # 如需重新生成教师标签
```

> **注意**：训练产物（`runs/`）、评估结果（`outputs/`）不上传 GitHub。

## 测试集评估

```bash
bash scripts/run_eval.sh
```

## 人机交互测试

```bash
bash scripts/start_quiz.sh
# 默认地址 http://0.0.0.0:7870
```

## 启动模型推理 Web 界面

```bash
bash scripts/start_web.sh
```
