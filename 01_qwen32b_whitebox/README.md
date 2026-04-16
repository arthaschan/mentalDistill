# Qwen2.5-32B → Qwen2.5-7B 白盒蒸馏

## 实验目标

该目录用于复现 Qwen2.5-32B-Instruct 到 Qwen2.5-7B-Instruct 的白盒蒸馏。该方案直接使用教师 logits 做 KL 蒸馏，是当前 `rebuild/` 中唯一需要本地同时加载 teacher 和 student 的方案。

## 历史结果

- 教师测试准确率：83.13%（69/83）
- 学生最佳测试准确率：80.72%
- 方案性质：白盒 Logit KL Distillation
- 特点：显存压力大，需要同时容纳 32B teacher 和 7B student

## 目录内容

- `data/train.jsonl`: 训练集
- `data/val.jsonl`: 验证集
- `data/test.jsonl`: 测试集
- `configs/train_config.json`: 历史最佳参数
- `scripts/run_teacher_eval.sh`: 32B 教师测试集评测入口
- `scripts/run_train.sh`: 训练入口
- `scripts/run_eval.sh`: 评估入口
- `scripts/run_pipeline.sh`: 训练+评估串行入口
- `scripts/start_web.sh`: 本地 Web 验证页入口
- `scripts/start_best_web.sh`: 直接加载 `outputs/best` 的快捷入口

## 迁移前准备

- 准备 Python 环境，并设置 `EASYEDIT_PY` 或使用自动探测
- 准备 7B 与 32B 基座模型
- 推荐放置位置：
   - `../models/Qwen2.5-7B-Instruct`
   - `../models/Qwen2.5-32B-Instruct`
- 或使用环境变量：
   - `BASE_MODEL_7B`
   - `BASE_MODEL_32B`

## 最优参数

- `learning_rate=1e-4`
- `num_epochs=4`
- `batch_size=1`
- `gradient_accumulation_steps=8`
- `rank=16`
- `lora_alpha=32`
- `temperature=2.0`
- `alpha=0.1`
- `alpha_warmup_epochs=1`
- `hard_upsample=2`
- `seed=2`

## 执行步骤

1. 评测 32B 教师

```bash
bash scripts/run_teacher_eval.sh
```

2. 训练

```bash
bash scripts/run_train.sh
```

3. 评估

```bash
bash scripts/run_eval.sh
```

4. 一键串行执行训练与评估

```bash
bash scripts/run_pipeline.sh
```

5. 启动本地验证页

```bash
bash scripts/start_web.sh
```

6. 直接加载最佳 adapter 启动验证页

```bash
bash scripts/start_best_web.sh
```

## 输出产物

- `outputs/`: LoRA adapter 输出目录
- `outputs/test_wrong.jsonl`: 测试集错误样本
- `logs/train.log`: 训练日志
- `logs/teacher_eval_report.json`: 32B 教师测试集准确率摘要

## 复现注意事项

- 该方案对 GPU 显存最敏感
- 迁移到独立项目后，优先验证模型路径和显存是否足够

## 模型准备（GitHub 不含模型文件）

本仓库上传到 GitHub 时 **不包含模型权重**。克隆代码后需自行下载：

```bash
# 下载 7B 学生模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct

# 下载 32B 教师模型
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir models/Qwen2.5-32B-Instruct
```

在 `setup.env` 中配置：
```bash
export BASE_MODEL_7B="/your/path/to/models/Qwen2.5-7B-Instruct"
export BASE_MODEL_32B="/your/path/to/models/Qwen2.5-32B-Instruct"
```

> **显存需求**：白盒蒸馏需同时加载 32B 教师 + 7B 学生，峰值约 90GB，需 H100 95GB 级别 GPU。

## 测试集评估

```bash
bash scripts/run_eval.sh
```

## 人机交互测试

```bash
bash scripts/start_quiz.sh
# 默认地址 http://0.0.0.0:7870
```
