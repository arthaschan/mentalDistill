# Qwen2.5-7B 基线 GT SFT

## 实验目标

该目录用于复现 Qwen2.5-7B-Instruct 在牙科选择题任务上的纯 GT SFT 基线。它不依赖教师模型，是后续所有蒸馏实验的对照组。

## 历史结果

- 测试集准确率：77.11%（64/83）
- 方案性质：纯 GT SFT
- 用途：所有蒸馏结果都应先与该基线对比

## 目录内容

- `data/train.jsonl`: 训练集
- `data/val.jsonl`: 验证集
- `data/test.jsonl`: 测试集
- `configs/train_config.json`: 历史训练配置
- `scripts/run_train.sh`: 训练入口
- `scripts/run_eval.sh`: 评估入口
- `scripts/run_pipeline.sh`: 训练+评估串行入口
- `scripts/run_teacher_eval.sh`: 占位脚本，说明该目录无教师模型
- `scripts/start_web.sh`: 本地 Web 验证页入口
- `scripts/start_best_web.sh`: 直接加载 `outputs/best` 的快捷入口

## 迁移前准备

- 准备 Python 环境，并通过 `EASYEDIT_PY` 指向解释器；若不设置，则脚本会自动探测
- 准备 7B 基座模型
- 推荐把 7B 模型放在 `../models/Qwen2.5-7B-Instruct`
- 或使用环境变量 `BASE_MODEL_7B` 覆盖默认模型路径

## 最优参数

- `learning_rate=2e-4`
- `num_epochs=5`
- `batch_size=4`
- `gradient_accumulation_steps=1`
- `rank=16`
- `lora_alpha=32`
- `alpha=0.0`
- `seed=42`

## 执行步骤

1. 训练

```bash
bash scripts/run_train.sh
```

2. 评估

```bash
bash scripts/run_eval.sh
```

3. 一键串行执行训练与评估

```bash
bash scripts/run_pipeline.sh
```

4. 启动本地验证页

```bash
bash scripts/start_web.sh
```

5. 直接加载最佳 adapter 启动验证页

```bash
bash scripts/start_best_web.sh
```

## 输出产物

- `outputs/best`: 最佳 LoRA adapter
- `outputs/test_wrong.jsonl`: 测试集错误样本
- `logs/train.log`: 训练日志
- `logs/teacher_eval.log`: 教师说明日志

## 复现注意事项

- 这是最先应该复现的目录
- 如果该基线无法稳定复现，后续蒸馏结果就缺少有效参照
