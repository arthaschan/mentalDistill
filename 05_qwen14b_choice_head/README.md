# Qwen2.5-14B → Qwen2.5-7B Choice-Head 两阶段蒸馏

## 实验目标

该目录用于复现本地 Qwen2.5-14B 到 Qwen2.5-7B 的两阶段蒸馏流程。它保留了一个重要负结果：教师与学生能力接近时，蒸馏可能产生负收益。

## 历史结果

- 教师测试准确率：77.11%（64/83）
- 教师训练集准确率：79.76%（536/672）
- 教师-GT 不一致率：20.24%（136/672）
- 学生最佳测试准确率：75.90%
- 结论：同等能力教师 + 高错误率 = 负蒸馏效应

## 目录内容

- `data/train.jsonl`: 训练集
- `data/val.jsonl`: 验证集
- `data/test.jsonl`: 测试集
- `data/teacher_train.jsonl`: 本地 14B 生成的训练集教师标签
- `data/teacher_test.jsonl`: 本地 14B 生成的测试集教师标签
- `data/teacher_train_soft.jsonl`: 平滑软标签
- `data/train_head_distill.jsonl`: Choice-Head 蒸馏数据
- `data/distill_dataset_report.txt`: 数据构造报告
- `configs/grid_params_best.json`: 历史最佳配置
- `scripts/run_teacher_eval.sh`: 14B 教师测试集评测入口
- `scripts/generate_teacher_labels.sh`: 教师硬标签生成
- `scripts/prepare_soft_labels.sh`: 平滑软标签生成
- `scripts/build_head_dataset.sh`: 蒸馏数据构造
- `scripts/run_train.sh`: 两阶段训练入口
- `scripts/run_eval.sh`: 评估入口
- `scripts/run_pipeline.sh`: 完整串行流程入口
- `scripts/start_web.sh`: 本地 Web 验证页入口
- `scripts/start_best_web.sh`: 直接加载默认最佳 stage2 adapter 的快捷入口

## 迁移前准备

- 准备 Python 环境
- 准备 7B 与 14B 本地模型
- 推荐模型位置：
	- `../models/Qwen2.5-7B-Instruct`
	- `../models/Qwen2.5-14B-Instruct`
- 或使用环境变量：
	- `BASE_MODEL_7B`
	- `BASE_MODEL_14B`

## 最优参数

- `learning_rate_stage1=1.2e-4`
- `learning_rate_stage2=1.2e-4`
- `num_epochs_stage1=1`
- `num_epochs_stage2=2`
- `batch_size=2`
- `gradient_accumulation_steps=4`
- `rank=16`
- `lora_alpha=32`
- `alpha_stage1=0.35`
- `smooth_eps=0.25`
- `min_entropy=0.20`
- `min_margin=0.03`
- `best_seeds=42,55`

## 执行步骤

1. 单独评测本地 14B 教师

```bash
bash scripts/run_teacher_eval.sh
```

2. 生成教师硬标签

```bash
bash scripts/generate_teacher_labels.sh
```

3. 生成平滑软标签

```bash
bash scripts/prepare_soft_labels.sh
```

4. 构造 Choice-Head 蒸馏数据

```bash
bash scripts/build_head_dataset.sh
```

5. 两阶段训练

```bash
bash scripts/run_train.sh
```

6. 评估

```bash
bash scripts/run_eval.sh
```

7. 一键串行执行完整流程

```bash
bash scripts/run_pipeline.sh
```

8. 启动本地验证页

```bash
bash scripts/start_web.sh
```

9. 直接加载默认最佳 stage2 adapter 启动验证页

```bash
bash scripts/start_best_web.sh
```

## 输出产物

- `runs/`: 两阶段训练结果
- `logs/`: 标签生成与训练日志
- `logs/teacher_eval_report.json`: 教师测试集准确率摘要
- `data/teacher_test.jsonl`: 教师测试集推理结果

## 复现注意事项

- 该目录是一个重要反例实验
- 不要因为结果低于基线就删除；它对分析“什么样的教师会带来负蒸馏”很关键
