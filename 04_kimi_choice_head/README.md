# Kimi → Qwen2.5-7B Choice-Head 两阶段蒸馏

## 实验目标

该目录用于复现 Kimi（moonshot-v1-32k）到 Qwen2.5-7B 的两阶段蒸馏流程。它的价值不在于性能提升，而在于保留一个“弱教师无效”的负例实验。

## 历史结果

- 教师测试准确率：61.45%（51/83）
- 教师训练集覆盖率：100%（672/672）
- 教师-GT 不一致率：0%
- 学生最佳测试准确率：77.11%（仅追平基线）
- 结论：弱教师 + 零不一致 = 无蒸馏增益

## 目录内容

- `data/train.jsonl`: 训练集
- `data/val.jsonl`: 验证集
- `data/test.jsonl`: 测试集
- `data/teacher_train.jsonl`: 教师硬标签
- `configs/grid_params_best.json`: 历史关键配置
- `configs/teacher_candidate.json`: Kimi API 配置
- `scripts/run_teacher_eval.sh`: Kimi 教师测试集评测入口
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
- 准备 7B 基座模型，推荐放在 `../models/Qwen2.5-7B-Instruct`
- 配置 Kimi API key 与 teacher candidate

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
- `seed=42`

## 执行步骤

1. 单独评测 Kimi 教师

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

## 白天 / 夜间模式

- 白天手动执行：默认不主动限速
- 夜间调度执行：设置 `NIGHT_MODE=1`

## 输出产物

- `runs/`: 两阶段训练输出
- `logs/`: API 与训练日志
- `logs/teacher_eval_report.json`: 教师测试集准确率摘要

## 复现注意事项

- 该目录是重要的负样本对照实验，不应因为结果不好就省略
- 迁移后建议保留它，用于验证“教师强度确实影响蒸馏收益”这一结论
