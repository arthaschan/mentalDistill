# DeepSeek-V3 → Qwen2.5-7B Choice-Head 两阶段蒸馏

## 实验目标

该目录用于复现 DeepSeek-V3 到 Qwen2.5-7B 的两阶段蒸馏方案：先做 Choice-Head KL 蒸馏，再用 GT 做 Stage2 SFT 续训。

## 历史结果

- 教师测试准确率：87.95%（73/83）
- 教师训练集覆盖率：100%（672/672）
- 教师-GT 不一致率：14.14%（95/672）
- 学生峰值测试准确率：81.93%（seed=11）
- 稳定可复现结果：79.52%

## 目录内容

- `data/train.jsonl`: 训练集
- `data/val.jsonl`: 验证集
- `data/test.jsonl`: 测试集
- `data/teacher_train.jsonl`: 教师硬标签
- `data/teacher_train_soft.jsonl`: 平滑软标签
- `data/train_head_distill.jsonl`: Choice-Head 训练数据
- `configs/grid_params_best.json`: 历史峰值参数
- `configs/teacher_candidate.json`: API 教师配置
- `scripts/run_teacher_eval.sh`: DeepSeek 教师测试集评测入口
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
- 配置 API key 与 teacher candidate

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
- `best_seed=11`

## 执行步骤

1. 单独评测 DeepSeek 教师

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

- 白天手动执行：默认不主动放慢 API 请求
- 夜间调度执行：设置 `NIGHT_MODE=1`，脚本会启用慢速和冷却策略

## 输出产物

- `runs/`: 两阶段训练完整运行目录
- `logs/`: 标签生成与训练日志
- `logs/teacher_eval_report.json`: 教师测试集准确率摘要
- `data/train_head_distill.jsonl`: 蒸馏数据集

## 复现注意事项

- DeepSeek 的峰值结果对 seed 较敏感
- 如果 API 额度紧张，优先使用夜间调度入口
