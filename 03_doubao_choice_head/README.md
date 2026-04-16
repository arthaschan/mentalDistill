# Doubao → Qwen2.5-7B Choice-Head 两阶段蒸馏

## 实验目标

该目录用于复现 Doubao 到 Qwen2.5-7B 的两阶段蒸馏方案。Doubao 的关键点不在硬标签本身，而在于多投票构造出来的软标签分布。

## 历史结果

- 教师测试准确率：98.80%（82/83）
- 教师训练集覆盖率：89.88%（604/672）
- 教师-GT 不一致率：0.33%（2/604）
- 学生峰值准确率：80.72%（单点）
- 稳定可复现结果：79.52%

## 目录内容

- `data/train.jsonl`: 训练集
- `data/val.jsonl`: 验证集
- `data/test.jsonl`: 测试集
- `data/teacher_train_soft.jsonl`: 已验证可用的历史多投票软标签
- `configs/grid_params_best.json`: 历史稳定参数
- `configs/teacher_candidate.json`: Doubao API 配置
- `scripts/run_teacher_eval.sh`: Doubao 教师测试集评测入口
- `scripts/generate_teacher_labels.sh`: 教师硬标签生成
- `scripts/generate_teacher_soft_labels.sh`: 多投票软标签生成
- `scripts/build_head_dataset.sh`: 蒸馏数据构造
- `scripts/run_train.sh`: 两阶段训练入口
- `scripts/run_eval.sh`: 评估入口
- `scripts/run_pipeline.sh`: 完整串行流程入口
- `scripts/start_web.sh`: 本地 Web 验证页入口
- `scripts/start_best_web.sh`: 直接加载默认最佳 stage2 adapter 的快捷入口

## 迁移前准备

- 准备 Python 环境
- 准备 7B 基座模型，推荐放在 `../models/Qwen2.5-7B-Instruct`
- 配置 Doubao API key 与 teacher candidate

## 最优稳定参数

- `learning_rate_stage1=1.2e-4`
- `learning_rate_stage2=1.2e-4`
- `num_epochs_stage1=1`
- `num_epochs_stage2=2`
- `batch_size=2`
- `gradient_accumulation_steps=4`
- `rank=16`
- `lora_alpha=32`
- `alpha_stage1=0.35`
- `seed=42`

## 执行步骤

1. 单独评测 Doubao 教师

```bash
bash scripts/run_teacher_eval.sh
```

2. 生成教师硬标签

```bash
bash scripts/generate_teacher_labels.sh
```

3. 生成多投票软标签

```bash
bash scripts/generate_teacher_soft_labels.sh
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
- 夜间调度执行：设置 `NIGHT_MODE=1`，尤其建议用于多投票软标签生成

## 输出产物

- `runs/`: 两阶段训练结果
- `logs/`: API 调用与训练日志
- `logs/teacher_eval_report.json`: 教师测试集准确率摘要
- `data/teacher_train_soft.generated.jsonl`: 从零重建时新生成的多投票软标签

## 复现注意事项

- 当前目录已放入历史上验证可用的 soft label 数据
- 如果你只想复现训练主链，可以直接从已有 `data/teacher_train_soft.jsonl` 开始
- 如果你要完全从零复现，则应先重跑 `generate_teacher_labels.sh` 与 `generate_teacher_soft_labels.sh`

## 模型准备（GitHub 不含模型文件）

本仓库上传到 GitHub 时 **不包含模型权重**。克隆代码后需自行下载：

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

在 `setup.env` 中配置：
```bash
export BASE_MODEL_7B="/your/path/to/models/Qwen2.5-7B-Instruct"
```

还需配置豆包 API key（`configs/teacher_candidate.json` 中的 endpoint）。

> **注意**：教师标签（`data/teacher_*.jsonl`）和训练产物（`runs/`）不上传到 GitHub。

## 测试集评估

```bash
bash scripts/run_eval.sh
```

## 人机交互测试

```bash
bash scripts/start_quiz.sh
# 默认地址 http://0.0.0.0:7870
```
