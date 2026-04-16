# Qwen2.5-32B API → Qwen2.5-7B Choice-Head 蒸馏

## 实验目标

该目录用于通过 API 调用 Qwen2.5-32B（而非本地白盒），获取选项级概率分布进行黑盒 Choice-Head 蒸馏到 Qwen2.5-7B。与 Module 01 白盒方案互为对比。

## 历史结果

- 教师测试准确率：83.13%（69/83）
- 方案性质：Qwen2.5-32B 黑盒 API + Choice-Head 两阶段蒸馏
- 状态：配置就绪，可直接执行

## 目录内容

- `data/train.jsonl`: 训练集
- `data/val.jsonl`: 验证集
- `data/test.jsonl`: 测试集
- `data/teacher_train.jsonl`: 教师硬标签
- `data/teacher_test.jsonl`: 教师测试集标签
- `configs/grid_params_best.json`: 训练配置
- `configs/teacher_candidate.json`: 32B API 配置
- `scripts/`: 完整流程脚本

## 模型准备（GitHub 不含模型文件）

本仓库上传到 GitHub 时 **不包含模型权重**。克隆代码后需自行下载：

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

在 `setup.env` 中配置：
```bash
export BASE_MODEL_7B="/your/path/to/models/Qwen2.5-7B-Instruct"
```

还需配置 Qwen2.5-32B API endpoint（`configs/teacher_candidate.json`）。

> **注意**：教师标签、训练产物、日志等不会上传到 GitHub。

## 执行步骤

1. 评测 32B API 教师：`bash scripts/run_teacher_eval.sh`
2. 生成教师标签：`bash scripts/generate_teacher_labels.sh`
3. 生成软标签：`bash scripts/prepare_soft_labels.sh`
4. 构造蒸馏数据：`bash scripts/build_head_dataset.sh`
5. 训练：`bash scripts/run_train.sh`
6. 评估：`bash scripts/run_eval.sh`
7. 一键流程：`bash scripts/run_pipeline.sh`

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
