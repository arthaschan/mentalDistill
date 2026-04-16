# 09_qwen7b_selfdistill — Qwen2.5-7B 自蒸馏

## 实验目标

让 Qwen2.5-7B 作为自身的教师进行自蒸馏，验证"自己教自己"是否有效。

## 背景

自蒸馏是一种不依赖外部教师的知识蒸馏方法。模型先生成教师标签（对自己的输出取 argmax 或 soft distribution），然后在这些标签上训练。这构成一个无外部依赖的基线。

## 目录内容

- `data/train.jsonl`: 训练集
- `data/val.jsonl`: 验证集
- `data/test.jsonl`: 测试集
- `configs/grid_params_best.json`: 训练配置
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

> **注意**：教师标签、训练产物等不会上传到 GitHub，需本地重新生成。

## 执行步骤

1. 生成自蒸馏教师标签：`bash scripts/generate_teacher_labels.sh`
2. 构造蒸馏数据：`bash scripts/build_head_dataset.sh`
3. 训练：`bash scripts/run_train.sh`
4. 评估：`bash scripts/run_eval.sh`
5. 一键执行：`bash scripts/run_pipeline.sh`

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
