# 08_step3_consistency_filter — Stage 3: 真实多票软标签融合 + 自一致性过滤

## 目标

用 API 多次采样（N=9）获取教师的真实概率分布，替代之前的 smooth_eps 假软标签。
结合自一致性过滤，排除教师自身不确定的样本，提升蒸馏质量。

## 背景

Stage 2 发现：所有 "软标签" 均为 smooth_eps 生成的假分布（0.80/0.05），
没有真正的 dark knowledge。多教师融合上限 79.52%，低于单教师最佳 81.93%。

## 数据来源

| 教师 | 准确率 | 权重 | 多票文件 |
|------|--------|------|----------|
| Doubao (ep-20260330210911-6pc2r) | 98.80% | 0.528 | `03_doubao_choice_head/data/teacher_train_multivote.jsonl` |
| DeepSeek-V3 | 87.95% | 0.472 | `02_deepseek_v3_choice_head/data/teacher_train_multivote.jsonl` |

多票采样: N=9 (1 base vote + 8 extra API calls), temperature=0.9

## 自一致性过滤策略

对每个样本的每个教师:
- **一致性** = max(vote_counts) / total_votes
- consistency ≥ 0.78 (7/9) → 全权重
- 0.56 ≤ consistency < 0.78 (5-6/9) → 权重 ×0.5
- consistency < 0.56 (≤4/9) → 排除该教师
- 全部排除 → GT one-hot fallback

## 执行

```bash
# 前置: 确保多票采样完成
wc -l 03_doubao_choice_head/data/teacher_train_multivote.jsonl
wc -l 02_deepseek_v3_choice_head/data/teacher_train_multivote.jsonl

# 运行完整 sweep (8 configs)
bash 08_step3_consistency_filter/scripts/run_pipeline.sh
```

## Sweep 配置

| Label | α | LR | Epochs | Seed | Filter | 说明 |
|-------|---|-----|--------|------|--------|------|
| 2t_real_a09 | 0.9 | 1e-4 | 6 | 42 | 默认 | baseline |
| 2t_real_a07 | 0.7 | 1e-4 | 6 | 42 | 默认 | lower GT weight |
| 2t_real_a05 | 0.5 | 1e-4 | 6 | 42 | 默认 | balanced |
| 2t_real_a09_s11 | 0.9 | 1e-4 | 6 | 11 | 默认 | seed test |
| 2t_real_a09_s8 | 0.9 | 1e-4 | 6 | 8 | 默认 | seed test |
| 2t_nofilter | 0.9 | 1e-4 | 6 | 42 | OFF | ablation |
| 2t_strict | 0.9 | 1e-4 | 6 | 42 | 严格 | high=0.89, low=0.67 |
| 2t_twostage | 0.35 | 1.2e-4 | 1+2 | 42 | 默认 | 两阶段 |

## 预期结果

超越单教师最佳 81.93%（DeepSeek-V3 两阶段），目标 ≥ 83%。

## 模型准备（GitHub 不含模型文件）

本仓库上传到 GitHub 时 **不包含模型权重**。克隆代码后需自行下载：

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

在 `setup.env` 中配置：
```bash
export BASE_MODEL_7B="/your/path/to/models/Qwen2.5-7B-Instruct"
```

还需配置 API key（用于多票采样）：
```bash
export DEEPSEEK_API_KEY="your-api-key"
export DOUBAO_API_KEY="your-api-key"
```

> **注意**：运行前需确保 Module 02、03 的多票采样数据已生成。训练产物不上传 GitHub。

## 测试集评估

评估集成在 pipeline 脚本中。也可使用通用测试界面：

```bash
bash scripts/start_quiz.sh
# 默认地址 http://0.0.0.0:7870
```
