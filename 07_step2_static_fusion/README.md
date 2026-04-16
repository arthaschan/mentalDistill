# 07_step2_static_fusion — Stage 2: 多教师静态标签融合

## 实验目标

将多个教师的软标签进行加权融合，测试多教师融合是否优于单教师蒸馏。

## 背景

单教师蒸馏的天花板来自单一教师的"认知盲区"。通过融合来自不同教师（DeepSeek-V3、Doubao、Kimi、14B 本地）的标签分布，期望获得更全面的知识信号。

## 历史结果

- 四教师融合最佳：79.52%（未超越单教师 DeepSeek 两阶段的 81.93%）
- 原因分析：所有"软标签"实际为 smooth_eps 假分布（0.80/0.05），融合的是假概率而非真实 dark knowledge
- 该结论推动了 Module 08（真实多票融合）的设计

## 数据来源

依赖 Module 02-05 已生成的教师标签数据。

## 目录内容

- `data/`: 融合后的训练数据
- `configs/`: 融合权重配置
- `scripts/run_grid_search.sh`: 网格搜索入口
- `scripts/run_round3_sweep.sh`: 第三轮参数扫描

## 模型准备（GitHub 不含模型文件）

本仓库上传到 GitHub 时 **不包含模型权重**。克隆代码后需自行下载：

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

在 `setup.env` 中配置：
```bash
export BASE_MODEL_7B="/your/path/to/models/Qwen2.5-7B-Instruct"
```

> **注意**：运行前需确保 Module 02-05 的教师标签已生成。训练产物不上传 GitHub。

## 执行

```bash
source setup.env
bash 07_step2_static_fusion/scripts/run_grid_search.sh
```

## 测试集评估

本模块的评估集成在 grid search 脚本中。也可使用通用测试界面：

```bash
bash scripts/start_quiz.sh
# 默认地址 http://0.0.0.0:7870（加载 Module 00 的测试集）
```

## 复现注意事项

- 该模块是重要的负结果实验——证明了假软标签融合无效
- 它直接推动了 Module 08 使用真实多票采样的设计
