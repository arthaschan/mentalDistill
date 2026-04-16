# Module 18: Fisher-Rao 信息几何教师质量分析

## 实验目标

使用 Fisher-Rao 测地线距离（概率单纯形上的黎曼度量）定量分析不同教师软标签的几何特征，解释"为什么弱教师蒸馏会失败"。

## 理论背景

在 5-选项概率单纯形 Δ⁴ 上，Fisher-Rao 距离是唯一（在变换群不变意义下）的黎曼度量：

$$d_{FR}(p, q) = 2 \arccos\left(\sum_{i=1}^{5} \sqrt{p_i q_i}\right)$$

其中 Bhattacharyya 系数 $BC(p,q) = \sum_i \sqrt{p_i q_i}$ 测量两个分布的重叠程度。

**分析维度**：
1. **FR → GT 距离**：教师分布到 GT one-hot 向量的距离 — 衡量教师正确性
2. **教师间距离矩阵**：不同教师之间的几何距离 — 衡量多样性
3. **熵分析**：教师标签的信息量 — 高熵=更多"暗知识"
4. **假软标签 vs 真软标签**：Module 07 的 smooth_eps 标签 vs Module 08 的真实 multivote 标签

**核心假设**：
- 好教师的 FR→GT 距离应该较小（接近 one-hot）
- 弱教师（如 Kimi）的 FR→GT 距离方差应较大（不稳定）
- 假软标签（smooth_eps）在几何上应该是"扁平的"——熵高但信息量低

## 数据来源

分析以下模块的教师标签：
| 模块 | 教师 | 数据文件 |
|------|------|---------|
| 02 | DeepSeek-V3 | `train_head_distill.jsonl` |
| 03 | Doubao | `train_head_distill.jsonl` |
| 04 | Kimi（弱教师） | `train_head_distill.jsonl` |
| 05 | Qwen2.5-14B（同量级） | `train_head_distill.jsonl` |
| 06 | Qwen2.5-32B-API | `train_head_distill.jsonl` |

## 执行

```bash
source setup.env
bash 18_fisher_rao_analysis/scripts/run_analysis.sh
```

纯 CPU 运算，无需 GPU，预计几秒完成。

## 输出

- `outputs/<timestamp>_fisher_rao/fisher_rao_report.json`：完整分析报告
- 控制台打印 Summary Table

## 预期发现

| 指标 | 强教师 (DeepSeek) | 弱教师 (Kimi) | 假软标签 |
|------|-----------------|--------------|---------|
| FR→GT mean | 小（高置信） | 大（低置信） | 固定（ε 决定） |
| FR→GT std | 小（稳定） | 大（不稳定） | ≈0（无变化） |
| Entropy mean | 适中 | 高 | 固定 |
| Peak prob | 高 | 低 | 1-4ε |
