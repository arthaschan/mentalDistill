# Module 17: α-散度蒸馏实验

## 实验目标

将 Choice-Head 蒸馏中的 KL 散度替换为信息几何中更一般的 **α-散度族**，验证不同散度度量对蒸馏效果的影响。

## 理论背景

Amari α-散度是信息几何中的核心概念，KL 散度是其特例：

$$D_\alpha(p \| q) = \frac{4}{1-\alpha^2}\left(1 - \sum_i p_i^{(1+\alpha)/2} \cdot q_i^{(1-\alpha)/2}\right)$$

| α 值 | 对应散度 | 特性 |
|------|---------|------|
| α → 1 | KL(p‖q) 前向 KL | mode-seeking，当前基线 |
| α = 0.5 | — | 介于 KL 和 Hellinger 之间 |
| α = 0 | Hellinger 距离 | 对称，对教师错误更鲁棒 |
| α → -1 | KL(q‖p) 反向 KL | mode-covering |

**核心假设**：弱教师（如 Module 04 Kimi, 61.45%）的蒸馏失败可能与 KL(p‖q) 的 mode-seeking 特性有关——它强迫学生在教师的错误选项上也分配概率。使用 α=0（Hellinger）可能对教师噪声更鲁棒。

## 实验设计

- 教师：DeepSeek-V3（复用 Module 02 数据）
- 学生：Qwen2.5-7B-Instruct
- Sweep：α ∈ {-1, 0, 0.5, 1} × seed ∈ {11, 42} = 8 runs
- 其余超参与 Module 02 Stage1 一致（lr=1.2e-4, rank=16, α_weight=0.35, 1 epoch）
- 测试集：83 题牙科

## 模型准备（GitHub 不含模型文件）

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

在 `setup.env` 中配置：
```bash
export BASE_MODEL_7B="/your/path/to/models/Qwen2.5-7B-Instruct"
```

## 数据准备

复用 Module 02 的数据（软链接或拷贝）：
```bash
cp 02_deepseek_v3_choice_head/data/train_head_distill.jsonl 17_alpha_divergence/data/
cp 02_deepseek_v3_choice_head/data/{train,val,test}.jsonl 17_alpha_divergence/data/
```

## 执行

```bash
source setup.env
bash 17_alpha_divergence/scripts/run_train.sh
```

## 结果汇总

```bash
grep "TEST-BEST" 17_alpha_divergence/runs/*/logs/train_*.log
```

## 测试集评估

```bash
bash scripts/start_quiz.sh
# 默认地址 http://0.0.0.0:7870
```

## 关键对比

| α | 散度名称 | Seed 11 Test | Seed 42 Test | 双种子均值 |
|---|---------|-------------|-------------|-----------|
| 0.5 | α=0.5 | **75.90%** | 72.29% | **74.10%** |
| 0.0 | Hellinger | 73.49% | 74.70% | **74.10%** |
| 1.0 | KL(p‖q) 前向 | 74.70% | 69.88% | 72.29% |
| -1.0 | KL(q‖p) 反向 | 72.29% | 72.29% | 72.29% |

> Module 02 基线（经超参搜索的 KL）：seed11=81.93%, seed42=79.52%
>
> **注**：Module 17 使用统一超参（未针对每个 α 值调参），与 Module 02 的绝对值不可直接比较。核心价值在于**相同超参下不同散度的相对排序**。

### 关键发现

1. **α=0.5 和 Hellinger 并列最优**（均值 74.10%），均超过标准 KL（72.29%）+1.81pp
2. **KL(p‖q) 种子敏感性最大**：seed 11→74.70%, seed 42→69.88%（Δ=4.82pp）
3. **逆 KL 最稳定**：两个种子均为 72.29%（Δ=0），但绝对准确率最低
4. **所有 seed=11 验证集准确率完全相同**（77.03%），差异仅在测试集泛化时显现
