# 11_multi_seed_ensemble

## 目标
方案 A：通过多 seed 集成训练消除随机种子方差，获得可靠的性能评估。

## 方法
- 使用 Step 3 最佳配置 (α=0.9, lr=1e-4, rank=16, 6 epochs, 2-teacher 融合标签)
- 训练 7 个不同 seed (7, 8, 11, 42, 55, 77, 123)
- 每个 seed 独立产生一个 LoRA adapter
- 推理阶段采用 Majority Vote

## 执行
```bash
source setup.env
bash 11_multi_seed_ensemble/scripts/run_ensemble.sh
```

## 输入
- `08_step3_consistency_filter/data/train_fused_2t_real_a09.jsonl` — 2-teacher 融合标签
- `00_baseline_gt_sft/data/{val,test}.jsonl`

## 输出
- `outputs/seed_*/best/` — 每个 seed 的最佳 adapter
- `seed_results.csv` — 各 seed 的 val/test 结果
- `ensemble_results.json` — 集成详细结果（含逐样本投票记录）

## 实验结果

| Seed | Best Val (%) | Test Acc (%) |
|------|-------------|-------------|
| 7 | 74.32 | 75.90 |
| 8 | 78.38 | 75.90 |
| 11 | 77.03 | 74.70 |
| **42** | **75.68** | **79.52** |
| 55 | 75.68 | 78.31 |
| 77 | 77.03 | 74.70 |
| 123 | 74.32 | 71.08 |

- **均值 Test**: 75.73% | **标准差**: 2.82pp
- **7-Seed Ensemble (Majority Vote)**: **79.52%** (66/83)

### 结论
- 集成等于最佳单 seed (42)，没有超越
- Seed 方差 8.44pp (71.08%–79.52%)，证实测试集太小（83题）
- Val 最佳与 Test 无稳定相关
