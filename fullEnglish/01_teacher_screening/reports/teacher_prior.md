# fullEnglish — Teacher Screening (教师预评估)

Pool: 600 items (MedQA/MedMCQA/MMLU 各 200). Teachers: 5.

## Teacher prior (zero-shot, English prompt)

| rank | teacher | acc% | mean_entropy |
|---|---|---|---|
| 1 | DeepSeekV3 | 81.67 | — |
| 2 | Llama70B-AWQ | 79.5 | 0.0547 |
| 3 | Qwen32B  ⬅ 学生 base | 72.67 | 0.0652 |
| 4 | Phi4 | 72.0 | 0.2426 |
| 5 | Qwen14B | 67.83 | 0.0767 |

## Per-source accuracy
| teacher | medmcqa | medqa | mmlu |
|---|---|---|---|
| DeepSeekV3 | 74.5 | 80.5 | 90.0 |
| Llama70B-AWQ | 70.0 | 80.5 | 88.0 |
| Qwen32B | 62.0 | 71.0 | 85.0 |
| Phi4 | 62.0 | 73.0 | 81.0 |
| Qwen14B | 57.5 | 66.5 | 79.5 |

## Headroom (能否超越教师的关键判据)
- 最强教师 **DeepSeekV3 = 81.67%**
- 学生零样本地板 **Qwen32B = 72.67%**
- **headroom = +9.00pp**
- => headroom 较大 (>8pp), 单靠决策空间监督难填平; 主实验仍跑, 但预期可能不超越, 如实报告, 参考英文牙科 (headroom≈20pp 未超越) 的机制分析.

### Per-source winners
| source | winner | acc% |
|---|---|---|
| medmcqa | DeepSeekV3 | 74.5 |
| medqa | DeepSeekV3 | 80.5 |
| mmlu | DeepSeekV3 | 90.0 |