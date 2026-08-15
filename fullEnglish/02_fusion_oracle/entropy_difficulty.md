# fullEnglish — 熵=难度外部验证 (n=600, 4 teachers)

金标准 = 跨模型共识错误数 (英文医学无人类难度标注).

## H4: 熵定位教师自身的错误子集
| teacher | acc% | high-entropy err% | low-entropy err% | ratio |
|---|---|---|---|---|
| Llama70B-AWQ | 79.5 | 37.18 | 6.19 | 6.01x |
| Qwen32B | 72.67 | 48.33 | 10.27 | 4.7x |
| Phi4 | 72.0 | 50.33 | 5.67 | 8.88x |
| Qwen14B | 67.83 | 54.88 | 9.9 | 5.54x |

## 5d: 熵 vs 跨模型共识难度 (外部金标准)
- **mean-entropy vs consensus rho = 0.6288** (p_perm=0.001)

| teacher | rho | p_perm |
|---|---|---|
| Llama70B-AWQ | 0.4287 | 0.001 |
| Phi4 | 0.577 | 0.001 |
| Qwen14B | 0.497 | 0.001 |
| Qwen32B | 0.4406 | 0.001 |

### 共识难度梯度 (教师平均熵随错题数上升)
| #teachers wrong | n items | mean teacher entropy |
|---|---|---|
| 0 | 344 | 0.0266 |
| 1 | 69 | 0.1563 |
| 2 | 54 | 0.2353 |
| 3 | 61 | 0.2615 |
| 4 | 72 | 0.2403 |

## 5d-null: 表面文本 artifact 对照 (期望约 0)
- entropy vs stem length rho = -0.0056
- entropy vs #negation words rho = 0.1364
- consensus vs stem length rho = 0.0534