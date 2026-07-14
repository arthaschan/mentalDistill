# English Dental — Entropy=Difficulty External Validation (n=636, 7 teachers)

Gold standard = CROSS-MODEL CONSENSUS (# of 7 teachers wrong), since English has no human difficulty labels.

## H4: entropy locates a teacher's own error subset
| teacher | acc% | high-entropy err% | low-entropy err% | ratio |
|---|---|---|---|---|
| Qwen32B | 73.9 | 44.92 | 8.76 | 5.13× |
| GLM32B | 72.64 | 47.48 | 7.23 | 6.57× |
| Qwen14B | 70.6 | 49.19 | 10.94 | 4.5× |
| Phi4 | 69.34 | 51.26 | 10.06 | 5.09× |
| Gemma27B | 65.72 | 56.29 | 12.26 | 4.59× |
| Yi34B | 62.89 | 55.97 | 18.24 | 3.07× |
| Qwen7B | 62.58 | 57.23 | 17.61 | 3.25× |

## 5d: entropy vs cross-model consensus difficulty (external gold)
- **mean-entropy vs consensus ρ = 0.6945** (p_perm=0.001)

| teacher | entropy vs consensus ρ | p_perm |
|---|---|---|
| GLM32B | 0.611 | 0.001 |
| Gemma27B | 0.6288 | 0.001 |
| Phi4 | 0.6275 | 0.001 |
| Qwen14B | 0.4722 | 0.001 |
| Qwen32B | 0.5337 | 0.001 |
| Qwen7B | 0.4926 | 0.001 |
| Yi34B | 0.4636 | 0.001 |

### Consensus difficulty gradient (mean teacher entropy rises with #wrong)
| #teachers wrong | n items | mean teacher entropy |
|---|---|---|
| 0 | 250 | 0.0508 |
| 1 | 88 | 0.1874 |
| 2 | 66 | 0.292 |
| 3 | 38 | 0.3518 |
| 4 | 44 | 0.3746 |
| 5 | 47 | 0.4172 |
| 6 | 52 | 0.4318 |
| 7 | 51 | 0.3312 |

## 5d-null: surface-text artifact controls (want ≈0)
- entropy vs stem length ρ = 0.046
- entropy vs #negation words ρ = 0.0855
- consensus vs stem length ρ = -0.0657