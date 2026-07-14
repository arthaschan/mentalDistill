# English Dental — Teacher Screening

Pool: 636 single-best items. Teachers screened: 7.

## Teacher prior (zero-shot, English prompt)

| rank | teacher | acc% | mean_entropy |
|---|---|---|---|
| 1 | Qwen32B | 73.9 | 0.0831 |
| 2 | GLM32B | 72.64 | 0.4101 |
| 3 | Qwen14B | 70.6 | 0.0832 |
| 4 | Phi4 | 69.34 | 0.2744 |
| 5 | Gemma27B | 65.72 | 0.1842 |
| 6 | Yi34B | 62.89 | 0.2998 |
| 7 | Qwen7B | 62.58 | 0.1761 |

## Complementarity check (GO/NO-GO precursor)

- Best overall teacher: **Qwen32B** (73.9%)
- Distinct per-subject winners: **5** / 16 subjects
- Single teacher dominates every subject: **False**
- => No single boss on English dental. Fusion has a chance; run fusion oracle (Screening #3).

### Per-subject winners
| subject | winner | acc% |
|---|---|---|
| Child Dental Health and Orthodontics | GLM32B | 68.0 |
| Dental Materials | GLM32B | 66.7 |
| Endodontics | GLM32B | 81.6 |
| Human Disease | Qwen32B | 84.8 |
| Operative Dentistry | GLM32B | 73.5 |
| Oral Diagnosis | Phi4 | 74.1 |
| Oral Medicine | Qwen14B | 88.9 |
| Oral Pathology | GLM32B | 83.9 |
| Oral Surgery | GLM32B | 81.0 |
| Oral and Maxillofacial Surgery | Gemma27B | 72.1 |
| Patient Management | Qwen32B | 85.5 |
| Periodontics | GLM32B | 83.6 |
| Pharmacology | GLM32B | 95.0 |
| Prosthodontics | Qwen32B | 63.4 |
| Radiology | Qwen14B | 59.3 |
| Restorative Dentistry | Qwen14B | 46.4 |