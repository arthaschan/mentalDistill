# 32B->7B No-overlap Comparison

- Teacher baseline (direct 32B): **83.13%**

| Config | Test Accuracy | Log | Wrong File |
|---|---:|---|---|
| A. Historical baseline params | 75.90% | `train32_nooverlap_hist_seed2.log` | `./dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_hist_seed2/test_wrong.jsonl` |
| B. Low-KL surpass-oriented params | 75.90% | `train32_nooverlap_lowkl_seed2.log` | `./dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_lowkl_seed2/test_wrong.jsonl` |

