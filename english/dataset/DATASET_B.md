# Dataset — Scheme B (stratified, seed=42)

## Splits
| split | n | composition | role |
|---|---|---|---|
| train_main | 2551 | UK/US 447 + MedMCQA 2104 | training pool |
| val | 95 | UK/US only | model selection |
| test_ukus | 94 | UK/US only (BoF+NBDE) | **PRIMARY headline** |
| test_medmcqa | 371 | MedMCQA only | cross-source generalization |

- train_main group mix: {'UKUS': 447, 'MedMCQA': 2104}
- train_main n_options: {4: 2279, 5: 272}
- test_ukus n_options: {4: 38, 5: 56} ; test_medmcqa n_options: {4: 371}

## Rationale
- Primary claim proven on CLEAN UK/US clinical-exam questions (test_ukus), uncontaminated by
  the Indian NEET-MDS distribution.
- MedMCQA expands training data (mitigates small-sample) AND yields a free cross-source
  generalization test (does the distilled student hold on a different English dental distribution?).
- val stays UK/US so selection matches the primary test distribution.

## Honest boundaries
- Distribution shift: MedMCQA (India) vs BoF (UK) / NBDE (US). Kept as SEPARATE test, not mixed in.
- MedMCQA 'Dental' label was polluted; we kept only keyword-verified dental single-best (2087).
- 4-option (MedMCQA) vs mixed 4/5-option (UK/US); Choice-Head handles variable option count.
