#!/usr/bin/env python3
"""Scheme B dataset rebuild (stratified design):
  - UK/US 636 (BoF+NBDE): keep FROZEN existing train/val/test (447/95/94).
    * test(94) = PRIMARY clean headline test set.
  - MedMCQA 2087 dental: split 85/15 -> medmcqa_train (join training pool) + medmcqa_test
    (cross-source generalization test).
Outputs english/dataset/:
  train_main.jsonl   = UK/US train (447) + MedMCQA train           (combined training pool)
  val.jsonl          = UK/US val (95)                              [unchanged, model selection]
  test_ukus.jsonl    = UK/US test (94)                             [PRIMARY headline]
  test_medmcqa.jsonl = MedMCQA test                                [cross-source generalization]
  DATASET_B.md       audit
"""
import json, os, random
from collections import Counter
random.seed(42)
DS="english/dataset"

def load(p): return [json.loads(l) for l in open(p)]
def dump(p,rows):
    with open(p,"w") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

# UK/US frozen splits (already built by build_dataset.py)
uk_train=load(f"{DS}/train.jsonl"); uk_val=load(f"{DS}/val.jsonl"); uk_test=load(f"{DS}/test.jsonl")
for r in uk_train+uk_val+uk_test: r.setdefault("group","UKUS")

# MedMCQA dental
med=load("english/00_data/medmcqa_dental.jsonl")
random.shuffle(med)
n_test=int(round(len(med)*0.15))
med_test=med[:n_test]; med_train=med[n_test:]
for r in med_train+med_test:
    r["group"]="MedMCQA"; r.setdefault("format","single_best")

train_main = uk_train + med_train
random.shuffle(train_main)

dump(f"{DS}/train_main.jsonl", train_main)
dump(f"{DS}/val.jsonl", uk_val)                 # unchanged
dump(f"{DS}/test_ukus.jsonl", uk_test)          # primary
dump(f"{DS}/test_medmcqa.jsonl", med_test)      # cross-source

def optdist(rows): return dict(sorted(Counter(r["n_options"] for r in rows).items()))
def grpdist(rows): return dict(Counter(r["group"] for r in rows))
md=f"""# Dataset — Scheme B (stratified, seed=42)

## Splits
| split | n | composition | role |
|---|---|---|---|
| train_main | {len(train_main)} | UK/US {len(uk_train)} + MedMCQA {len(med_train)} | training pool |
| val | {len(uk_val)} | UK/US only | model selection |
| test_ukus | {len(uk_test)} | UK/US only (BoF+NBDE) | **PRIMARY headline** |
| test_medmcqa | {len(med_test)} | MedMCQA only | cross-source generalization |

- train_main group mix: {grpdist(train_main)}
- train_main n_options: {optdist(train_main)}
- test_ukus n_options: {optdist(uk_test)} ; test_medmcqa n_options: {optdist(med_test)}

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
"""
open(f"{DS}/DATASET_B.md","w").write(md)
print(f"train_main={len(train_main)} (UKUS {len(uk_train)} + MedMCQA {len(med_train)})")
print(f"val={len(uk_val)}  test_ukus={len(uk_test)}  test_medmcqa={len(med_test)}")
print(f"-> {DS}/DATASET_B.md")
