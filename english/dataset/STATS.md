# English Dental MCQ Dataset — Build Stats (seed=42)

## Single-best pool (main KD task: choose one of A-E)
- total after clean+dedup: **636**  (BoF 272 + NBDE 364)
- split: train **447** / val **95** / test **94**  (70/15/15 stratified by subject)
- answer-letter dist (all): {'A': 121, 'B': 161, 'C': 144, 'D': 135, 'E': 75}
- n_options dist: {4: 250, 5: 386}

### Subjects (single-best pool)
| subject | n |
|---|---|
| Periodontics | 73 |
| Pharmacology | 60 |
| Oral Diagnosis | 58 |
| Patient Management | 55 |
| Operative Dentistry | 49 |
| Oral and Maxillofacial Surgery | 43 |
| Prosthodontics | 41 |
| Endodontics | 38 |
| Human Disease | 33 |
| Oral Pathology | 31 |
| Restorative Dentistry | 28 |
| Oral Medicine | 27 |
| Dental Materials | 27 |
| Radiology | 27 |
| Child Dental Health and Orthodontics | 25 |
| Oral Surgery | 21 |

### Split subject balance
- train subjects: {'Periodontics': 51, 'Pharmacology': 42, 'Oral Diagnosis': 41, 'Patient Management': 38, 'Operative Dentistry': 34, 'Oral and Maxillofacial Surgery': 30, 'Prosthodontics': 29, 'Endodontics': 27, 'Human Disease': 23, 'Oral Pathology': 22, 'Restorative Dentistry': 20, 'Dental Materials': 19, 'Oral Medicine': 19, 'Radiology': 19, 'Child Dental Health and Orthodontics': 18, 'Oral Surgery': 15}
- val   subjects: {'Periodontics': 11, 'Pharmacology': 9, 'Oral Diagnosis': 9, 'Patient Management': 8, 'Operative Dentistry': 7, 'Prosthodontics': 6, 'Oral and Maxillofacial Surgery': 6, 'Endodontics': 6, 'Human Disease': 5, 'Oral Pathology': 5, 'Dental Materials': 4, 'Child Dental Health and Orthodontics': 4, 'Oral Medicine': 4, 'Radiology': 4, 'Restorative Dentistry': 4, 'Oral Surgery': 3}
- test  subjects: {'Periodontics': 11, 'Patient Management': 9, 'Pharmacology': 9, 'Oral Diagnosis': 8, 'Operative Dentistry': 8, 'Oral and Maxillofacial Surgery': 7, 'Prosthodontics': 6, 'Endodontics': 5, 'Human Disease': 5, 'Oral Medicine': 4, 'Radiology': 4, 'Dental Materials': 4, 'Oral Pathology': 4, 'Restorative Dentistry': 4, 'Child Dental Health and Orthodontics': 3, 'Oral Surgery': 3}

## True/False auxiliary set (source: MCQs for Dentistry)
- total after clean+dedup: **378**  (multi-select; answer = set of TRUE statements)
- kept as generalization/robustness set, NOT in main train/val/test.
- subjects:
| Restorative Dentistry | 69 |
| Dental Materials | 53 |
| Oral Surgery | 42 |
| Human Disease | 41 |
| Oral Medicine | 38 |
| Oral Pathology | 37 |
| General Dentistry | 33 |
| Therapeutics | 33 |
| Child Dental Health and Orthodontics | 32 |

## Files
- dataset/single_best_all.jsonl, train/val/test.jsonl (main)
- dataset/tf_all.jsonl (auxiliary)

## Notes / honest boundaries
- Small training set (~445 items): use few-shot teacher labeling + multi-seed + report CI.
- OCR residue possible (spelling of Latin/drug terms); answers/keys verified structurally.
- Sources are copyrighted revision books — labels/derived data for internal research only; do not redistribute raw text.
