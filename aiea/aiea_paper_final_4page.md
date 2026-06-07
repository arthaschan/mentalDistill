---
title: Choice-Head Distillation for Dental Multiple-Choice Question Answering
author:
  - Tianyuan Chen
date: June 2026
keywords:
  - knowledge distillation
  - medical question answering
  - large language models
  - multiple-choice reasoning
  - decision-space supervision
---

Tianyuan Chen  
Master of Science in Applied Artificial Intelligence  
Hong Kong Chu Hai College, Hong Kong SAR, China
Supervisor: Dr. Richard Tai-Chiu Hsung, Associate Professor, Department of Computer Science, Hong Kong Chu Hai College

# Abstract

Medical large language models achieve strong scores on exam benchmarks, but they remain expensive to deploy. This problem is pronounced in dental multiple-choice question answering, where the output space is limited to five options but many distillation methods still supervise the full vocabulary. This paper proposes Choice-Head distillation, which transfers only the teacher distribution over the answer options. The method is task-aligned, computationally lighter than vocabulary-level distillation, and compatible with black-box API teachers. Experiments use DeepSeek-V3 as the teacher and Qwen2.5-7B and Qwen2.5-14B as students on a CMExam-based resplit. On the 991-question full test set, the best 14B student reaches 89.10% accuracy, exceeding the 87.18% teacher. These results show that for structured medical multiple-choice tasks, decision-space distillation is a practical route to smaller and more deployable models.

# Keywords

knowledge distillation; medical question answering; large language models; multiple-choice reasoning; decision-space supervision

# 1. Introduction

Knowledge distillation compresses large models by transferring soft targets to smaller students [1]-[4]. Recent medical and general-purpose language models also motivate this direction because strong benchmark results often come with high inference cost and limited deployability [5]-[7]. In exam-style medical QA, this trade-off is especially important because the deployment target is often a smaller assistant model rather than the largest available teacher.

This paper studies dental multiple-choice question answering, a five-option decision task. The task structure is simple, but many distillation pipelines still use full-vocabulary logits or free-form targets. That design is inefficient for a fixed-choice problem and is hard to use with API teachers that do not expose internal logits.

We address this mismatch with Choice-Head distillation. The method transfers only the teacher distribution over the five answer options. This paper makes two contributions. First, it presents a task-aligned distillation objective for five-option medical multiple-choice tasks. Second, it shows that this formulation can produce a student that outperforms its teacher: a Qwen2.5-14B student distilled from DeepSeek-V3 reaches 89.10% accuracy on a 991-question CMExam test set, above the teacher accuracy of 87.18%.

Figure 1 shows the difference between full-vocabulary distillation and the proposed decision-space distillation.

# 2. Related Work

Classic distillation transfers softened target distributions from a teacher to a student [1]. Later work extends this idea to compact transformer models, parameter-efficient adaptation, and toolkits for large-scale compression [2]-[4]. In language models, the usual target remains the token vocabulary distribution.

For medical QA, stronger teachers and larger evaluation sets have improved reported performance [6], [7]. However, less attention has been given to the question of whether the supervision target itself should change when the downstream task is a fixed-choice exam. Our method differs from generic LLM distillation because it treats the answer-option distribution, not the full vocabulary, as the transfer object.

# 3. Method

## 3.1 Problem Setting

Given a multiple-choice question $x$ with five candidate answers, the goal is to predict the correct option $y \in \{A, B, C, D, E\}$. Let $p_T$ be the teacher distribution over the five options and $p_S$ be the student distribution over the same options.

Traditional distillation minimizes divergence over the full token vocabulary. In contrast, we distill only the distribution over the decision-relevant option set.

## 3.2 Choice-Head Distillation

Stage 1 training combines distillation and ground-truth supervision:

$$
L = \alpha D_{KL}(p_T \parallel p_S) + (1-\alpha)L_{CE}.
$$

(1)

In the main setting, $\alpha = 0.35$. This objective has three direct advantages over vocabulary-level distillation. It matches the task output space, removes irrelevant supervision dimensions, and works with black-box teachers.

Figure 2 shows the Choice-Head training pipeline.

## 3.3 Training Strategy

The broader project also tested a second stage that applies ground-truth supervised fine-tuning after Stage 1. For the strongest 14B student, however, Stage 2 does not help and can be mildly harmful. The best-performing 14B configuration in this paper therefore uses Stage 1 only.

# 4. Experimental Setup

The evaluation uses a CMExam-based resplit with 6,591 single-choice medical questions across seven subjects [7]. The train, validation, and test splits contain 4,608, 991, and 991 questions. A 125-question dental subset is retained for specialty-focused analysis.

The teacher is DeepSeek-V3 [6]. The students are Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct [5]. Training uses LoRA with rank 16 and LoRA alpha 32 [3]. For the main 14B setting, Stage 1 is run for one epoch with learning rate $1 \times 10^{-4}$.

# 5. Results and Discussion

Table 1. Main results on the CMExam full-data and dental test sets.

| Model | Setting | Full Test Accuracy | Dental Test Accuracy |
| --- | --- | ---: | ---: |
| Qwen2.5-7B | Zero-shot baseline | 76.49% | 68.80% |
| Qwen2.5-14B | Zero-shot baseline | 83.55% | 74.40% |
| DeepSeek-V3 | Teacher | 87.18% | 79.20% |
| Qwen2.5-7B | Choice-Head Stage 1 mean | 85.60% | 73.60% |
| Qwen2.5-14B | Choice-Head Stage 1 mean | 88.67% | 79.20% |
| Qwen2.5-14B | Choice-Head Stage 1 best | 89.10% | 78.40% |

The method improves both student sizes. On the full test set, the 7B student rises from 76.49% to 85.60%, a gain of 9.11 percentage points. The 14B student rises from 83.55% to 88.67% on average, a gain of 5.12 percentage points. The best 14B run reaches 89.10%, which exceeds the 87.18% teacher.

Figure 3 compares the 14B zero-shot baseline, the teacher, and the distilled 14B student.

The main conclusion should be read from the full 991-question test set rather than from the smaller dental subset. The dental subset remains useful, but its smaller size makes it more sensitive to variance. On the main benchmark, the distilled 14B student is stronger than both its zero-shot baseline and the teacher.

The student-over-teacher result does not mean the teacher is weak. It shows that a task-aligned student can combine teacher uncertainty with direct task optimization to learn a stronger decision boundary. Another practical finding is that Stage 2 is not always beneficial. For a strong 14B student, extra ground-truth fine-tuning can erase useful soft-label structure instead of improving it.

Overall, the method works because it keeps only the uncertainty structure that matters for the final option choice and discards irrelevant supervision dimensions.

# 6. Conclusion

This paper presented Choice-Head distillation for dental multiple-choice question answering. The method distills only the five-option answer distribution, not the full vocabulary. This makes the supervision target closer to the task and keeps the framework compatible with black-box teachers. On the 991-question CMExam-based test set, the best 14B student reaches 89.10% accuracy and exceeds the 87.18% DeepSeek-V3 teacher. For structured medical multiple-choice tasks, decision-space distillation is therefore a practical path to smaller and more deployable QA systems.

# References

[1] G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," in NIPS Deep Learning and Representation Learning Workshop, Montreal, Canada, 2015. Available: https://arxiv.org/abs/1503.02531

[2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019. Available: https://arxiv.org/abs/1910.01108

[3] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685, 2021. Available: https://arxiv.org/abs/2106.09685

[4] X. Liu et al., "TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing," arXiv:2002.12620, 2020. Available: https://arxiv.org/abs/2002.12620

[5] Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024. Available: https://arxiv.org/abs/2412.15115

[6] DeepSeek-AI, "DeepSeek-V3 Technical Report," arXiv:2412.19437, 2024. Available: https://arxiv.org/abs/2412.19437

[7] T. Liu et al., "Benchmarking Large Language Models on CMExam: A Comprehensive Chinese Medical Exam Dataset," in Advances in Neural Information Processing Systems, 2023, doi: 10.52202/075280-2283.
