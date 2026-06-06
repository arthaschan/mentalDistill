# Choice-Head Distillation for Efficient Dental Multiple-Choice Question Answering

## Abstract

Large language models perform strongly on medical question answering benchmarks, but their deployment cost remains high. This challenge is especially evident in standardized dental multiple-choice question answering, where the output space is small and structured, yet conventional distillation still relies on full-vocabulary supervision. We propose Choice-Head distillation, a task-aligned framework that distills only the probability distribution over the five answer options A/B/C/D/E. This design reduces computational redundancy, supports black-box API teachers, and aligns learning with the downstream decision space. Experiments on CMExam-based full-data and dental settings use DeepSeek-V3 as the teacher and Qwen2.5-7B and Qwen2.5-14B as students. On a 991-question full-data test set, the best 14B student reaches 89.10% accuracy, surpassing the 87.18% teacher while remaining much more deployable. The results show that decision-space distillation is an effective and practical alternative to vocabulary-level supervision for structured medical multiple-choice tasks.

## Index Terms

Knowledge distillation, medical question answering, large language models, multiple-choice reasoning, decision-space supervision.

## 1. Introduction

Medical large language models have achieved strong results on professional examination benchmarks, including Chinese medical licensing datasets. However, strong benchmark accuracy does not automatically translate into practical deployment value. In settings such as educational support, exam training, and lightweight decision assistance, inference cost and reproducibility matter as much as raw performance.

This paper studies dental multiple-choice question answering, where each question has five candidate answers. Although the output space is naturally limited, many LLM distillation methods still operate on full-vocabulary logits or free-form generation targets. This creates unnecessary redundancy and makes black-box API teachers difficult to use, because internal logits are often inaccessible.

To address this mismatch, we reformulate distillation around the task decision structure rather than the general language modeling space. We propose Choice-Head distillation, which transfers the teacher distribution over the answer options A/B/C/D/E.

This paper makes two contributions. First, it introduces a task-aligned distillation framework for five-option medical multiple-choice tasks. Second, it shows that under this formulation, a smaller student can surpass a stronger teacher: a Qwen2.5-14B student distilled from DeepSeek-V3 reaches 89.10% accuracy on a 991-question CMExam test set, exceeding the teacher accuracy of 87.18%.

Figure 1 conceptually contrasts full-vocabulary distillation with the proposed Choice-Head decision-space distillation.

## 2. Related Work

Knowledge distillation transfers knowledge from large models to smaller students through soft targets, intermediate representations, or task-specific supervision. In large language models, distillation has been extended to vocabulary-level logits, rationale generation, and engineering toolkits for practical compression. For medical LLMs, most prior work emphasizes stronger teachers and larger datasets, but less attention has been paid to whether the supervision target itself should change with task structure.

Our method differs from generic LLM distillation by treating the answer-option distribution as the primary transfer object. This makes the framework lightweight, black-box compatible, and better aligned with standardized multiple-choice tasks.

## 3. Method

### 3.1 Problem Setting

Given a multiple-choice question $x$ with five candidate answers, the goal is to predict the correct option $y \in \{A, B, C, D, E\}$. Let $p_T$ be the teacher distribution over the five options and $p_S$ be the student distribution over the same options.

Traditional distillation minimizes divergence over the full token vocabulary. In contrast, we distill only the distribution over the decision-relevant option set.

### 3.2 Choice-Head Distillation

Stage 1 training uses a combination of distillation and ground-truth supervision:

$$
L = \alpha D_{KL}(p_T \parallel p_S) + (1-\alpha)L_{CE}.
$$

In our main setting, $\alpha = 0.35$. Compared with vocabulary-level distillation, this formulation has three advantages: it is more task-aligned, more computationally efficient, and naturally compatible with black-box teachers.

Figure 2 shows the Choice-Head training pipeline with option-level teacher supervision.

### 3.3 Training Strategy

The broader project explored a two-stage strategy in which Stage 2 adds ground-truth supervised fine-tuning after Stage 1. However, large-scale experiments show that for a strong 14B student, Stage 2 is unnecessary and can even be mildly harmful. Therefore, the best-performing 14B configuration in this paper uses Stage 1 only.

## 4. Experimental Setup

We use a CMExam-based resplit built from 6,591 single-choice medical questions across seven subjects. The training, validation, and test sets contain 4,608, 991, and 991 questions, respectively. A dental subset is also retained for specialty-focused evaluation.

The main teacher is DeepSeek-V3, which reaches 87.18% accuracy on the 991-question test set. Student models are Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct. Training uses LoRA with rank 16 and LoRA alpha 32. For the main 14B setting, we use one Stage 1 epoch with learning rate $1 \times 10^{-4}$.

## 5. Results and Discussion

Table 1 reports the main results on the CMExam full-data and dental test sets.

| Model | Setting | Full Test Accuracy | Dental Test Accuracy |
| --- | --- | ---: | ---: |
| Qwen2.5-7B | Zero-shot baseline | 76.49% | 68.80% |
| Qwen2.5-14B | Zero-shot baseline | 83.55% | 74.40% |
| DeepSeek-V3 | Teacher | 87.18% | 79.20% |
| Qwen2.5-7B | Choice-Head Stage 1 mean | 85.60% | 73.60% |
| Qwen2.5-14B | Choice-Head Stage 1 mean | 88.67% | 79.20% |
| Qwen2.5-14B | Choice-Head Stage 1 best | 89.10% | 78.40% |

The method yields strong gains for both student sizes. The 7B student improves by 9.11 percentage points over its full-test zero-shot baseline, while the 14B student improves by 5.12 percentage points on average. Most importantly, the best 14B student reaches 89.10%, surpassing the 87.18% teacher.

Figure 3 compares the 14B zero-shot baseline, the teacher, and the distilled 14B student.

This student-over-teacher result should not be interpreted as a failure of distillation. Instead, it suggests that when the distillation target is aligned with task structure, the student can combine teacher soft-label information with task-specific optimization to produce a stronger decision boundary.

Another practical finding is that Stage 2 is not always beneficial. For stronger students and larger training sets, additional ground-truth fine-tuning can erase useful soft-label structure rather than improve it. In our setting, the simpler Stage 1-only configuration is also the best one.

The method works because it removes irrelevant supervision dimensions, preserves the most useful uncertainty structure across answer options, and allows strong API teachers to be incorporated into training.

## 6. Conclusion

This paper presented Choice-Head distillation for dental multiple-choice question answering. By distilling only the five-option answer distribution instead of the full vocabulary, the method aligns supervision with task structure and remains compatible with black-box teachers. On a 991-question CMExam-based test set, the best 14B student reaches 89.10% accuracy and surpasses the 87.18% DeepSeek-V3 teacher. These results suggest that for structured medical multiple-choice tasks, decision-space distillation is a practical path toward smaller and more deployable QA systems.

## References

[1] G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," in NIPS Deep Learning and Representation Learning Workshop, Montreal, Canada, 2015.

[2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019.

[3] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685, 2021.

[4] X. Liu et al., "TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing," arXiv:2002.12620, 2020.

[5] Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024.

[6] DeepSeek-AI, "DeepSeek-V3 Technical Report," arXiv:2412.19437, 2024.

[7] D. Chong et al., "Benchmarking Large Language Models on CMExam: A Comprehensive Chinese Medical Exam Dataset," in Advances in Neural Information Processing Systems, 2023.
