# AIEA 2026 Conference PPT Full Script

## Usage

- Audience: AIEA 2026 conference presentation
- Recommended total duration: 12 to 15 minutes
- Language on slides: English
- Speaker notes: Chinese
- Recommended slide count: 10

---

## Slide 1. Title

### Slide Text

Title:

Choice-Head Distillation for Efficient Dental Multiple-Choice Question Answering

Subtitle:

- Tianyuan Chen
- Department / Affiliation
- Supervisor
- AIEA 2026, Shenzhen

### Visual Suggestion

- Clean title slide
- One short subtitle only
- Optional bottom line with: CMExam, DeepSeek-V3, Qwen2.5-14B

### English Speaker Script

Good morning everyone. My presentation is titled Choice-Head Distillation for Efficient Dental Multiple-Choice Question Answering. This work studies how to transfer the capability of large medical language models into smaller and more deployable student models for standardized dental multiple-choice question answering.

### 中文讲稿

各位老师、各位专家好。我今天汇报的题目是用于高效牙科选择题自动答题的 Choice-Head 蒸馏方法。这个工作的核心目标，是把大型医疗语言模型的答题能力迁移到更小、更容易部署的学生模型上。

---

## Slide 2. Motivation

### Slide Text

Title:

Motivation

Bullets:

- Medical LLMs are strong but costly to deploy
- Dental MCQ answering is a structured five-option task
- Full-vocabulary distillation is inefficient for this setting
- Goal: preserve performance while improving deployability

### Visual Suggestion

- Left: strong teacher model icon
- Right: smaller student model icon
- Middle: arrow labeled distillation

### English Speaker Script

Medical large language models perform strongly on exam-style benchmarks, but deployment remains expensive in terms of inference cost, hardware demand, and reproducibility. In our setting, the task is not open-ended conversation but five-option dental multiple-choice question answering. This makes the task structure much clearer and suggests that a more targeted distillation objective may be more appropriate than generic full-vocabulary supervision.

### 中文讲稿

医疗大语言模型在考试类基准上已经表现得很强，但它们的部署成本依然很高，包括推理成本、硬件需求和复现成本。我们的任务也不是开放式聊天，而是标准的五选一牙科选择题。既然任务结构这么明确，就说明蒸馏目标也不一定要继续沿用全词表那套通用做法。

---

## Slide 3. Research Question

### Slide Text

Title:

Research Question

Bullets:

- What should be distilled for five-option medical MCQs?
- Can a task-aligned student outperform a stronger teacher?
- Can the framework remain compatible with API-based teachers?

### English Speaker Script

The central question is not how to imitate the entire language model, but what information should be transferred for a five-option exam task. We ask whether the student should learn the whole vocabulary distribution, or only the decision structure that matters at evaluation time. We also ask whether such a framework can remain compatible with black-box teachers accessed through APIs.

### 中文讲稿

这里最核心的问题不是“怎样完整模仿一个大语言模型”，而是“对于五选一考试题，到底什么信息最值得迁移”。学生究竟要学整个词表分布，还是只学最后真正影响答题的决策结构？同时，这个框架还必须兼容 API 教师，因为现实中最强教师往往是黑盒模型。

---

## Slide 4. Core Method

### Slide Text

Title:

Choice-Head Distillation

Bullets:

- Distill only A/B/C/D/E option probabilities
- Replace vocabulary-space supervision with decision-space supervision
- Stage 1 loss: KL + CE
- Compatible with both local and API teachers

Formula:

$$
L = \alpha D_{KL}(p_T \parallel p_S) + (1-\alpha)L_{CE}
$$

### Visual Suggestion

- Small method diagram
- Teacher output reduced to five options
- Student choice head supervised by KL + CE

### English Speaker Script

Our method is called Choice-Head distillation. Instead of matching the teacher over the full token vocabulary, we distill only the probability distribution over the five answer options A, B, C, D, and E. This changes the supervision target from generic token space to task-specific decision space. In Stage 1, the student is trained with a mixture of KL divergence on the option distribution and cross-entropy on the gold answer. In our main setting, the distillation weight is 0.35.

### 中文讲稿

我们的方法叫 Choice-Head distillation。核心思想是，不再让学生拟合教师的完整词表输出，而是只蒸馏五个答案选项 A、B、C、D、E 上的概率分布。这样监督目标就从通用 token 空间变成了任务决策空间。在第一阶段里，我们用选项分布上的 KL 散度加标准答案上的交叉熵来共同训练学生。

---

## Slide 5. Why It Matters

### Slide Text

Title:

Why Choice-Head Distillation?

Bullets:

- Better aligned with task structure
- Less redundant than full-vocabulary distillation
- Lower training and memory cost
- Naturally supports black-box teachers

### English Speaker Script

This design matters for three reasons. First, it aligns the training target with the actual downstream task. Second, it removes a large amount of irrelevant supervision, because most vocabulary logits do not matter for a five-option answer. Third, it makes strong black-box teachers usable without requiring internal hidden states or full logits. In short, we keep the part of the teacher signal that directly matters for the final choice.

### 中文讲稿

这个设计有三个关键价值。第一，它让训练目标和任务本身对齐。第二，它去掉了大量无关监督，因为对五选一任务来说，绝大多数词表 logits 都没用。第三，它让黑盒强教师也能被纳入蒸馏流程，而不需要访问内部隐藏状态或完整 logits。

---

## Slide 6. Experimental Setup

### Slide Text

Title:

Experimental Setup

Bullets:

- Dataset: CMExam-based full-data resplit
- Train / Val / Test: 4608 / 991 / 991
- Dental subset test: 125
- Teacher: DeepSeek-V3
- Students: Qwen2.5-7B, Qwen2.5-14B
- LoRA fine-tuning, rank 16, alpha 32

### Visual Suggestion

- Compact table instead of long bullets if desired

### English Speaker Script

We evaluate the method on a CMExam-based full-data resplit built from 6,591 single-choice medical questions. The train, validation, and test splits are 4,608, 991, and 991. The main teacher is DeepSeek-V3, and the students are Qwen2.5-7B and Qwen2.5-14B. The strongest 14B setting uses one stage of Choice-Head distillation with LoRA fine-tuning.

### 中文讲稿

实验使用的是基于 CMExam 构建的全量重分割数据，一共 6591 道单选医学题。主教师是 DeepSeek-V3，学生是 Qwen2.5-7B 和 Qwen2.5-14B。表现最好的 14B 配置只使用了一阶段的 Choice-Head 蒸馏，并采用 LoRA 微调。

---

## Slide 7. Main Results

### Slide Text

Title:

Main Results

Table or bullets:

- Teacher: 87.18%
- 14B zero-shot: 83.55%
- 14B distilled mean: 88.67%
- 14B distilled best: 89.10%

### Visual Suggestion

- Four-bar chart
- Zero-shot, Teacher, Distilled Mean, Distilled Best

### English Speaker Script

The main result is straightforward. The 14B zero-shot baseline reaches 83.55 percent. The teacher reaches 87.18 percent. After Choice-Head distillation, the 14B student reaches 88.67 percent on average and 89.10 percent at best. This means the distilled student not only improves substantially over its own baseline, but also surpasses the teacher on the same 991-question test set. The 7B student also improves strongly, showing that the method is not limited to one model size.

### 中文讲稿

核心结果非常直接。14B 学生的零样本基线是 83.55%，教师是 87.18%。经过 Choice-Head 蒸馏后，14B 学生的平均准确率达到 88.67%，最佳达到 89.10%。也就是说，学生不仅明显超过了自己的零样本基线，而且在同一个 991 题测试集上也超过了教师。

---

## Slide 8. Contribution 1 and 2

### Slide Text

Title:

Two Main Contributions

Bullets:

- Contribution 1: task-aligned decision-space distillation
- Contribution 2: student-over-teacher performance on a 991-question test set

### English Speaker Script

The first contribution is methodological. We show that for structured medical multiple-choice tasks, the distillation target should follow the decision structure of the task. The second contribution is empirical. On a 991-question test set, a smaller student can outperform a stronger teacher under this task-aligned formulation.

### 中文讲稿

第一个贡献是方法上的：我们说明了对于结构化医学选择题，蒸馏目标应该围绕任务决策结构来定义。第二个贡献是实证上的：在更大、更可靠的测试集上，一个更小的学生模型在这种任务对齐的设置下可以超过更强的教师模型。

---

## Slide 9. Discussion

### Slide Text

Title:

Discussion

Bullets:

- Stage 2 is not always beneficial
- Simpler training can be stronger
- Effective transfer depends on task-aligned supervision

### English Speaker Script

One practical finding is that a more complex pipeline is not always better. In our experiments, the strongest 14B result comes from Stage 1 only. This suggests that for stronger students and larger training sets, extra fine-tuning can erase useful soft-label structure instead of improving it. The broader message is that distillation should be designed around task structure, not inherited unchanged from general LLM training practice.

### 中文讲稿

一个很重要的实际发现是，更复杂的训练流程不一定更好。在我们的实验里，最强的 14B 结果恰恰来自只做 Stage 1。这说明对于更强的学生和更大的训练集来说，额外的微调有时会抹去有价值的软标签结构，而不是带来提升。更大的结论是，蒸馏应该围绕任务结构来设计，而不是机械沿用通用大模型训练范式。

---

## Slide 10. Conclusion

### Slide Text

Title:

Conclusion

Bullets:

- Choice-Head distillation is effective for dental MCQ answering
- Smaller students can remain deployable and highly accurate
- Decision-space transfer is a practical path for structured medical QA

### English Speaker Script

In conclusion, this work suggests that efficient deployment does not require reproducing a teacher's entire language model behavior. For structured medical multiple-choice tasks, transferring the option-level decision distribution can be enough to produce strong performance. Thank you for listening, and I welcome your questions.

### 中文讲稿

最后，这项工作说明了一件事情：想实现高效部署，并不一定要重现教师模型的全部语言行为。对于结构化医学选择题任务，只要迁移选项级决策分布，就可能获得很强的效果。谢谢各位老师和专家聆听，欢迎批评指正。

---

## Q&A Preparation

### Q1. Why distill only five options instead of the full vocabulary?

Short English answer:

Because the downstream task is a five-option decision problem. Full-vocabulary supervision introduces unnecessary dimensions, while option-level supervision is more task-aligned and more efficient.

中文回答：

因为这个下游任务本质上就是五选一决策问题。全词表监督会引入大量无关维度，而选项级监督更贴近任务目标，也更高效。

### Q2. Why can the student outperform the teacher?

Short English answer:

The student learns teacher soft-label structure while still optimizing directly for the benchmark task. Under a task-aligned objective, this can produce a stronger task-specific decision boundary.

中文回答：

学生一方面学习教师的软标签结构，另一方面又直接针对基准任务优化。在任务对齐的目标下，这两部分信息可能共同形成更强的任务决策边界。

### Q3. Can this method be extended to open-ended QA?

Short English answer:

Not directly. The current method is designed for structured multiple-choice tasks. Extending it to open-ended QA would require a different supervision target.

中文回答：

不能直接照搬。当前方法是为结构化选择题任务设计的。如果要扩展到开放式问答，就需要重新定义监督目标，而不能只沿用现在的五选项决策空间。

中文回答：

不能直接照搬。当前方法是专门为结构化选择题任务设计的。如果扩展到开放问答，就需要重新定义监督目标，而不能简单沿用五选项决策空间。
