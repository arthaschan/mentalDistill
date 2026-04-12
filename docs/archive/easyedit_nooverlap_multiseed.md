# 32B Teacher -> 7B Student Black-box Distillation

## 摘要
本文在牙科选择题任务上评估了黑盒蒸馏中三类 32B->7B 配方：历史基线配方（Config A）、弱化 KL 约束配方（Config C），以及基于跨实验复发错题的监督增强配方（Config E, hard-boost）。为避免评估偏差，实验采用严格无重叠划分（train/val/test 题干去重）。在扩展实验中，Config C 评估到 5 个随机种子（2/3/4/5/6），Config E 进行 2 个种子试验（5/6）。结果显示：Config C 的 5-seed 均值为 75.66%，低于 Config A（75.90%）0.24 个百分点；Config E 的 2-seed 均值为 74.69%，低于 Config C 0.97 个百分点。三者最佳结果均未接近 32B 教师基线 83.13%。该结果表明，在当前数据规模与黑盒蒸馏设定下，弱 KL 与 hard-boost 方案均未形成可复现的跨 seed 稳定提升。

## 1. 实验设置

### 1.1 研究问题
在严格无泄漏的数据划分下，弱化 KL 约束的黑盒蒸馏配方（Config C）是否能稳定优于历史基线配方（Config A），并缩小与 32B 老师模型的差距。

### 1.2 数据与评测协议
- 数据文件：
	- 训练集：`data/cmexam_dental_choice_train_nooverlap.jsonl`（672）
	- 验证集：`data/cmexam_dental_choice_val_nooverlap.jsonl`（74）
	- 测试集：`data/cmexam_dental_choice_test.jsonl`（83）
- 划分约束：按题干文本严格去重，`train/val/test` 两两无重叠。
- 老师基线：`Qwen2.5-32B-Instruct` 直接测试。
- 学生模型：`Qwen2.5-7B-Instruct` + LoRA，训练脚本 `train_dental_lora32.py`。
- 随机种子：`2, 3, 4`。

### 1.3 对比配置
- Config A (historical)
	- `alpha=0.08`
	- `temperature=1.5`
	- `hard_upsample=1`
	- `rank=16`
	- `lora_alpha=32`
- Config C (weakKL)
	- `alpha=0.02`
	- `temperature=1.2`
	- `hard_upsample=1`
	- `rank=32`
	- `lora_alpha=64`
	- `augment=True`
- Config E (hardboost, enhanced supervision)
	- 先从 A/C 历史错题集中提取“跨实验复发错题”（在 6 份错题文件中出现次数 >=2）
	- 在训练集内对这些题目做 3x 权重（原样本 +2 次复制）得到 `train_nooverlap_hardboost`
	- 训练超参与 C 一致：`alpha=0.02`, `temperature=1.2`, `rank=32`, `lora_alpha=64`, `augment=True`

## 2. 结果

### 2.1 主结果
- Teacher 32B 直接评测：**83.13%**
- C(5-seed) 相对 A(3-seed) 的均值变化：**-0.24 pt**
- C 相对 A 的最佳提升（best-best）：**+1.21 pt**
- C 最佳结果相对 Teacher 差距：**-6.02 pt**
- E(2-seed) 均值相对 C(5-seed) 变化：**-0.97 pt**

### 2.2 分 seed 结果

| Config | Seed2 | Seed3 | Seed4 | Seed5 | Seed6 | Mean | Median | Std | Best | Gap to Teacher(best) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A(hist) | 75.90% | 75.90% | 75.90% | N/A | N/A | 75.90% | 75.90% | 0.00 | 75.90% | -7.23pt |
| C(weakKL) | 75.90% | 77.11% | 75.90% | 75.90% | 73.49% | 75.66% | 75.90% | 1.18 | 77.11% | -6.02pt |
| E(hardboost) | N/A | N/A | N/A | 75.90% | 73.49% | 74.69% | 74.69% | 1.20 | 75.90% | -7.23pt |

### 2.3 配对差值（C - A）
- Seed2: `+0.00 pt`
- Seed3: `+1.21 pt`
- Seed4: `+0.00 pt`
- Seed5: `N/A`
- Seed6: `N/A`
- 提升 seed 数（有配对子集）：`1/3`

### 2.4 扩展实验（>=5 seeds 与监督增强）
1. C 的扩展 seed（5/6）结果分别为 `75.90%` 与 `73.49%`，拉低整体均值到 `75.66%`。
2. E（hardboost）在 seed5/6 的结果为 `75.90%` 与 `73.49%`，均值 `74.69%`，未优于 C。
3. 说明“弱 KL 提升”在 3-seed 下观察到的增益不具备跨 seed 稳定复现性；hard-boost 监督增强也未带来正向收益。

### 2.5 结果解读
1. 在扩展到 >=5 seeds 后，C 的均值不再优于 A，说明此前提升主要是小样本波动。
2. E 的试点结果低于 C，提示“复发错题硬增强”在当前实现下可能引入过拟合或样本分布偏移。
3. 当前黑盒蒸馏配置（A/C/E）均未接近老师模型性能上限。

## 3. 威胁与局限

### 3.1 统计效度
- C 已扩展到 `n=5`，结果显示均值回落，验证了“3-seed 增益不稳定”的风险。
- E 目前仅 `n=2`（pilot），仍不足以下强结论，需继续扩展 seed。

### 3.2 外部效度
- 当前仅在单一牙科选择题测试集上验证，任务迁移性尚未验证。

### 3.3 构念效度
- 准确率为单指标，未纳入置信度校准、类别公平性、错误类型严重度等维度。

### 3.4 工程效度
- LoRA 容量、采样策略和训练预算仍可能限制学生模型上限。

## 4. 附录命令

### 4.1 运行脚本
```bash
bash run_compare_nooverlap_AC_multiseed.sh
bash run_followup_5seed_hardboost.sh
```

### 4.2 核验核心指标
```bash
grep -nE '正确率：' teacher32_baseline_autotest_nooverlap.log
grep -nE '测试集准确率' train32_nooverlap_hist_seed2.log train32_nooverlap_hist_seed3.log train32_nooverlap_hist_seed4.log
grep -nE '测试集准确率' train32_nooverlap_weakkl_seed2.log train32_nooverlap_weakkl_seed3.log train32_nooverlap_weakkl_seed4.log
grep -nE '测试集准确率' train32_nooverlap_weakkl_seed5.log train32_nooverlap_weakkl_seed6.log
grep -nE '测试集准确率' train32_nooverlap_hardboost_seed5.log train32_nooverlap_hardboost_seed6.log
```

### 4.3 复现实验统计
```bash
python - <<'PY'
import re, statistics as st
from pathlib import Path

def acc(path, pat):
		txt = Path(path).read_text(encoding='utf-8', errors='ignore')
		return float(re.findall(pat, txt)[-1])

A = [
		acc('train32_nooverlap_hist_seed2.log', r'测试集准确率: ([0-9.]+)%'),
		acc('train32_nooverlap_hist_seed3.log', r'测试集准确率: ([0-9.]+)%'),
		acc('train32_nooverlap_hist_seed4.log', r'测试集准确率: ([0-9.]+)%'),
]
C = [
		acc('train32_nooverlap_weakkl_seed2.log', r'测试集准确率: ([0-9.]+)%'),
		acc('train32_nooverlap_weakkl_seed3.log', r'测试集准确率: ([0-9.]+)%'),
		acc('train32_nooverlap_weakkl_seed4.log', r'测试集准确率: ([0-9.]+)%'),
]
E = [
	acc('train32_nooverlap_hardboost_seed5.log', r'测试集准确率: ([0-9.]+)%'),
	acc('train32_nooverlap_hardboost_seed6.log', r'测试集准确率: ([0-9.]+)%'),
]
T = acc('teacher32_baseline_autotest_nooverlap.log', r'正确率：([0-9.]+)%')

print('A mean/median/std/best =', sum(A)/3, st.median(A), st.pstdev(A), max(A))
print('C mean/median/std/best =', sum(C)/len(C), st.median(C), st.pstdev(C), max(C))
print('E mean/median/std/best =', sum(E)/len(E), st.median(E), st.pstdev(E), max(E))
print('C-A mean delta =', sum(C)/len(C) - sum(A)/3)
print('E-C mean delta =', sum(E)/len(E) - sum(C)/len(C))
print('C best - Teacher =', max(C) - T)
print('E best - Teacher =', max(E) - T)
PY
```

## 5. 可复现文件索引
- Teacher baseline: `teacher32_baseline_autotest_nooverlap.log`
- Config A logs:
	- `train32_nooverlap_hist_seed2.log`
	- `train32_nooverlap_hist_seed3.log`
	- `train32_nooverlap_hist_seed4.log`
- Config C logs:
	- `train32_nooverlap_weakkl_seed2.log`
	- `train32_nooverlap_weakkl_seed3.log`
	- `train32_nooverlap_weakkl_seed4.log`
 	- `train32_nooverlap_weakkl_seed5.log`
 	- `train32_nooverlap_weakkl_seed6.log`
- Config E logs:
	- `train32_nooverlap_hardboost_seed5.log`
	- `train32_nooverlap_hardboost_seed6.log`
- Runner: `run_compare_nooverlap_AC_multiseed.sh`
 - Follow-up runner: `run_followup_5seed_hardboost.sh`

## 小结
1. 在无泄漏协议下，C 的 3-seed 小幅增益在扩展到 5-seed 后未能保持，均值反而低于 A（-0.24pt）。
2. 监督增强方案 E（hardboost, pilot 2-seed）未带来提升，均值低于 C（-0.97pt）。
3. 当前最优学生结果仍为 77.11%，距教师 83.13% 仍有约 6.02pt 差距，学生反超目标未达成。
4. 下一阶段建议：继续扩展 E 到 >=5 seeds，并在 hard 样本构造中引入更强过滤（如仅保留 teacher 高置信且跨模型一致错题）后再复测。
