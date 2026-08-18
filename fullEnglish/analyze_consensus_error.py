#!/usr/bin/env python3
"""分析训练集上 flash / Llama70B / Qwen3-32B 三个模型的逐题对错分布。

回答：答错的题是不是"哪个老师都答错"？多老师学习的收益上限是多少？
关键量：个体准确率、并集准确率(至少一个对)、共识错误(全错)、错误重叠率。
"""
import json
from collections import Counter

def load_answers(path, ta_field="TeacherAnswer", gt_field="OriginalAnswer", uid_field="uid"):
    d = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        uid = r.get(uid_field)
        if not uid:
            continue
        ta = str(r.get(ta_field, "")).strip().upper()
        gt = str(r.get(gt_field) or r.get("Answer", "")).strip().upper()
        if gt in "ABCDE" and ta in "ABCDE":
            d[uid] = (ta, gt)
    return d

flash = load_answers("fullEnglish/03_main_distill/labels/teacher_train.jsonl")
llama = load_answers("fullEnglish/03_main_distill/labels/teacher_train_llama70b.jsonl")
qwen3 = load_answers("fullEnglish/03_main_distill/data/qwen3_train_logprobs.jsonl")

models = {"flash": flash, "llama70b": llama, "qwen3": qwen3}

# 三模型都有答案的 uid 交集
common = set(flash) & set(llama) & set(qwen3)
print(f"三模型共有的题数: {len(common)}")

# 个体准确率
print("\n=== 个体准确率(训练集) ===")
acc = {}
for name, d in models.items():
    c = sum(1 for u in common if d[u][0] == d[u][1])
    acc[name] = round(100.0 * c / len(common), 2)
    print(f"  {name:9s}: {acc[name]}%")

# 每题做对/做错的模型数
wrong_count = Counter()
union_correct = 0   # 至少一个做对
all_wrong = 0       # 全部做错(共识错误)
for u in common:
    wrongs = sum(1 for name in models if models[name][u][0] != models[name][u][1])
    wrong_count[wrongs] += 1
    if wrongs < 3:
        union_correct += 1
    if wrongs == 3:
        all_wrong += 1

print("\n=== 每题做错模型数分布 ===")
for k in range(4):
    n = wrong_count.get(k, 0)
    print(f"  {k} 个模型做错: {n} 题 ({100.0*n/len(common):.1f}%)")

union_acc = round(100.0 * union_correct / len(common), 2)
best = max(acc.values())
print(f"\n=== 多老师收益上限 ===")
print(f"  并集准确率(至少一个对): {union_acc}%")
print(f"  最强单个模型: {best}%")
print(f"  多老师理论上限增益: +{union_acc - best:.2f}pp")
print(f"  共识错误(全错)比例: {100.0*all_wrong/len(common):.1f}% ({all_wrong} 题)")

# 错误重叠：flash 做错的题里，另外两个模型也错的概率
print("\n=== 错误重叠（flash 做错时，其它模型也错的概率）===")
flash_wrong = [u for u in common if flash[u][0] != flash[u][1]]
for other in ["llama70b", "qwen3"]:
    both_wrong = sum(1 for u in flash_wrong if models[other][u][0] != models[other][u][1])
    print(f"  flash 做错 {len(flash_wrong)} 题，其中 {other} 也做错 {both_wrong} 题 "
          f"({100.0*both_wrong/len(flash_wrong):.1f}%)")
# 三个都错（flash 做错时）
all3_wrong = sum(1 for u in flash_wrong if llama[u][0] != llama[u][1] and qwen3[u][0] != qwen3[u][1])
print(f"  flash 做错时，三个模型都错的比例: {100.0*all3_wrong/len(flash_wrong):.1f}%")
