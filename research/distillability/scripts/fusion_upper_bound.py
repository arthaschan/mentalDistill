#!/usr/bin/env python3
"""
增益上界估算 (纯分析, 不训练学生): 领域路由融合 vs 单一最强教师, 教师准确率差多少。

逻辑: 学生蒸馏后准确率上限 ≈ 教师软标签的准确率(学生学不过教师标签)。
  - 单一最强教师: 全部样本用同一个最强教师的答案。
  - 领域路由融合: 每个样本用"它所在学科里准确率最高的教师"的答案 (oracle 路由)。
  - 跨模型 oracle 上界: 每个样本只要"有任一教师答对"就算对 (理论天花板)。
若融合准确率比单一最强教师高很多 -> 值得做完整融合蒸馏; 若只高零点几个点 -> 增益有限。

纯 CPU。用法: python research/distillability/scripts/fusion_upper_bound.py
"""
import json, os
import numpy as np
from collections import defaultdict

OPTION_LETTERS=["A","B","C","D","E"]
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO=os.path.dirname(REPO)
DIST=os.path.join(REPO,"research","distillability")
LABELS=os.path.join(DIST,"teacher_labels")
OUT=os.path.join(DIST,"outputs","fusion_upper_bound.json")

TEACHERS={
  "Qwen32B":"qwen32b_train_logprobs.jsonl","Qwen14B":"qwen14b_train_logprobs.jsonl",
  "GLM32B":"glm32b_train_logprobs.jsonl","Yi34B":"yi34b_train_logprobs.jsonl",
  "Gemma27B":"gemma27b_train_logprobs.jsonl","Phi4":"phi4_train_logprobs.jsonl",
}

def load(path):
    """返回 {qkey: {"correct":0/1,"domain":str}}"""
    out={}
    for line in open(path,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        dist=r.get("TeacherDist",{}); gt=str(r.get("OriginalAnswer") or r.get("Answer","")).strip().upper()
        if not dist or gt not in OPTION_LETTERS: continue
        raw=[float(dist.get(c,0.0)) for c in OPTION_LETTERS]
        if sum(raw)<=1e-9: continue
        ta=OPTION_LETTERS[int(np.argmax(raw))]
        qkey=r.get("Question","")[:80]
        out[qkey]={"correct":1 if ta==gt else 0,
                   "domain":str(r.get("Medical Discipline","")).strip() or "unknown"}
    return out

def main():
    data={}
    for label,fn in TEACHERS.items():
        p=os.path.join(LABELS,fn)
        if os.path.exists(p): data[label]=load(p)
    teachers=list(data.keys())
    # 对齐: 所有教师都覆盖的题
    common=set.intersection(*[set(d.keys()) for d in data.values()])
    common=sorted(common)
    n=len(common)
    print(f"对齐题数(所有{len(teachers)}教师都覆盖): {n}")

    # 各教师整体准确率
    acc={t:np.mean([data[t][q]["correct"] for q in common])*100 for t in teachers}
    best_teacher=max(acc,key=acc.get)
    print("\n各教师整体准确率(对齐集):")
    for t in sorted(acc,key=lambda x:-acc[x]): print(f"  {t:10s}{acc[t]:6.2f}%")
    print(f"\n单一最强教师 = {best_teacher} ({acc[best_teacher]:.2f}%)  ← 基线")

    # 领域(用 best_teacher 的 domain 标注, 各教师 domain 应一致)
    domain_of={q:data[best_teacher][q]["domain"] for q in common}
    domains=sorted(set(domain_of.values()))

    # 每个领域里, 哪个教师准确率最高 (oracle 领域路由)
    dom_best={}
    for d in domains:
        qs=[q for q in common if domain_of[q]==d]
        if len(qs)<20: continue
        dacc={t:np.mean([data[t][q]["correct"] for q in qs])*100 for t in teachers}
        dom_best[d]=(max(dacc,key=dacc.get),dacc)

    # 1) 领域路由融合准确率: 每题用其领域最优教师
    routed_correct=[]
    for q in common:
        d=domain_of[q]
        if d in dom_best:
            t=dom_best[d][0]
        else:
            t=best_teacher
        routed_correct.append(data[t][q]["correct"])
    routed_acc=np.mean(routed_correct)*100

    # 2) 跨模型 oracle 上界: 任一教师答对即对
    any_correct=[max(data[t][q]["correct"] for t in teachers) for q in common]
    oracle_acc=np.mean(any_correct)*100

    print("\n"+"="*60)
    print("增益上界估算")
    print("="*60)
    print(f"  单一最强教师({best_teacher}):      {acc[best_teacher]:6.2f}%")
    print(f"  领域路由融合(每领域选最优):        {routed_acc:6.2f}%   (Δ={routed_acc-acc[best_teacher]:+.2f}pp)")
    print(f"  跨模型oracle上界(任一答对即对):    {oracle_acc:6.2f}%   (Δ={oracle_acc-acc[best_teacher]:+.2f}pp)")

    print("\n各领域最优教师(看是否真的换了教师):")
    print(f"  {'领域':<14}{'最优教师':<10}{'该领域准确率':>12}{'最强教师在此':>14}")
    for d,(bt,dacc) in dom_best.items():
        print(f"  {d:<14}{bt:<10}{dacc[bt]:>11.1f}%{dacc[best_teacher]:>13.1f}%")

    res={"n_aligned":n,"teacher_acc":{t:round(acc[t],2) for t in teachers},
         "best_teacher":best_teacher,"best_acc":round(acc[best_teacher],2),
         "routed_fusion_acc":round(routed_acc,2),
         "routed_gain":round(routed_acc-acc[best_teacher],2),
         "cross_model_oracle_acc":round(oracle_acc,2),
         "oracle_gain":round(oracle_acc-acc[best_teacher],2),
         "domain_best_teacher":{d:bt for d,(bt,_) in dom_best.items()}}
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump(res,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[SAVED] {OUT}")

    print("\n判读:")
    g=routed_acc-acc[best_teacher]
    if g>=2: print(f"  ✅ 领域路由融合 +{g:.2f}pp >> 0 → 值得做完整融合蒸馏, 价值大。")
    elif g>=0.5: print(f"  ⚠️ 融合 +{g:.2f}pp → 有限增益, 可做但非主卖点。")
    else: print(f"  ❌ 融合仅 +{g:.2f}pp → 增益太小, 诚实结论'互补存在但融合无实质收益', 不值得当主线。")
    print(f"  注: oracle上界 +{oracle_acc-acc[best_teacher]:.2f}pp 是理论天花板(需完美路由器才达到)。")

if __name__=="__main__":
    main()
