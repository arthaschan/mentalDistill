#!/usr/bin/env python3
"""
交叉验证 (go/no-go): 几何/熵标记的"不可信样本"是否富集在【人类专家标注的高难度题】?

动机: 我们的"不可信"目前用教师自己的熵定义, 风险是"熵高=不可信"近乎同义反复(就是置信度校准)。
若能证明"熵标记的不可信样本"与一个【独立的外部金标准——CMExam 的 Difficulty level(人类标注)】吻合,
则"不可信"对应真实题目难度, 不是模型主观过度自信 → 科学价值确立。

CMExam Difficulty level: 1(最易)~5(最难)。
两个独立信号:
  - 模型侧: 教师输出熵 (越高=模型越不确定=我们标记的"不可信")
  - 人类侧: Difficulty level (越高=人类认为越难)
若两者正相关 → 模型的"不可信"对应人类的"难" → 抓到真实难点(非同义反复)。

另含【跨模型一致性】: 用全部教师, 看"多个模型共同高熵"的题是否更难。
纯 CPU。用法: python research/distillability/scripts/cross_validate_difficulty.py
"""
import json, os
import numpy as np
from scipy import stats

OPTION_LETTERS=["A","B","C","D","E"]
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO=os.path.dirname(REPO)
DIST=os.path.join(REPO,"research","distillability")
LABELS=os.path.join(DIST,"teacher_labels")
OUT=os.path.join(DIST,"outputs","cross_validate_difficulty.json")

TEACHERS={
  "Qwen32B":"qwen32b_train_logprobs.jsonl","Qwen14B":"qwen14b_train_logprobs.jsonl",
  "GLM32B":"glm32b_train_logprobs.jsonl","Yi34B":"yi34b_train_logprobs.jsonl",
  "Gemma27B":"gemma27b_train_logprobs.jsonl","Phi4":"phi4_train_logprobs.jsonl",
}

def entropy(raw):
    p=np.clip(np.array(raw,dtype=np.float64),1e-12,None); p=p/p.sum()
    return float(-np.sum(p*np.log(p)))

def load(path):
    out=[]
    for line in open(path,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        dist=r.get("TeacherDist",{}); gt=str(r.get("OriginalAnswer") or r.get("Answer","")).strip().upper()
        diff=str(r.get("Difficulty level","")).strip()
        if not dist or gt not in OPTION_LETTERS or diff not in ["1","2","3","4","5"]: continue
        raw=[float(dist.get(c,0.0)) for c in OPTION_LETTERS]
        if sum(raw)<=1e-9: continue
        ta=OPTION_LETTERS[int(np.argmax(raw))]
        out.append({"ent":entropy(raw),"diff":int(diff),"correct":1 if ta==gt else 0,
                    "key":r.get("Question","")[:50]})
    return out

def main():
    results={}
    print("="*80)
    print("交叉验证: 教师熵(模型侧'不可信') vs Difficulty level(人类侧'难度')")
    print("="*80)
    per_teacher_corr=[]
    for label,fn in TEACHERS.items():
        path=os.path.join(LABELS,fn)
        if not os.path.exists(path): continue
        rows=load(path)
        ent=np.array([r["ent"] for r in rows]); diff=np.array([r["diff"] for r in rows])
        corr_acc=np.array([r["correct"] for r in rows])
        # 1) 熵 vs 难度 的 Spearman 相关
        sp=stats.spearmanr(ent,diff)
        # 2) 各难度档的平均熵 + 教师错误率
        print(f"\n--- {label} (n={len(rows)}) ---")
        print(f"  熵 vs 人类难度 Spearman ρ={sp.correlation:+.3f} (p={sp.pvalue:.2e})")
        print(f"  {'难度':<6}{'样本数':>7}{'平均熵':>9}{'教师错误率':>11}")
        band={}
        for d in [1,2,3,4,5]:
            m=diff==d
            if m.sum()==0: continue
            band[d]={"n":int(m.sum()),"mean_ent":round(float(ent[m].mean()),4),
                     "err":round(float((1-corr_acc[m]).mean()*100),2)}
            print(f"  {d:<6}{int(m.sum()):>7}{ent[m].mean():>9.3f}{(1-corr_acc[m]).mean()*100:>10.1f}%")
        results[label]={"spearman_ent_vs_difficulty":round(sp.correlation,4),
                        "p_value":float(sp.pvalue),"by_difficulty":band}
        per_teacher_corr.append(sp.correlation)

    # 3) 跨模型一致性: 每题在多少个模型上高熵(>该模型中位数), 这个"共识不确定度"vs难度
    print("\n"+"="*80)
    print("跨模型一致性: '多模型共同不确定'的题是否更难")
    print("="*80)
    # 用 Question key 对齐
    bykey={}
    for label,fn in TEACHERS.items():
        path=os.path.join(LABELS,fn)
        if not os.path.exists(path): continue
        rows=load(path)
        med=np.median([r["ent"] for r in rows])
        for r in rows:
            k=r["key"]
            bykey.setdefault(k,{"diff":r["diff"],"high_ent_count":0,"n_models":0})
            bykey[k]["n_models"]+=1
            if r["ent"]>med: bykey[k]["high_ent_count"]+=1
    # 只看所有模型都覆盖的题
    full=[v for v in bykey.values() if v["n_models"]==len(TEACHERS)]
    if full:
        consensus=np.array([v["high_ent_count"] for v in full])  # 0~6: 多少模型在此题高熵
        diffs=np.array([v["diff"] for v in full])
        sp2=stats.spearmanr(consensus,diffs)
        print(f"  覆盖全部{len(TEACHERS)}模型的题: {len(full)}")
        print(f"  '共识不确定度(高熵模型数)' vs 人类难度 Spearman ρ={sp2.correlation:+.3f} (p={sp2.pvalue:.2e})")
        print(f"  {'高熵模型数':<10}{'题数':>7}{'平均难度':>10}")
        for c in range(len(TEACHERS)+1):
            m=consensus==c
            if m.sum()==0: continue
            print(f"  {c:<10}{int(m.sum()):>7}{diffs[m].mean():>10.2f}")
        results["cross_model"]={"spearman_consensus_vs_difficulty":round(sp2.correlation,4),
                                "p_value":float(sp2.pvalue),"n_full":len(full)}

    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump(results,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[SAVED] {OUT}")
    print("\n="*1)
    print("GO/NO-GO 判读:")
    mean_corr=np.mean(per_teacher_corr)
    print(f"  6教师 熵-难度相关均值 = {mean_corr:+.3f}")
    if mean_corr>0.2:
        print("  ✅ GO: 熵标记的'不可信'与人类难度正相关 → 对应真实难点, 不是熵的同义反复。科学价值确立。")
    elif mean_corr>0.1:
        print("  ⚠️ 弱正相关: 有信号但不强, 需谨慎表述。")
    else:
        print("  ❌ NO-GO: 与人类难度几乎无关 → '不可信'只是模型主观不确定, 降级为置信度方法。")

if __name__=="__main__":
    main()
