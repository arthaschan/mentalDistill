#!/usr/bin/env python3
"""
C + D 合并分析 (纯CPU, 不抢GPU):

C — 难度感知前提 (为 D1×D2 协同 H6 打基础):
   验证"题目人类难度越高, 模型越难区分(熵越高、跨模型共识错误越多)"。
   若成立, 则'任务难度'可作为'所需学生容量'的预测特征 (H6前提)。

D — 题目客观属性独立验证 (给 5d 再加一道不依赖任何模型的证据):
   算每题的客观属性(题干长度、选项数、是否含否定词/多步推理标志),
   验证"不可信样本(高熵)"是否在这些【完全不依赖模型】的客观属性上系统偏难。
   这是比"人类难度标注"更独立的第三方验证维度。

用法: python research/distillability/scripts/exp_CD_difficulty_objective.py
"""
import json, os, re
import numpy as np
from scipy import stats

OPT=["A","B","C","D","E"]
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO=os.path.dirname(REPO)
DIST=os.path.join(REPO,"research","distillability")
LABELS=os.path.join(DIST,"teacher_labels")
OUT=os.path.join(DIST,"outputs","exp_CD_difficulty_objective.json")

TEACHERS={"Qwen32B":"qwen32b_train_logprobs.jsonl","Qwen14B":"qwen14b_train_logprobs.jsonl",
  "GLM32B":"glm32b_train_logprobs.jsonl","Yi34B":"yi34b_train_logprobs.jsonl",
  "Gemma27B":"gemma27b_train_logprobs.jsonl","Phi4":"phi4_train_logprobs.jsonl"}

# 否定/多步推理标志词(中文医学题常见)
NEG_WORDS=["不属于","不是","错误","除外","不正确","不包括","不能","不会","禁忌","不宜","以下哪项不"]
MULTI_WORDS=["首先","其次","最可能","最佳","最合适","综合","首选","下一步","进一步"]

def entropy(raw):
    p=np.clip(np.array(raw,dtype=np.float64),1e-12,None); p=p/p.sum()
    return float(-np.sum(p*np.log(p)))

def objective_feats(q, opts):
    """完全不依赖任何模型的题目客观属性。"""
    qlen=len(q)
    nopt=len([k for k in opts if opts.get(k)]) if isinstance(opts,dict) else 0
    neg=1 if any(w in q for w in NEG_WORDS) else 0
    multi=1 if any(w in q for w in MULTI_WORDS) else 0
    return qlen, nopt, neg, multi

def load(path):
    out=[]
    for line in open(path,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        dist=r.get("TeacherDist",{}); gt=str(r.get("OriginalAnswer") or r.get("Answer","")).strip().upper()
        diff=str(r.get("Difficulty level","")).strip()
        if not dist or gt not in OPT: continue
        raw=[float(dist.get(c,0.0)) for c in OPT]
        if sum(raw)<=1e-9: continue
        ta=OPT[int(np.argmax(raw))]
        q=r.get("Question",""); opts=r.get("Options",{})
        qlen,nopt,neg,multi=objective_feats(q,opts)
        out.append({"ent":entropy(raw),"correct":1 if ta==gt else 0,
                    "diff":int(diff) if diff in ["1","2","3","4","5"] else None,
                    "key":q[:80],"qlen":qlen,"neg":neg,"multi":multi})
    return out

def sp(a,b):
    r=stats.spearmanr(a,b); return float(r.correlation),float(r.pvalue)

def main():
    res={}
    # === C: 难度 vs 跨模型共识错误 ===
    print("="*78); print("C — 难度感知前提: 人类难度 vs 跨模型共识错误率"); print("="*78)
    bykey={}
    for label,fn in TEACHERS.items():
        p=os.path.join(LABELS,fn)
        if not os.path.exists(p): continue
        for r in load(p):
            k=r["key"]
            bykey.setdefault(k,{"diff":r["diff"],"wrong":0,"n":0,"qlen":r["qlen"],"neg":r["neg"],"multi":r["multi"],"ents":[]})
            bykey[k]["n"]+=1; bykey[k]["wrong"]+=(1-r["correct"]); bykey[k]["ents"].append(r["ent"])
    full=[v for v in bykey.values() if v["n"]==len(TEACHERS) and v["diff"]]
    print(f"  覆盖全部{len(TEACHERS)}模型且有难度的题: {len(full)}")
    diffs=[v["diff"] for v in full]
    nwrong=[v["wrong"] for v in full]   # 多少模型在此题错
    mean_ent=[float(np.mean(v["ents"])) for v in full]
    rho1,p1=sp(diffs,nwrong); rho2,p2=sp(diffs,mean_ent)
    print(f"  人类难度 vs '共识错误数(0-6)': ρ={rho1:+.3f} (p={p1:.1e})")
    print(f"  人类难度 vs '平均熵':          ρ={rho2:+.3f} (p={p2:.1e})")
    print(f"  {'难度':<5}{'题数':>6}{'平均共识错误数':>14}{'平均熵':>9}")
    for d in [1,2,3,4,5]:
        sub=[v for v in full if v["diff"]==d]
        if not sub: continue
        print(f"  {d:<5}{len(sub):>6}{np.mean([v['wrong'] for v in sub]):>14.2f}{np.mean([np.mean(v['ents']) for v in sub]):>9.3f}")
    res["C_difficulty_vs_consensus_error"]={"rho_diff_vs_nwrong":round(rho1,4),"rho_diff_vs_entropy":round(rho2,4),"n":len(full)}

    # === D: 客观属性 vs 不可信(高熵/出错) ===
    print("\n"+"="*78); print("D — 题目客观属性(不依赖模型) vs 不可信"); print("="*78)
    qlen=[v["qlen"] for v in full]; neg=[v["neg"] for v in full]; multi=[v["multi"] for v in full]
    # 客观属性 vs 共识错误数
    print("  题目客观属性 vs '共识错误数':")
    for name,feat in [("题干长度",qlen),("含否定词",neg),("含多步推理词",multi)]:
        r,pp=sp(feat,nwrong)
        print(f"    {name:<12} ρ={r:+.3f} (p={pp:.1e})")
        res.setdefault("D_objective",{})[name]=round(r,4)
    # 否定词题 vs 普通题的错误率对比
    neg_err=np.mean([v["wrong"]/v["n"] for v in full if v["neg"]])*100
    pos_err=np.mean([v["wrong"]/v["n"] for v in full if not v["neg"]])*100
    print(f"  含否定词题 共识错误率={neg_err:.1f}% vs 普通题={pos_err:.1f}%  (Δ={neg_err-pos_err:+.1f}pp)")
    res["D_neg_vs_normal_error"]={"neg":round(neg_err,2),"normal":round(pos_err,2)}

    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump(res,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[SAVED] {OUT}")
    print("\n判读:")
    print(f"  C: 难度↔共识错误 ρ={rho1:.2f} → 越难的题模型越一致地错 (H6前提: 难度可作容量预测特征)")
    print(f"  D: 若客观属性(题长/否定词)也与'不可信'正相关 → 不依赖任何模型的第三方验证, 比熵更独立")

if __name__=="__main__":
    main()
