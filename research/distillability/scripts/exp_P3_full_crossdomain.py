#!/usr/bin/env python3
"""
P3 核心分析 (C1-C5, 纯CPU): 教师不可信检测的跨域稳定性。
3共同教师(Qwen32B/Phi4/Yi34B) × 4域(CMExam中文 / MedQA英文医 / MMLU-med英文医 / MMLU-full英文通用)。

C1: 熵检测AUC跨域 (熵作"会答错"检测器)
C3: 检测锐利度(高/低熵档错误率分离倍数)的域依赖 + 与域内正确率关系
C4: 跨模型共识做难度代理 — 熵 vs 跨3教师共识错误数, 跨域相关稳定性
C5: 校准 — 教师置信度(max prob) vs 实际正确率, ECE跨域
(C2子集重叠需题目对齐, 跨数据集题目不同, 仅在同数据集内同教师不同采样适用 → 跳过,留GPU阶段做同域稳定性)
"""
import json, os
import numpy as np

REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO=os.path.dirname(REPO)
DIST=os.path.join(REPO,"research","distillability")
OUT=os.path.join(DIST,"outputs","exp_P3_full_crossdomain.json")
OPTS=["A","B","C","D","E"]

# 教师 -> {域: 路径}
DATA={
 "Qwen32B":{
   "CMExam":"teacher_labels/qwen32b_train_logprobs.jsonl",
   "MedQA":"teacher_labels_ext/medqa_Qwen32B_logprobs.jsonl",
   "MMLU-med":"teacher_labels_ext/mmlu_med_Qwen32B_logprobs.jsonl",
   "MMLU-full":"teacher_labels_ext/mmlu_full_Qwen32B_logprobs.jsonl"},
 "Phi4":{
   "CMExam":"teacher_labels/phi4_train_logprobs.jsonl",
   "MedQA":"teacher_labels_ext/medqa_Phi4_logprobs.jsonl",
   "MMLU-med":"teacher_labels_ext/mmlu_med_Phi4_logprobs.jsonl",
   "MMLU-full":"teacher_labels_ext/mmlu_full_Phi4_logprobs.jsonl"},
 "Yi34B":{
   "CMExam":"teacher_labels/yi34b_train_logprobs.jsonl",
   "MedQA":"teacher_labels_ext/medqa_Yi34B_logprobs.jsonl",
   "MMLU-med":"teacher_labels_ext/mmlu_med_Yi34B_logprobs.jsonl",
   "MMLU-full":"teacher_labels_ext/mmlu_full_Yi34B_logprobs.jsonl"},
}
DOMAINS=["CMExam","MedQA","MMLU-med","MMLU-full"]

def entropy(p):
    p=np.clip(p,1e-12,None); p=p/p.sum()
    return float(-np.sum(p*np.log(p)))

def load(path):
    rows=[]; full=os.path.join(DIST,path)
    if not os.path.exists(full): return rows
    for line in open(full,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        dist=r.get("TeacherDist",{}); gt=str(r.get("OriginalAnswer") or r.get("Answer","")).strip().upper()
        if not dist or gt not in OPTS: continue
        raw=np.array([float(dist.get(c,0.0)) for c in OPTS])
        if raw.sum()<=1e-9: continue
        p=raw/raw.sum()
        rows.append({"q":r.get("Question","")[:80],"ent":entropy(p),"maxp":float(p.max()),
                     "pred":OPTS[int(np.argmax(p))],"gt":gt,"wrong":0 if OPTS[int(np.argmax(p))]==gt else 1})
    return rows

def auc(rows):
    ents=np.array([r["ent"] for r in rows]); ys=np.array([r["wrong"] for r in rows])
    if ys.sum()==0 or ys.sum()==len(ys): return float("nan")
    order=np.argsort(ents); ranks=np.empty(len(ents)); ranks[order]=np.arange(1,len(ents)+1)
    npos=ys.sum(); nneg=len(ys)-npos
    return float((ranks[ys==1].sum()-npos*(npos+1)/2)/(npos*nneg))

def bands(rows):
    e=np.array([r["ent"] for r in rows]); w=np.array([r["wrong"] for r in rows])
    q33,q67=np.quantile(e,[0.33,0.67])
    lo=w[e<=q33].mean()*100; hi=w[e>q67].mean()*100
    return lo,hi,(hi/max(lo,0.1))

def ece(rows,nb=10):
    conf=np.array([r["maxp"] for r in rows]); acc=np.array([1-r["wrong"] for r in rows])
    e=0.0
    for i in range(nb):
        m=(conf>i/nb)&(conf<=(i+1)/nb)
        if m.sum()==0: continue
        e+=m.mean()*abs(acc[m].mean()-conf[m].mean())
    return float(e)

def main():
    print("="*92)
    print("P3 跨域综合: 3教师 × 4域  教师不可信检测稳定性")
    print("="*92)
    res={}
    # C1+C3+C5 主表
    print(f"\n{'教师':<9}{'域':<11}{'n':>6}{'正确率':>8}{'熵AUC':>8}{'低熵错%':>8}{'高熵错%':>8}{'分离×':>7}{'ECE':>7}")
    for t,doms in DATA.items():
        res[t]={}
        for d in DOMAINS:
            rows=load(doms[d])
            if len(rows)<50: 
                print(f"  {t:<9}{d:<11} 数据不足({len(rows)})"); continue
            acc=(1-np.mean([r["wrong"] for r in rows]))*100
            a=auc(rows); lo,hi,sep=bands(rows); e=ece(rows)
            print(f"  {t:<9}{d:<11}{len(rows):>6}{acc:>7.1f}%{a:>8.3f}{lo:>7.1f}%{hi:>7.1f}%{sep:>6.1f}×{e:>7.3f}")
            res[t][d]={"n":len(rows),"acc":round(acc,2),"auc":round(a,4),"lo_err":round(lo,2),"hi_err":round(hi,2),"sep":round(sep,2),"ece":round(e,4)}
    # C4: 跨模型共识难度代理(每域: 用3教师, 共识错误数 vs 单教师熵)
    print(f"\n--- C4: 跨3教师共识 vs 单教师熵 (每域内, 共识难度代理) ---")
    from collections import defaultdict
    for d in DOMAINS:
        # 按题聚合3教师
        bykey=defaultdict(lambda:{"wrong":0,"n":0,"ents":[]})
        for t in DATA:
            for r in load(DATA[t][d]):
                k=r["q"]; bykey[k]["wrong"]+=r["wrong"]; bykey[k]["n"]+=1; bykey[k]["ents"].append(r["ent"])
        full=[v for v in bykey.values() if v["n"]==3]
        if len(full)<50: print(f"  {d}: 对齐题不足({len(full)})"); continue
        nwrong=np.array([v["wrong"] for v in full]); ment=np.array([np.mean(v["ents"]) for v in full])
        from scipy.stats import spearmanr
        rho=spearmanr(ment,nwrong).correlation
        print(f"  {d:<11} 对齐题={len(full):>5}  平均熵↔共识错误数 ρ={rho:+.3f}")
        res.setdefault("C4_consensus",{})[d]={"n":len(full),"rho_ent_vs_consensus_wrong":round(float(rho),4)}

    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump(res,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[SAVED] {OUT}")
    # 判读: 每教师的AUC跨域range
    print("\n判读(检测稳定性):")
    for t in DATA:
        aucs=[res[t][d]["auc"] for d in DOMAINS if d in res[t] and "auc" in res[t][d]]
        if aucs: print(f"  {t:<9} 熵AUC跨{len(aucs)}域: {min(aucs):.3f}~{max(aucs):.3f} (range={max(aucs)-min(aucs):.3f})")

if __name__=="__main__":
    main()
