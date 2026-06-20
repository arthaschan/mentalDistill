#!/usr/bin/env python3
"""
P3 6教师版跨域分析: 在 p3_extend_teachers.sh 补完 GLM/Gemma/Qwen14B 后自动运行。
缺失教师/域自动跳过(幂等)。结构同 exp_P3_full_crossdomain.py, 扩到6教师。
"""
import json, os
import numpy as np

REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO=os.path.dirname(REPO)
DIST=os.path.join(REPO,"research","distillability")
OUT=os.path.join(DIST,"outputs","exp_P3_full_crossdomain_6teachers.json")
OPTS=["A","B","C","D","E"]

# 6教师 × 4域。CMExam用主目录, 其余用ext目录(注意大小写命名)
def cm(p): return f"teacher_labels/{p}_train_logprobs.jsonl"
def ext(ds,t): return f"teacher_labels_ext/{ds}_{t}_logprobs.jsonl"
DATA={
 "Qwen32B":{"CMExam":cm("qwen32b"),"MedQA":ext("medqa","Qwen32B"),"MMLU-med":ext("mmlu_med","Qwen32B"),"MMLU-full":ext("mmlu_full","Qwen32B")},
 "Qwen14B":{"CMExam":cm("qwen14b"),"MedQA":ext("medqa","Qwen14B"),"MMLU-med":ext("mmlu_med","Qwen14B"),"MMLU-full":ext("mmlu_full","Qwen14B")},
 "GLM32B":{"CMExam":cm("glm32b"),"MedQA":ext("medqa","GLM32B"),"MMLU-med":ext("mmlu_med","GLM32B"),"MMLU-full":ext("mmlu_full","GLM32B")},
 "Yi34B":{"CMExam":cm("yi34b"),"MedQA":ext("medqa","Yi34B"),"MMLU-med":ext("mmlu_med","Yi34B"),"MMLU-full":ext("mmlu_full","Yi34B")},
 "Gemma27B":{"CMExam":cm("gemma27b"),"MedQA":ext("medqa","Gemma27B"),"MMLU-med":ext("mmlu_med","Gemma27B"),"MMLU-full":ext("mmlu_full","Gemma27B")},
 "Phi4":{"CMExam":cm("phi4"),"MedQA":ext("medqa","Phi4"),"MMLU-med":ext("mmlu_med","Phi4"),"MMLU-full":ext("mmlu_full","Phi4")},
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
                     "wrong":0 if OPTS[int(np.argmax(p))]==gt else 1})
    return rows

def auc(rows):
    e=np.array([r["ent"] for r in rows]); y=np.array([r["wrong"] for r in rows])
    if y.sum()==0 or y.sum()==len(y): return float("nan")
    order=np.argsort(e); ranks=np.empty(len(e)); ranks[order]=np.arange(1,len(e)+1)
    npos=y.sum(); return float((ranks[y==1].sum()-npos*(npos+1)/2)/(npos*(len(y)-npos)))

def bands(rows):
    e=np.array([r["ent"] for r in rows]); w=np.array([r["wrong"] for r in rows])
    q33,q67=np.quantile(e,[0.33,0.67])
    lo=w[e<=q33].mean()*100; hi=w[e>q67].mean()*100; return lo,hi,hi/max(lo,0.1)

def ece(rows,nb=10):
    c=np.array([r["maxp"] for r in rows]); a=np.array([1-r["wrong"] for r in rows]); e=0.0
    for i in range(nb):
        m=(c>i/nb)&(c<=(i+1)/nb)
        if m.sum(): e+=m.mean()*abs(a[m].mean()-c[m].mean())
    return float(e)

def main():
    print("="*94); print("P3 6教师 × 4域 跨域综合"); print("="*94)
    res={}
    print(f"\n{'教师':<9}{'域':<11}{'n':>6}{'正确率':>8}{'熵AUC':>8}{'低熵错%':>8}{'高熵错%':>8}{'分离×':>7}{'ECE':>7}")
    for t,doms in DATA.items():
        res[t]={}
        for d in DOMAINS:
            rows=load(doms[d])
            if len(rows)<50: continue
            acc=(1-np.mean([r["wrong"] for r in rows]))*100
            a=auc(rows); lo,hi,sep=bands(rows); e=ece(rows)
            print(f"  {t:<9}{d:<11}{len(rows):>6}{acc:>7.1f}%{a:>8.3f}{lo:>7.1f}%{hi:>7.1f}%{sep:>6.1f}×{e:>7.3f}")
            res[t][d]={"n":len(rows),"acc":round(acc,2),"auc":round(a,4),"sep":round(sep,2),"ece":round(e,4)}
    # C4 共识(用所有可用教师)
    from collections import defaultdict
    print(f"\n--- C4: 跨教师共识 vs 单教师熵 (每域) ---")
    try:
        from scipy.stats import spearmanr
        for d in DOMAINS:
            bykey=defaultdict(lambda:{"wrong":0,"n":0,"ents":[]})
            navail=0
            for t in DATA:
                rs=load(DATA[t][d])
                if rs: navail+=1
                for r in rs:
                    bykey[r["q"]]["wrong"]+=r["wrong"]; bykey[r["q"]]["n"]+=1; bykey[r["q"]]["ents"].append(r["ent"])
            full=[v for v in bykey.values() if v["n"]==navail and navail>=2]
            if len(full)<50: continue
            rho=spearmanr([np.mean(v["ents"]) for v in full],[v["wrong"] for v in full]).correlation
            print(f"  {d:<11} 教师数={navail} 对齐题={len(full):>5}  平均熵↔共识错误 ρ={rho:+.3f}")
            res.setdefault("C4",{})[d]={"n_teachers":navail,"n":len(full),"rho":round(float(rho),4)}
    except ImportError:
        print("  (scipy缺, 跳过C4)")
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump(res,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[SAVED] {OUT}")
    print("\n判读:")
    for t in DATA:
        aucs=[res[t][d]["auc"] for d in DOMAINS if d in res.get(t,{})]
        if len(aucs)>=2: print(f"  {t:<9} 熵AUC跨{len(aucs)}域: {min(aucs):.3f}~{max(aucs):.3f}")

if __name__=="__main__":
    main()
