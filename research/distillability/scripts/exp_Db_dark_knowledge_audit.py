#!/usr/bin/env python3
"""
D-b: 暗知识质量审计 (纯CPU, go/no-go预实验) —— 证伪"软标签暗知识总是有益"。

Hinton信仰: 教师软标签里"非最高概率选项"的概率分布(暗知识/dark knowledge)编码了类间关系, 对学生有益。
我们的质疑: 在教师【高熵(在猜)】的样本上, 这些暗知识可能是噪声——把概率质量分给了错误选项, 误导学生。

审计方法(不训练, 纯分析教师软标签):
对每个样本, 看教师软标签的"次要概率质量"(1 - max_prob, 即暗知识部分)主要流向:
  - 正确答案? (暗知识有益: 即使没选对, 也把概率倾向正确答案)
  - 错误答案? (暗知识有害: 概率分给了错的, 会误导学生)
按教师熵分档, 看高熵档的暗知识是否系统性地指向错误。

关键指标: "暗知识命中率" = 在教师没把最高概率给正确答案时, 次高概率是否是正确答案。
若高熵样本的暗知识命中率显著低 → 证明高熵样本的暗知识是噪声 → 证伪"暗知识总是有益"。

用法: python research/distillability/scripts/exp_Db_dark_knowledge_audit.py
"""
import json, os
import numpy as np

OPT=["A","B","C","D","E"]
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO=os.path.dirname(REPO)
DIST=os.path.join(REPO,"research","distillability")
LABELS=os.path.join(DIST,"teacher_labels")
OUT=os.path.join(DIST,"outputs","exp_Db_dark_knowledge_audit.json")

TEACHERS={"Qwen32B":"qwen32b_train_logprobs.jsonl","Qwen14B":"qwen14b_train_logprobs.jsonl",
  "GLM32B":"glm32b_train_logprobs.jsonl","Yi34B":"yi34b_train_logprobs.jsonl",
  "Gemma27B":"gemma27b_train_logprobs.jsonl","Phi4":"phi4_train_logprobs.jsonl"}

def entropy(p):
    p=np.clip(p,1e-12,None); p=p/p.sum()
    return float(-np.sum(p*np.log(p)))

def load(path):
    out=[]
    for line in open(path,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        dist=r.get("TeacherDist",{}); gt=str(r.get("OriginalAnswer") or r.get("Answer","")).strip().upper()
        if not dist or gt not in OPT: continue
        raw=np.array([float(dist.get(c,0.0)) for c in OPT])
        if raw.sum()<=1e-9: continue
        p=raw/raw.sum(); gi=OPT.index(gt)
        order=np.argsort(p)[::-1]  # 概率从高到低的选项索引
        top1=order[0]; top2=order[1]
        out.append({
            "ent":entropy(p),"p":p,"gi":gi,"top1":top1,"top2":top2,
            "correct":1 if top1==gi else 0,
            "darkmass":float(1.0-p[top1]),                 # 暗知识总质量(非top1的概率和)
            "dark_to_correct":float(p[gi]) if top1!=gi else 0.0,  # 暗知识里流向正确答案的质量(仅当top1错时有意义)
            "top2_is_correct":1 if (top1!=gi and top2==gi) else 0, # top1错时, 暗知识(次高)是否救回正确
        })
    return out

def main():
    print("="*80)
    print("D-b 暗知识质量审计: 高熵样本的'暗知识'是有益还是噪声?")
    print("="*80)
    res={}
    for label,fn in TEACHERS.items():
        path=os.path.join(LABELS,fn)
        if not os.path.exists(path): continue
        rows=load(path)
        ent=np.array([r["ent"] for r in rows])
        # 按熵三分: 低/中/高
        q33,q67=np.quantile(ent,[0.33,0.67])
        bands={"低熵(可信)":ent<=q33,"中熵":(ent>q33)&(ent<=q67),"高熵(在猜)":ent>q67}
        print(f"\n--- {label} (n={len(rows)}) ---")
        print(f"  {'熵档':<12}{'样本':>6}{'top1错误率':>11}{'暗知识命中率*':>14}{'暗知识质量':>11}")
        band_stat={}
        for bname,mask in bands.items():
            sub=[rows[i] for i in range(len(rows)) if mask[i]]
            if not sub: continue
            err=np.mean([1-r["correct"] for r in sub])*100
            # 暗知识命中率: 在top1错的样本里, 次高概率(暗知识首选)命中正确答案的比例
            wrong=[r for r in sub if r["correct"]==0]
            dark_hit=np.mean([r["top2_is_correct"] for r in wrong])*100 if wrong else 0.0
            darkmass=np.mean([r["darkmass"] for r in sub])
            print(f"  {bname:<12}{len(sub):>6}{err:>10.1f}%{dark_hit:>13.1f}%{darkmass:>11.3f}")
            band_stat[bname]={"n":len(sub),"err":round(err,2),"dark_hit_rate":round(dark_hit,2),"dark_mass":round(float(darkmass),4)}
        res[label]=band_stat
    print("\n  *暗知识命中率 = 教师top1答错时, 次高概率选项正好是正确答案的比例(暗知识'救回'的能力)")
    print("   随机基线(5选项, top1错后剩4选项里猜): ~25%")

    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump(res,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[SAVED] {OUT}")

    # 判读: 跨教师看 低熵vs高熵 的暗知识命中率
    print("\n判读(go/no-go):")
    print(f"  {'教师':<10}{'低熵暗知识命中':>14}{'高熵暗知识命中':>14}{'差值':>8}")
    deltas=[]
    for label,bs in res.items():
        lo=bs.get("低熵(可信)",{}).get("dark_hit_rate")
        hi=bs.get("高熵(在猜)",{}).get("dark_hit_rate")
        if lo is not None and hi is not None:
            print(f"  {label:<10}{lo:>13.1f}%{hi:>13.1f}%{hi-lo:>+7.1f}")
            deltas.append(hi-lo)
    if deltas:
        md=np.mean(deltas)
        print(f"\n  平均差值(高熵-低熵暗知识命中率) = {md:+.1f}pp")
        if md < -5:
            print("  ✅ 证伪'暗知识总有益': 高熵样本暗知识命中率显著更低 → 高熵暗知识是噪声, 不能救回正确答案")
        elif md < 0:
            print("  ⚠️ 弱趋势: 高熵暗知识略差, 但不强")
        else:
            print("  ❌ 未证伪: 高熵暗知识命中率不低于低熵")

if __name__=="__main__":
    main()
