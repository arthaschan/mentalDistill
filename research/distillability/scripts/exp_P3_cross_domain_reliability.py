#!/usr/bin/env python3
"""
P3 预实验: 教师"不可信信号"的跨域稳定性 (CMExam中文 vs MedQA英文)。

核心问题: 教师"高熵=高错误率"这个不可信检测规律, 在中文医学和英文医学上是否一致?
若一致 → 教师诊断工具跨域稳健(正结果, 工具泛化强)。
若不一致 → 教师可靠性有域局限(诚实负结果, 也有价值)。

方法(不依赖难度标注, 两域都有熵+正确性):
对每个共同教师(Qwen32B/Phi4/Yi34B), 在两个域分别:
  1. 按熵三分档(低/中/高), 算各档错误率 → 看"高熵档错误率/低熵档错误率"的分离倍数。
  2. 算 熵 vs 正确性 的 AUC(熵作为"会答错"的检测器)。
跨域对比这两个指标, 看不可信检测能力是否迁移。

用法: python research/distillability/scripts/exp_P3_cross_domain_reliability.py
"""
import json, os
import numpy as np

REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO=os.path.dirname(REPO)
DIST=os.path.join(REPO,"research","distillability")
OUT=os.path.join(DIST,"outputs","exp_P3_cross_domain_reliability.json")
OPT=["A","B","C","D","E"]

# 共同教师: CMExam路径 vs MedQA路径
TEACHERS={
 "Qwen32B":("teacher_labels/qwen32b_train_logprobs.jsonl","teacher_labels_ext/medqa_Qwen32B_logprobs.jsonl"),
 "Phi4":("teacher_labels/phi4_train_logprobs.jsonl","teacher_labels_ext/medqa_Phi4_logprobs.jsonl"),
 "Yi34B":("teacher_labels/yi34b_train_logprobs.jsonl","teacher_labels_ext/medqa_Yi34B_logprobs.jsonl"),
}

def entropy(p):
    p=np.clip(p,1e-12,None); p=p/p.sum()
    return float(-np.sum(p*np.log(p)))

def load(path):
    rows=[]
    full=os.path.join(DIST,path)
    if not os.path.exists(full): return rows
    for line in open(full,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        dist=r.get("TeacherDist",{}); gt=str(r.get("OriginalAnswer") or r.get("Answer","")).strip().upper()
        if not dist or gt not in OPT: continue
        raw=np.array([float(dist.get(c,0.0)) for c in OPT])
        if raw.sum()<=1e-9: continue
        p=raw/raw.sum()
        ent=entropy(p); top1=OPT[int(np.argmax(p))]
        rows.append({"ent":ent,"wrong":0 if top1==gt else 1})
    return rows

def auc_ent_vs_wrong(rows):
    # 熵作为"会答错"的检测器的AUC
    ents=np.array([r["ent"] for r in rows]); ys=np.array([r["wrong"] for r in rows])
    if ys.sum()==0 or ys.sum()==len(ys): return float("nan")
    pos=ents[ys==1]; neg=ents[ys==0]
    # AUC = P(ent_pos > ent_neg)
    import random
    n=min(20000,len(pos)*len(neg))
    # 高效: 排序法
    order=np.argsort(ents); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(ents)+1)
    auc=(ranks[ys==1].sum()-pos.size*(pos.size+1)/2)/(pos.size*neg.size)
    return float(auc)

def band_sep(rows):
    ents=np.array([r["ent"] for r in rows]); ws=np.array([r["wrong"] for r in rows])
    q33,q67=np.quantile(ents,[0.33,0.67])
    lo=ws[ents<=q33].mean()*100; mid=ws[(ents>q33)&(ents<=q67)].mean()*100; hi=ws[ents>q67].mean()*100
    return lo,mid,hi

def main():
    print("="*80); print("P3: 教师不可信信号的跨域稳定性 (CMExam中文 vs MedQA英文)"); print("="*80)
    res={}
    for label,(cm,mq) in TEACHERS.items():
        cm_rows=load(cm); mq_rows=load(mq)
        if not cm_rows or not mq_rows:
            print(f"\n--- {label}: 数据缺失(CMExam={len(cm_rows)}, MedQA={len(mq_rows)}) ---"); continue
        cm_acc=(1-np.mean([r["wrong"] for r in cm_rows]))*100
        mq_acc=(1-np.mean([r["wrong"] for r in mq_rows]))*100
        cm_auc=auc_ent_vs_wrong(cm_rows); mq_auc=auc_ent_vs_wrong(mq_rows)
        cl,cm_,ch=band_sep(cm_rows); ml,mm,mh=band_sep(mq_rows)
        print(f"\n--- {label} ---")
        print(f"  整体正确率:   CMExam={cm_acc:.1f}%(n={len(cm_rows)})   MedQA={mq_acc:.1f}%(n={len(mq_rows)})")
        print(f"  熵检测AUC:    CMExam={cm_auc:.3f}              MedQA={mq_auc:.3f}    (跨域差={abs(cm_auc-mq_auc):.3f})")
        print(f"  CMExam 错误率 低/中/高熵档: {cl:.1f}% / {cm_:.1f}% / {ch:.1f}%  (高/低={ch/max(cl,0.1):.1f}×)")
        print(f"  MedQA  错误率 低/中/高熵档: {ml:.1f}% / {mm:.1f}% / {mh:.1f}%  (高/低={mh/max(ml,0.1):.1f}×)")
        res[label]={"cm_acc":round(cm_acc,2),"mq_acc":round(mq_acc,2),
            "cm_auc":round(cm_auc,4),"mq_auc":round(mq_auc,4),"auc_gap":round(abs(cm_auc-mq_auc),4),
            "cm_bands":[round(cl,2),round(cm_,2),round(ch,2)],"mq_bands":[round(ml,2),round(mm,2),round(mh,2)]}
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump(res,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[SAVED] {OUT}")
    # 判读
    if res:
        gaps=[v["auc_gap"] for v in res.values()]
        print(f"\n判读: 熵检测AUC跨域平均差异 = {np.mean(gaps):.3f}")
        print("  若AUC在两域都>0.65且差异<0.1 → 不可信检测跨域稳健(工具泛化强, 正结果)")
        print("  若某域AUC崩到~0.5或差异大 → 教师可靠性检测有域局限(诚实负结果)")

if __name__=="__main__":
    main()
