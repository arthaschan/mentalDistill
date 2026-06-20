#!/usr/bin/env python3
"""
实验A-labelfree: 完全不用 GT、不训练分类器, 纯几何单特征能否识别"教师在猜"的样本。

对比实验A(用5折CV分类器, 训练时碰了对错标签):
  本脚本直接用【单个几何特征】排序分档, 完全无监督(label-free)。
  若纯熵/纯margin 分档也能让"低可信档"教师错误率 >> "高可信档",
  则证明"几何识别教师不可信样本"是真正 training-free 的, 不依赖任何标签。

几何特征的先验方向(无需学习):
  - entropy 高   -> 教师不确定 -> 可能在猜 (可信度 = -entropy)
  - margin 低    -> top1/top2 接近 -> 可能在猜 (可信度 = margin)
  - peak 低      -> top1 概率低 -> 可能在猜 (可信度 = peak)
  - logdet_g 高  -> 分布"摊开" -> 可能在猜 (可信度 = -logdet_g)

用法: python research/distillability/scripts/expA_labelfree.py
"""
import json, os
import numpy as np

OPTION_LETTERS=["A","B","C","D","E"]
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO=os.path.dirname(REPO)
DIST=os.path.join(REPO,"research","distillability")
LABELS=os.path.join(DIST,"teacher_labels")
OUT=os.path.join(DIST,"outputs","expA_labelfree.json")

TEACHERS={
  "Qwen32B":"qwen32b_train_logprobs.jsonl",
  "Qwen14B":"qwen14b_train_logprobs.jsonl",
  "GLM32B":"glm32b_train_logprobs.jsonl",
  "Yi34B":"yi34b_train_logprobs.jsonl",
  "Gemma27B":"gemma27b_train_logprobs.jsonl",
  "Phi4":"phi4_train_logprobs.jsonl",
}

def feats(raw):
    p=np.clip(np.array(raw,dtype=np.float64),1e-12,None); p=p/p.sum()
    srt=np.sort(p)[::-1]
    return {
      "neg_entropy": float(np.sum(p*np.log(p))),     # = -entropy, 越大越可信
      "margin": float(srt[0]-srt[1]),                # 越大越可信
      "peak": float(srt[0]),                         # 越大越可信
      "neg_logdet_g": float(np.sum(np.log(p))),      # = -logdet近似, 越大越可信
    }

def load(path):
    rows=[]
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
        f=feats(raw); f["correct"]=1 if ta==gt else 0
        rows.append(f)
    return rows

def auc(scores, labels):
    """无监督单特征 vs 对错的 AUC (只用于事后评估, 不参与分档)。"""
    s=np.array(scores); y=np.array(labels)
    npos=y.sum(); nneg=len(y)-npos
    if npos==0 or nneg==0: return None
    order=np.argsort(s); ranks=np.empty(len(s)); ranks[order]=np.arange(1,len(s)+1)
    return float((ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))

def main():
    SINGLE=["neg_entropy","margin","peak","neg_logdet_g"]
    results={}
    print("="*86)
    print("实验A-labelfree: 纯几何单特征(无训练/无GT)分档 vs 教师真实错误率")
    print("="*86)
    for label,fn in TEACHERS.items():
        path=os.path.join(LABELS,fn)
        if not os.path.exists(path): continue
        rows=load(path)
        y=np.array([r["correct"] for r in rows])
        overall_err=float((1-y).mean()*100)
        print(f"\n--- {label} (整体错误率 {overall_err:.1f}%, n={len(rows)}) ---")
        results[label]={"overall_error":round(overall_err,2),"features":{}}
        for feat in SINGLE:
            s=np.array([r[feat] for r in rows])
            # 按该特征三等分(低可信=特征值最低的1/3)
            q1,q2=np.quantile(s,[1/3,2/3])
            lo=(1-y[s<=q1]).mean()*100
            hi=(1-y[s>q2]).mean()*100
            a=auc(s,y)
            sep = lo/hi if hi>0 else float('inf')
            results[label]["features"][feat]={"low_cred_err":round(float(lo),2),
                "high_cred_err":round(float(hi),2),"auc":round(a,4) if a else None}
            print(f"  {feat:14s} 低可信档错误率={lo:5.1f}%  高可信档错误率={hi:5.1f}%  AUC={a:.3f}")
    json.dump(results,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[SAVED] {OUT}")
    print("\n判读: 纯单特征(完全无监督)若仍让 低可信档错误率 >> 高可信档, 则 label-free 识别成立。")
    print("      对比实验A(训练分类器, 用了GT): 看无监督版能达到多少 — 这是诚实的 training-free 证据。")

if __name__=="__main__":
    main()
