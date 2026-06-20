#!/usr/bin/env python3
"""
实验 A + B: 强教师内部的细粒度可蒸馏性 (回应用户质疑"强教师哪里不值得蒸馏")

实验A — 几何-错误对齐校准:
  对每个教师, 用 5 个 GT-无关几何特征训练 logistic 回归预测"教师该样本对错"(5折CV),
  得每样本"几何可信度分"。按分数分档(低/中/高), 看每档的教师真实错误率。
  验证 H4: 几何低可信档的教师错误率 >> 高可信档 (几何能定位"教师在猜"的样本)。

实验B — 领域可蒸馏性地图:
  用 CMExam 的 Medical Discipline (7学科) 分组, 算每学科内教师的真实准确率 + 平均几何可信度,
  找出强教师"整体强但某学科弱"的子领域 (= 不值得蒸馏的地方)。

纯 CPU 分析, 不需训练学生。
用法: python research/distillability/scripts/exp_AB_strong_teacher_map.py
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

OPTION_LETTERS = ["A","B","C","D","E"]
GEOM_FEATS = ["logdet_g","boundary","entropy","margin","peak"]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(REPO)
DIST = os.path.join(REPO,"research","distillability")
LABELS = os.path.join(DIST,"teacher_labels")
OUT = os.path.join(DIST,"outputs","exp_AB_strong_teacher.json")

TEACHERS = {
  "Qwen32B":"qwen32b_train_logprobs.jsonl",   # 强
  "Qwen14B":"qwen14b_train_logprobs.jsonl",   # 强
  "GLM32B":"glm32b_train_logprobs.jsonl",     # 强
  "Yi34B":"yi34b_train_logprobs.jsonl",       # 中
  "Gemma27B":"gemma27b_train_logprobs.jsonl", # 弱
  "Phi4":"phi4_train_logprobs.jsonl",         # 弱
}


def geom_features(p):
    """从 5 维概率分布算 GT-无关几何特征 (与 sample_geometry 一致的近似)。"""
    p = np.clip(np.array(p,dtype=np.float64),1e-12,None); p=p/p.sum()
    srt = np.sort(p)[::-1]
    entropy = float(-np.sum(p*np.log(p)))
    margin = float(srt[0]-srt[1])
    peak = float(srt[0])
    boundary = float(srt[-1])                       # 离单纯形边界(最小分量)
    # logdet_g: Fisher-Rao 度量体积元的对数, 用 sum(log p) 近似(对角 Fisher)
    logdet_g = float(-np.sum(np.log(p)))
    return [logdet_g, boundary, entropy, margin, peak]


def load(path):
    feats,correct,disc = [],[],[]
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
        feats.append(geom_features(raw))
        correct.append(1 if ta==gt else 0)
        disc.append(str(r.get("Medical Discipline","")).strip() or "未知")
    return np.array(feats), np.array(correct), np.array(disc)


def credibility_scores(X, y):
    """5折CV: 用几何特征预测对错, 返回每样本 out-of-fold 预测的'可信度'(P(correct))。"""
    scores=np.zeros(len(y))
    if len(set(y))<2:   # 全对或全错, 无法分类
        return None
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
    Xs=(X-X.mean(0))/(X.std(0)+1e-9)
    for tr,te in skf.split(Xs,y):
        clf=LogisticRegression(max_iter=1000).fit(Xs[tr],y[tr])
        scores[te]=clf.predict_proba(Xs[te])[:,1]
    return scores


def main():
    results={}
    print("="*78); print("实验A: 几何可信度分档 vs 教师真实错误率 (验证 H4)"); print("="*78)
    for label,fn in TEACHERS.items():
        path=os.path.join(LABELS,fn)
        if not os.path.exists(path): print(f"[skip] {label}"); continue
        X,y,disc=load(path)
        acc=float(y.mean()*100)
        cred=credibility_scores(X,y)
        if cred is None: continue
        # 分档: 按可信度分三档
        q1,q2=np.quantile(cred,[1/3,2/3])
        bands={"低(可疑)":cred<=q1,"中":(cred>q1)&(cred<=q2),"高(可信)":cred>q2}
        print(f"\n--- {label} (整体准确率 {acc:.1f}%) ---")
        print(f"  {'档位':<10}{'样本数':>7}{'教师错误率':>12}")
        band_stat={}
        for bn,mask in bands.items():
            if mask.sum()==0: continue
            err=float((1-y[mask]).mean()*100)
            band_stat[bn]={"n":int(mask.sum()),"teacher_error_rate":round(err,2)}
            print(f"  {bn:<10}{int(mask.sum()):>7}{err:>11.1f}%")
        results.setdefault(label,{})["expA_bands"]=band_stat
        results[label]["overall_acc"]=round(acc,2)

    print("\n"+"="*78); print("实验B: 领域可蒸馏性地图 (强教师在哪些学科不可信)"); print("="*78)
    for label,fn in TEACHERS.items():
        path=os.path.join(LABELS,fn)
        if not os.path.exists(path): continue
        X,y,disc=load(path)
        overall=float(y.mean()*100)
        print(f"\n--- {label} (整体 {overall:.1f}%) — 各学科准确率 ---")
        print(f"  {'学科':<14}{'样本数':>7}{'准确率':>9}{'vs整体':>9}")
        dmap={}
        for d in sorted(set(disc)):
            m=disc==d
            if m.sum()<20: continue   # 样本太少不稳定
            da=float(y[m].mean()*100)
            dmap[d]={"n":int(m.sum()),"acc":round(da,2),"delta_vs_overall":round(da-overall,2)}
            flag=" ←弱区" if da<overall-10 else ""
            print(f"  {d:<14}{int(m.sum()):>7}{da:>8.1f}%{da-overall:>+8.1f}{flag}")
        results.setdefault(label,{})["expB_domains"]=dmap

    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    json.dump(results,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[SAVED] {OUT}")
    print("\n判读:")
    print("  A: 若'低可信档'教师错误率 >> '高可信档' → 几何能定位强教师在猜的样本 (H4成立)")
    print("  B: 强教师里 delta<−10 的学科 = 整体强但该领域弱 = '不值得蒸馏的地方'")


if __name__=="__main__":
    main()
