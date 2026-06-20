#!/usr/bin/env python3
"""从已有教师logprobs文件提取干净题目, 重建 P3 补教师生成所需的输入数据集。
(原 data_ext/*/test.jsonl 已不在; 但题目完整保存在已生成的 logprobs 里。)"""
import json, os

REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO=os.path.dirname(REPO)
DIST=os.path.join(REPO,"research","distillability")
EXT=os.path.join(DIST,"teacher_labels_ext")
OUTDIR=os.path.join(DIST,"data_ext_rebuilt")

# 每个数据集用一个已存在的教师文件作题目来源
SRC={
  "medqa":"medqa_Qwen32B_logprobs.jsonl",
  "mmlu_med":"mmlu_med_Qwen32B_logprobs.jsonl",
  "mmlu_full":"mmlu_full_Qwen32B_logprobs.jsonl",
}
KEEP=["Question","Options","Answer","Subject","_num_options","OriginalAnswer"]

for ds,fn in SRC.items():
    src=os.path.join(EXT,fn)
    if not os.path.exists(src):
        print(f"[skip] {ds}: 源文件缺 {fn}"); continue
    outd=os.path.join(OUTDIR,ds); os.makedirs(outd,exist_ok=True)
    out=os.path.join(outd,"test.jsonl")
    n=0
    with open(out,"w",encoding="utf-8") as w:
        for line in open(src,encoding="utf-8"):
            line=line.strip()
            if not line: continue
            try: r=json.loads(line)
            except: continue
            clean={k:r[k] for k in KEEP if k in r}
            if "Question" not in clean: continue
            w.write(json.dumps(clean,ensure_ascii=False)+"\n"); n+=1
    print(f"[ok] {ds}: 重建 {n} 题 -> {out}")
print("DONE")
