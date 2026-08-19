#!/usr/bin/env python3
"""汇总"去掉印度 MedMCQA"前后的各模型准确率、增益、headroom、超越情况。

只读已有评测结果 JSON + 教师标签，不重新训练。
口径：全量 MCQ = medqa(1273)+medmcqa(4183)+mmlu(2837)=8293；
      无印度 = medqa+mmlu = 4110（去掉 MedMCQA）。
"""
import json

SRC = {
    "medqa": 1273, "medmcqa": 4183, "mmlu": 2837, "pubmedqa": 1000,
}
MCQ = ["medqa", "medmcqa", "mmlu"]
NO_INDIA = ["medqa", "mmlu"]

RUN = "fullEnglish/03_main_distill/runs"
LAB = "fullEnglish/03_main_distill/labels"


def weighted(per, sources):
    n = sum(SRC[s] for s in sources)
    return round(sum(per[s] * SRC[s] for s in sources) / n, 2)


def teacher_flash():
    """从标签文件算 flash 教师各来源准确率。"""
    per = {}
    for s in ["medqa", "medmcqa", "mmlu"]:
        path = f"{LAB}/teacher_test_{s}.jsonl"
        c = n = 0
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ta = str(r.get("TeacherAnswer") or "").strip().upper()[:1]
            gt = str(r.get("OriginalAnswer") or r.get("Answer") or "").strip().upper()[:1]
            if ta in "ABCDE" and gt in "ABCDE":
                n += 1
                c += (ta == gt)
        per[s] = round(100.0 * c / n, 2)
    return per


def main():
    flash = teacher_flash()
    print(f"flash 教师各来源: {flash}", flush=True)
    print(f"  flash 全量={weighted(flash, MCQ)}%  无印度={weighted(flash, NO_INDIA)}%", flush=True)

    # 学生：zero(per-source) + trained(per-source mean)
    students = {
        "Qwen2.5-32B": {
            "zero": {"medqa": 67.4, "medmcqa": 64.4, "mmlu": 83.47, "pubmedqa": 48.6},
            "train": {"medqa": (72.43 + 73.06 + 74.0) / 3,
                      "medmcqa": (69.09 + 69.07 + 69.21) / 3,
                      "mmlu": (86.01 + 86.43 + 86.68) / 3,
                      "pubmedqa": (59.3 + 60.3 + 59.7) / 3},
        },
        "Qwen2.5-14B": {
            "zero": None,  # 14B 零样本无分来源，只有 ~67.83 估计
            "train": {"medqa": 68.97, "medmcqa": 66.13, "mmlu": 83.22, "pubmedqa": 58.47},
        },
        "Llama-70B": {
            "zero": {"medqa": 71.41, "medmcqa": 65.93, "mmlu": 82.48, "pubmedqa": 59.7},
            "train": {"medqa": 74.47, "medmcqa": 70.88, "mmlu": 85.02, "pubmedqa": 63.37},
        },
        "Qwen3-32B": {
            "zero": {"medqa": 70.07, "medmcqa": 67.58, "mmlu": 84.77, "pubmedqa": 55.2},
            "train": {"medqa": 77.38, "medmcqa": 71.52, "mmlu": 86.65, "pubmedqa": 58.7},
        },
    }

    print("\n模型         口径    零样本   蒸馏后   增益    flash    headroom   超越?")
    for name, d in students.items():
        if d["zero"] is None:
            continue
        for label, srcs in [("全量", MCQ), ("无印度", NO_INDIA)]:
            z = weighted(d["zero"], srcs)
            t = weighted(d["train"], srcs)
            g = round(t - z, 2)
            f = weighted(flash, srcs)
            h = round(f - z, 2)
            sup = "超" if t > f else "不超"
            print(f"{name:12s} {label:5s} {z:6.2f}% {t:6.2f}% +{g:5.2f} "
                  f"{f:6.2f}% {h:+6.2f}  {sup}", flush=True)

    # 增益分来源（Qwen2.5-32B 举例）
    print("\n=== 增益分来源（Qwen2.5-32B） ===", flush=True)
    d = students["Qwen2.5-32B"]
    for s in MCQ:
        g = d["train"][s] - d["zero"][s]
        print(f"  {s:9s}: 零样本 {d['zero'][s]:.2f}% → 蒸馏 {d['train'][s]:.2f}%  增益 +{g:.2f}", flush=True)


if __name__ == "__main__":
    main()
