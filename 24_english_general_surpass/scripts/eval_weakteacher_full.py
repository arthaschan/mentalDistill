#!/usr/bin/env python3
"""英文全科"弱教师"组合评估：训练后学生 vs 弱教师 Qwen3-32B(73.84%)。

训练后/零样本数字复用 fullEnglish 主实验已算好的结果 JSON（无需重训/重测）：
- Qwen2.5-32B → fullEnglish/03_main_distill/runs/eval_results.json（结构为 per-test，需加权合并）
- Llama-70B   → fullEnglish/03_main_distill/runs/eval_results_llama70b.json（已有 combined 字段）
- 弱教师零样本 → fullEnglish/03_main_distill/runs/eval_results_qwen3_zeroshot.json
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.path.join(ROOT, "runs")
FE_RUNS = os.path.join(ROOT, "..", "fullEnglish", "03_main_distill", "runs")

WEAK_TEACHER = {"name": "Qwen3-32B(弱教师)", "acc": 73.84}
FLASH = 79.80
MCQ = ["test_medqa", "test_medmcqa", "test_mmlu"]
SET_COUNTS = {"test_medqa": 1273, "test_medmcqa": 4183, "test_mmlu": 2837}


def combined_mcq(per_test):
    """per-test 准确率 dict → 组合 MCQ 加权准确率。"""
    n = sum(SET_COUNTS[s] for s in MCQ)
    return round(sum(per_test[s] * SET_COUNTS[s] for s in MCQ) / n, 2)


def main():
    r32 = json.load(open(os.path.join(FE_RUNS, "eval_results.json")))
    rl = json.load(open(os.path.join(FE_RUNS, "eval_results_llama70b.json")))
    rz = json.load(open(os.path.join(FE_RUNS, "eval_results_qwen3_zeroshot.json")))

    # Qwen2.5-32B：零样本 per-test 加权 + 训练后 3-seed 均值加权
    qwen25_zero = combined_mcq(r32["zeroshot"])
    seeds = ["32B_a00_s11", "32B_a00_s42", "32B_a00_s8"]
    qwen25_trained = round(statistics.mean(
        [combined_mcq(r32["adapters"][s]["acc"]) for s in seeds]), 2)

    # Llama-70B：已有 combined 字段
    llama_zero = rl["combined_zeroshot"]
    llama_trained = rl["combined_student"]

    weak = WEAK_TEACHER["acc"]

    print("=== 英文全科(8293题) 弱教师组合 ===")
    print(f"弱教师 Qwen3-32B 零样本: {weak}%")
    print(f"学生 Qwen2.5-32B: 零样本 {qwen25_zero}% → 训练后 {qwen25_trained}%  "
          f"超弱教师 {qwen25_trained - weak:+.2f}pp")
    print(f"学生 Llama-70B: 零样本 {llama_zero}% → 训练后 {llama_trained}%  "
          f"超弱教师 {llama_trained - weak:+.2f}pp")
    print(f"（强教师 flash {FLASH}%，学生都追不上）")

    json.dump({"weak_teacher": WEAK_TEACHER, "flash": FLASH,
               "qwen25_32b": {"zero": qwen25_zero, "trained": qwen25_trained,
                              "delta": round(qwen25_trained - weak, 2)},
               "llama70b": {"zero": llama_zero, "trained": llama_trained,
                            "delta": round(llama_trained - weak, 2)}},
              open(os.path.join(RUN, "eval_results_en_general_weakteacher.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"-> {RUN}/eval_results_en_general_weakteacher.json")


if __name__ == "__main__":
    main()
