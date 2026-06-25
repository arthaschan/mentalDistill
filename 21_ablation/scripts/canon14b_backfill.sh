#!/usr/bin/env bash
set -uo pipefail
# ============================================================
# 组合① (DeepSeek→14B) canonical eval 补全 — Module 21
#
# 背景: ① 当初只对最优 α=0 做了 canonical eval (full=89.14%),
#       其余 6 个 α 点 (0.15/0.25/0.35/0.50/0.65/1.0) 只有内置 eval。
#       为让头条三曲线图三组同口径 (②③均为全7点 canonical),
#       此脚本对 ① 其余 6 α × 3 seed 补跑 canonical (full991 + dental125)。
#
# 复用已验证的 evaluate_model.py 直评 (与 ②③ 完全同一评估器/同一 prompt)。
# adapter 已存在 (21点全在), 不重训, 仅评估。
# 幂等: 已有 marker 的跳过。
# ============================================================
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"
ABL14=15_fulldata_resplit/runs/alpha_ablation_14b   # ① adapter 在 module15
OUT=21_ablation/runs/alpha_ablation_14b_canon       # 补全结果落 21 统一管理
LOG=21_ablation/logs/canon14b_backfill.log
mkdir -p "$OUT" 21_ablation/logs
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if grep -q "CANON14B_BACKFILL_ALL_DONE" "$LOG" 2>/dev/null; then log "已全部完成,跳过"; exit 0; fi

BASE="${BASE_MODEL_14B:-$PWD/models/Qwen2.5-14B-Instruct}"
EVAL="shared/evaluate_model.py"
FULL=21_ablation/data/test.jsonl
DENTAL=21_ablation/data/test_dental.jsonl

# 仅补其余 6 个 α (α=0 已有 canonical, 不重跑)
ALPHAS=(0p15 0p25 0p35 0p50 0p65 1p0)
SEEDS=(11 42 8)

log "=== ① 14B canonical 补全 (6α × 3seed × full+dental) ==="
for atag in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    adir="$ABL14/outputs/a${atag}_s${seed}/stage1_head/best"
    [ -f "$adir/adapter_config.json" ] || { log "WARN 缺 $adir,跳过"; continue; }
    for tname in full dental; do
      tfile=$([ "$tname" = full ] && echo "$FULL" || echo "$DENTAL")
      marker="CANON14B_a${atag}_s${seed}_${tname}_DONE"
      grep -q "$marker" "$LOG" 2>/dev/null && { log "$marker 已存在,跳过"; continue; }
      log "评估 ①14B α=$atag seed=$seed [$tname]"
      "$PY" "$EVAL" --base_model "$BASE" --adapter_dir "$adir" --test_data "$tfile" \
        --wrong_log "$OUT/canon_wrong_a${atag}_s${seed}_${tname}.jsonl" \
        >> "21_ablation/logs/canon14b_a${atag}_s${seed}_${tname}.log" 2>&1 \
        && log "$marker" || log "WARN ①14B α=$atag s$seed $tname 失败"
    done
  done
done

# 汇总 (含已有的 α=0: 直接引用 results.md 的 89.14% / dental 82.13%)
log "=== ① 14B canonical 补全完成, 汇总 ==="
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import re,os
base="21_ablation/logs"
ALPHAS=["0p15","0p25","0p35","0p50","0p65","1p0"]
ALABEL={"0p15":"0.15","0p25":"0.25","0p35":"0.35","0p50":"0.50","0p65":"0.65","1p0":"1.0"}
SEEDS=[11,42,8]
def acc(f):
    if not os.path.exists(f): return None
    cands=[]
    for line in open(f,encoding="utf-8",errors="ignore"):
        if re.search(r'accuracy|正确率|准确率|correct',line,re.I):
            m=re.search(r'(\d+\.?\d*)\s*%',line) or re.search(r'(\d+)\s*/\s*(\d+)',line) or re.search(r'[:：]\s*(\d+\.\d+)',line)
            if m:
                if m.lastindex==2: cands.append(100*int(m.group(1))/int(m.group(2)))
                else: cands.append(float(m.group(1)))
    return cands[-1] if cands else None
print("\n=== ① DeepSeek→14B canonical 完整曲线 (Module 21 补全) ===")
print("(α=0 来自既有 results.md: full=89.14% / dental=82.13%)")
for tname in ["full","dental"]:
    print(f"\n[{tname}]")
    print(f"  α=0.0   (既有) full=89.14% / dental=82.13%" if tname=="full" else "  α=0.0   (既有) dental=82.13%")
    for atag in ALPHAS:
        vals=[]
        for s in SEEDS:
            a=acc(f"{base}/canon14b_a{atag}_s{s}_{tname}.log")
            if a is not None: vals.append(a)
        if vals:
            mean=sum(vals)/len(vals)
            print(f"  α={ALABEL[atag]:<5} mean={mean:.2f}%  n={len(vals)}  seeds={['%.2f'%v for v in vals]}")
        else:
            print(f"  α={ALABEL[atag]:<5} (无结果)")
PYEOF
log "CANON14B_BACKFILL_ALL_DONE"; echo "CANON14B_BACKFILL_ALL_DONE" >> "$LOG"
