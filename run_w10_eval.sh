#!/usr/bin/env bash
# W10 G8 evaluation: point/fair metrics + C3 + C4 diagnostics for the 3 ablations.
# Runs after run_w10_ablations.sh finishes.  CPU/GPU mixed; no retraining.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p probes/results
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"

for v in nophasor nofc hardsplit; do
  ck="runs/w10_${v}_seed0/best_stage3.pth"
  if [ ! -f "$ck" ]; then
    echo "MISSING checkpoint: $ck"
    exit 2
  fi
  extra=""
  if [ "$v" = "nophasor" ]; then extra="--no-phasor"; fi
  outdir="probes/results/w10_${v}_g5"
  mkdir -p "$outdir"
  # reuse the deterministic S5 baseline caches (same val/test/sf data)
  for split in val test sf_test; do
    src="probes/results/w5_5seeds/seed0/baseline_${split}.json"
    [ -f "$src" ] || src="probes/results/w5_p0_final/baseline_${split}.json"
    if [ -f "$src" ]; then cp "$src" "$outdir/"; fi
  done
  echo "=== G8 point/fair: $v ==="
  "$PYTHON" code/w5_eval.py --ckpt "$ck" --splits val,test,sf_test \
    --out "$outdir" $extra
  echo "=== G8 C3: $v ==="
  "$PYTHON" code/w8_conformal.py --ckpt "$ck" --cal val --test test \
    --alphas 0.05,0.10,0.20 --betas 0.05 \
    --out "probes/results/w10_${v}_c3.json" $extra
  echo "=== G8 C4 diagnostic: $v ==="
  "$PYTHON" code/w8_rcps.py --ckpt "$ck" --cal val --test test \
    --alphas 0.05,0.10,0.20 --out "probes/results/w10_${v}_rcps.json" $extra
done
echo "W10_EVAL_DONE"
