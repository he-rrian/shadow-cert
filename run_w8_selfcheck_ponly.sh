#!/bin/bash
set -e
cd /root/autodl-fs/P2
mkdir -p probes/results/w8_selfcheck_ponly
for s in 0 1 2 3 4; do
  /root/miniconda3/bin/python -u code/w8_selfcheck.py \
    --ckpt runs/w5_p0_seed${s}/best_stage3.pth \
    --split test --p-only --reps 20 \
    --alphas 0.05,0.10,0.20 --betas 0.05 \
    --out probes/results/w8_selfcheck_ponly/seed${s}.json
done
touch /root/W8_SELFCHECK_P_ONLY_DONE