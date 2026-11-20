#!/bin/bash
set -e
cd /root/autodl-fs/P2
for s in 0 1 2 3 4; do
  /root/miniconda3/bin/python -u code/w8_conformal.py \
    --ckpt runs/w5_p0_seed${s}/best_stage3.pth \
    --cal val --test test --gamma 0.75 \
    --alphas 0.05,0.10,0.20 --betas 0.05 \
    --out probes/results/w8_p_only/seed${s}.json
done
touch /root/W8_P_ONLY_DONE