#!/usr/bin/env bash
# W10 retraining ablations (3 items, seed 0).  Run on the GPU server from 2/P2.
# Each run matches the S5 main-table recipe: 100 epochs (30/30/40), batch 16, 4 workers.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"

echo "=== W10-1: no phasor (7 -> 3 input channels: I, roi, M) ==="
"$PYTHON" code/w5_train.py --seed 0 --out runs/w10_nophasor_seed0 --no-phasor \
  --epochs1 30 --epochs2 30 --epochs3 40 --workers 4

echo "=== W10-2: no L_fc (fringe-consistency weight = 0) ==="
"$PYTHON" code/w5_train.py --seed 0 --out runs/w10_nofc_seed0 --no-fc \
  --epochs1 30 --epochs2 30 --epochs3 40 --workers 4

echo "=== W10-3: hard-split 1:1 oversampling vs the mixed S5 recipe ==="
"$PYTHON" code/w5_train.py --seed 0 --out runs/w10_hardsplit_seed0 --hard-split --hard-ratio 0.5 \
  --epochs1 30 --epochs2 30 --epochs3 40 --workers 4

echo "done.  checkpoints: runs/w10_{nophasor,nofc,hardsplit}_seed0/best_stage3.pth"
