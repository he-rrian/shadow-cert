#!/usr/bin/env bash
# W10 retraining ablations (3 items, seed 0).  Run on the GPU server from 2/P2.
# Each run is the S5 recipe (130 epochs: 35/35/40) with one change.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs

echo "=== W10-1: no phasor (7 -> 3 input channels: I, roi, M) ==="
python code/w5_train.py --seed 0 --out runs/w10_nophasor_seed0 --no-phasor

echo "=== W10-2: no L_fc (fringe-consistency weight = 0) ==="
python code/w5_train.py --seed 0 --out runs/w10_nofc_seed0 --no-fc

echo "=== W10-3: hard-split 1:1 oversampling vs the mixed S5 recipe ==="
python code/w5_train.py --seed 0 --out runs/w10_hardsplit_seed0 --hard-split --hard-ratio 0.5

echo "done.  checkpoints: runs/w10_{nophasor,nofc,hardsplit}_seed0/best_stage3.pth"
