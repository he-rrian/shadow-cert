#!/usr/bin/env bash
# Wait for the W10 three-run script to finish, then run G8 evaluation.
set -euo pipefail
cd "$(dirname "$0")"
for i in $(seq 1 720); do
  if grep -q '^done\.' runs/w10_all.log 2>/dev/null; then
    bash run_w10_eval.sh > runs/w10_eval.log 2>&1
    echo "W10_EVAL_LAUNCHED $(date -Is)" >> runs/w10_waiter.log
    exit 0
  fi
  sleep 60
done
echo "W10_WAIT_TIMEOUT $(date -Is)" >> runs/w10_waiter.log
exit 3
