#!/bin/bash
source /home/syeugene/miniconda3/bin/activate chocolate

# Ensure the results directory is clean
rm -rf /data1/user_syeugene/fintech/congress_reapplying_nodes/results/experiments/*

echo "Starting 2019-2023 Multi-GPU Processing..."

# GPU 0: 3M, 6M, 12M
(
  export CUDA_VISIBLE_DEVICES=0
  for hz in 3M 6M 12M; do
    echo "[GPU 0] Starting horizon: $hz at $(date)"
    chocolate-train --horizon $hz --epochs 5 --start-year 2019 --end-year 2023 --seed 42 > training_${hz}.log 2>&1
    echo "[GPU 0] Done with $hz at $(date)"
  done
) &

# GPU 1: 18M, 24M (These typically evaluate faster/have less labels)
(
  export CUDA_VISIBLE_DEVICES=1
  for hz in 18M 24M; do
    echo "[GPU 1] Starting horizon: $hz at $(date)"
    chocolate-train --horizon $hz --epochs 5 --start-year 2019 --end-year 2023 --seed 42 > training_${hz}.log 2>&1
    echo "[GPU 1] Done with $hz at $(date)"
  done
) &

echo "All jobs dispatched to GPU 0 and GPU 1. Waiting for completion..."
wait
echo "All horizons finished across both GPUs!"
