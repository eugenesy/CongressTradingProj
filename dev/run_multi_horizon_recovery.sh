#!/bin/bash

# Resume from 3M
HORIZONS=("3M" "6M" "8M" "12M" "18M" "24M")
ALPHA=0.0
START_YEAR=2019
END_YEAR=2024
EPOCHS=5
SEED=42

echo "Resuming Multi-Horizon Study (Alpha=$ALPHA, Years=$START_YEAR-$END_YEAR)"
echo "Horizons: ${HORIZONS[*]}"

for H in "${HORIZONS[@]}"; do
    echo "=================================================="
    echo "Running Horizon: $H (Parallel)"
    echo "=================================================="
    
    # 1. TGN (Background)
    echo "[$(date)] Launching TGN ($H)..."
    /home/syeugene/miniconda3/envs/chocolate/bin/python dev/run_study.py \
        --horizon $H \
        --alpha $ALPHA \
        --epochs $EPOCHS \
        --start-year $START_YEAR \
        --end-year $END_YEAR \
        --seed $SEED > dev/run_tgn_${H}.log 2>&1 &
    PID_TGN=$!

    # 2. Baselines (Background)
    echo "[$(date)] Launching Baselines ($H)..."
    /home/syeugene/miniconda3/envs/chocolate/bin/python dev/baselines_fair.py \
        --horizon $H \
        --alpha $ALPHA \
        --start-year $START_YEAR \
        --end-year $END_YEAR > dev/run_baseline_${H}.log 2>&1 &
    PID_BASE=$!

    # Wait for both
    wait $PID_TGN
    STATUS_TGN=$?
    
    wait $PID_BASE
    STATUS_BASE=$?
    
    if [ $STATUS_TGN -eq 0 ]; then
        echo "[$(date)] TGN ($H) Completed."
    else
        echo "[$(date)] TGN ($H) FAILED. Check dev/run_tgn_${H}.log"
    fi
    
    if [ $STATUS_BASE -eq 0 ]; then
        echo "[$(date)] Baselines ($H) Completed."
    else
        echo "[$(date)] Baselines ($H) FAILED. Check dev/run_baseline_${H}.log"
    fi
    
    echo "--------------------------------------------------"
done

echo "All Horizons (Recovery) Completed."
