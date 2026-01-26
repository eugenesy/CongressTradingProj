#!/bin/bash

# Full list of Horizons
HORIZONS=("3M" "6M" "8M" "12M" "18M" "24M")
ALPHA=0.0
START_YEAR=2019
END_YEAR=2024
EPOCHS=5
SEED=42

# Optional: Allow passing an output directory as first argument
OUT_ROOT=${1:-"results"}
EXP_DIR="$OUT_ROOT/experiments"
BASE_DIR="$OUT_ROOT/baselines"

# Ensure directories exist before redirection
mkdir -p "$EXP_DIR" "$BASE_DIR"

echo "Starting Production Grid Search (Alpha=$ALPHA, Years=$START_YEAR-$END_YEAR)"
echo "Output Directory: $OUT_ROOT"
echo "Horizons: ${HORIZONS[*]}"

for H in "${HORIZONS[@]}"; do
    echo "=================================================="
    echo "Running Horizon: $H (Parallel)"
    echo "=================================================="
    
    # 1. TGN
    echo "[$(date)] Launching GAP-TGN ($H)..."
    python scripts/train_gap_tgn.py \
        --horizon $H \
        --alpha $ALPHA \
        --epochs $EPOCHS \
        --start-year $START_YEAR \
        --end-year $END_YEAR \
        --seed $SEED \
        --out-dir $EXP_DIR > "$EXP_DIR/log_tgn_${H}.txt" 2>&1 &
    PID_TGN=$!

    # 2. Baselines
    echo "[$(date)] Launching Baselines ($H)..."
    python scripts/train_baselines.py \
        --horizon $H \
        --alpha $ALPHA \
        --start-year $START_YEAR \
        --end-year $END_YEAR \
        --out-dir $BASE_DIR > "$BASE_DIR/log_baseline_${H}.txt" 2>&1 &
    PID_BASE=$!

    # Wait for both
    wait $PID_TGN
    STATUS_TGN=$?
    
    wait $PID_BASE
    STATUS_BASE=$?
    
    if [ $STATUS_TGN -eq 0 ]; then
        echo "✅ GAP-TGN ($H) Completed."
    else
        echo "❌ GAP-TGN ($H) FAILED. Check $EXP_DIR/log_tgn_${H}.txt"
    fi
    
    if [ $STATUS_BASE -eq 0 ]; then
        echo "✅ Baselines ($H) Completed."
    else
        echo "❌ Baselines ($H) FAILED. Check $BASE_DIR/log_baseline_${H}.txt"
    fi
    
    echo "--------------------------------------------------"
done

echo "Grid Search Completed."
