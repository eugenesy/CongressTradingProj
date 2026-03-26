#!/bin/bash
# run_short_horizons.sh

# Target short-term horizons
HORIZONS=("1W" "2W" "1M" "2M" "3M" "4M")
START_YEAR=2014
END_YEAR=2025
ALPHA=0.0
EPOCHS=5
SEED=42

# Directory for outputs
OUT_DIR="results/experiments_short_horizons"
mkdir -p "$OUT_DIR"

echo "Starting GAP-TGN Training for Short Horizons ($START_YEAR-$END_YEAR)"

for H in "${HORIZONS[@]}"; do
    echo "=================================================="
    echo "Launching GAP-TGN ($H)..."
    
    # Run sequentially to avoid overloading the server's memory
    python scripts/train_gap_tgn.py \
        --horizon $H \
        --alpha $ALPHA \
        --epochs $EPOCHS \
        --start-year $START_YEAR \
        --end-year $END_YEAR \
        --seed $SEED \
        --out-dir $OUT_DIR > "$OUT_DIR/log_tgn_${H}.txt" 2>&1
        
    if [ $? -eq 0 ]; then
        echo "✅ GAP-TGN ($H) Completed."
    else
        echo "❌ GAP-TGN ($H) FAILED. Check $OUT_DIR/log_tgn_${H}.txt"
    fi
done

echo "=================================================="
echo "All short horizons completed."