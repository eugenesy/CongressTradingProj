#!/bin/bash
# Run TGN ablation for all horizons sequentially
# Usage: bash scripts/run_all_horizons.sh

set -e  # Exit on error

echo "=========================================="
echo "TGN Ablation: All Horizons Sequential Run"
echo "=========================================="
echo ""
echo "Running full model (2019-2024) for all horizons with alpha=0.0"
echo ""

# All available horizons
HORIZONS=("1M" "2M" "3M" "6M" "8M" "12M" "18M" "24M")
ALPHA=0.0

for horizon in "${HORIZONS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting Horizon: $horizon"
    echo "=========================================="
    echo ""
    
    python scripts/run_ablation.py --full-run --full-only --horizon "$horizon" --alpha "$ALPHA"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Completed: $horizon"
        echo "   Results: results/experiments/H_${horizon}_A_${ALPHA}/"
    else
        echo ""
        echo "❌ FAILED: $horizon"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "✅ ALL HORIZONS COMPLETED!"
echo "=========================================="
echo ""
echo "Results saved in:"
for horizon in "${HORIZONS[@]}"; do
    echo "  - results/experiments/H_${horizon}_A_${ALPHA}/"
done
echo ""
