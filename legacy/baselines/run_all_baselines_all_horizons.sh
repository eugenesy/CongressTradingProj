#!/bin/bash
# Run baseline models for all horizons sequentially
# Usage: bash baselines/run_all_baselines_all_horizons.sh

set -e  # Exit on error

echo "=========================================="
echo "Baseline Models: All Horizons Sequential Run"
echo "=========================================="
echo ""
echo "Running all 6 models for all 8 horizons with alpha=0.0"
echo "Models: XGBoost, LightGBM, Random Forest, MLP, Logistic Regression, KNN"
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
    
    # Run all models for this horizon
    python baselines/run_baselines.py --horizon "$horizon" --alpha "$ALPHA" --all
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Completed: $horizon (all models)"
        echo "   Results: results/baselines/H_${horizon}_A_${ALPHA}/"
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
    echo "  - results/baselines/H_${horizon}_A_${ALPHA}/"
done
echo ""
echo "Summary files:"
for horizon in "${HORIZONS[@]}"; do
    echo "  - results/baselines/H_${horizon}_A_${ALPHA}/summary_all_models.csv"
done
echo ""
