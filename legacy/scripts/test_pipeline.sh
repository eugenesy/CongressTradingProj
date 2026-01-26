#!/bin/bash
# End-to-end pipeline test for Project Chocolate
# Tests: Dataset building → Temporal graph → Ablation study (full-only, 1M, α=0.0)
# Usage: bash scripts/test_pipeline.sh

set -e  # Exit on error

echo "=========================================="
echo "Project Chocolate: End-to-End Pipeline Test"
echo "=========================================="
echo ""

# Activate conda environment (optional, comment out if already activated)
# conda activate chocolate

# Step 0: Check if package is installed
echo "[0/4] Checking package installation..."
python -c "import src" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Package not installed!"
    echo ""
    echo "Please install the package first:"
    echo "  pip install -e ."
    echo ""
    exit 1
fi
echo "✅ Package installed"
echo ""

# Step 1: Check if data files exist
echo "[1/4] Checking for required data files..."
if [ ! -f "data/processed/ml_dataset_reduced_attributes.csv" ]; then
    echo "❌ ERROR: ml_dataset_reduced_attributes.csv not found!"
    echo "   Run: python scripts/build_dataset.py"
    exit 1
fi

if [ ! -f "data/price_sequences.pt" ]; then
    echo "❌ ERROR: price_sequences.pt not found!"
    echo "   Run: python scripts/build_dataset.py"
    exit 1
fi

echo "✅ Data files found"
echo ""

# Step 2: Build temporal graph
echo "[2/4] Building temporal graph (Filed date)..."
python src/temporal_data.py
if [ $? -eq 0 ]; then
    echo "✅ Temporal graph built successfully"
else
    echo "❌ ERROR: Temporal graph building failed"
    exit 1
fi
echo ""

# Step 3: Verify temporal_data.pt exists
echo "[3/4] Verifying temporal_data.pt..."
if [ ! -f "data/temporal_data.pt" ]; then
    echo "❌ ERROR: temporal_data.pt not generated!"
    exit 1
fi
echo "✅ temporal_data.pt verified"
echo ""

# Step 4: Run ablation study (full-only, 1M horizon, α=0.0)
echo "[4/4] Running ablation study..."
echo "   Mode: Full-only"
echo "   Horizon: 1M"
echo "   Alpha: 0.0"
echo ""

python scripts/run_ablation.py --full-only --horizon 1M --alpha 0.0

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ PIPELINE TEST COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo ""
    echo "Results saved to: results/experiments/H_1M_A_0.0/"
    echo ""
    echo "Check reports:"
    echo "  - results/experiments/H_1M_A_0.0/reports/"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ PIPELINE TEST FAILED"
    echo "=========================================="
    exit 1
fi
