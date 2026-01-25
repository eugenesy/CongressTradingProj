#!/bin/bash
# FULL end-to-end pipeline test INCLUDING dataset building
# Tests: Dataset building → Temporal graph → Ablation study (full-only, 1M, α=0.0)
# Usage: bash scripts/test_full_pipeline.sh
# WARNING: This will regenerate all data files (takes several hours)

set -e  # Exit on error

echo "=========================================="
echo "Project Chocolate: FULL Pipeline Test"
echo "Including Dataset Generation"
echo "=========================================="
echo ""
echo "⚠️  WARNING: This will regenerate all data files!"
echo "   This process may take several hours."
echo ""

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

# Step 1: Build dataset from scratch
echo "[1/4] Building dataset from raw source..."
echo "   This may take several hours..."
python scripts/build_dataset.py

if [ $? -eq 0 ]; then
    echo "✅ Dataset built successfully"
else
    echo "❌ ERROR: Dataset building failed"
    exit 1
fi
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
    echo "✅ FULL PIPELINE TEST COMPLETED!"
    echo "=========================================="
    echo ""
    echo "Results saved to: results/experiments/H_1M_A_0.0/"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ PIPELINE TEST FAILED"
    echo "=========================================="
    exit 1
fi
