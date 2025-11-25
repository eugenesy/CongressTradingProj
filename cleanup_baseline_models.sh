#!/bin/bash

# Script to safely remove the baseline_models directory after successful migration
# Usage: bash cleanup_baseline_models.sh

echo "=================================================="
echo "🧹 Cleanup: Remove baseline_models Directory"
echo "=================================================="
echo ""

# Check if baseline_models exists
if [ ! -d "baseline_models" ]; then
    echo "❌ baseline_models directory not found!"
    echo "   Nothing to clean up."
    exit 1
fi

echo "📊 Current directory size:"
du -sh baseline_models/
echo ""

# Ask for confirmation
echo "⚠️  This will PERMANENTLY DELETE the baseline_models directory."
echo ""
read -p "Are you sure you want to proceed? (yes/no): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo ""
    echo "❌ Cleanup cancelled."
    exit 0
fi

# Optional: Create backup
read -p "Do you want to create a backup first? (yes/no): " backup_choice

if [ "$backup_choice" = "yes" ]; then
    echo ""
    echo "📦 Creating backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_file="baseline_models_backup_${timestamp}.tar.gz"
    tar -czf "$backup_file" baseline_models/
    
    if [ $? -eq 0 ]; then
        echo "✅ Backup created: $backup_file"
        du -sh "$backup_file"
    else
        echo "❌ Backup failed! Aborting cleanup."
        exit 1
    fi
    echo ""
fi

# Remove the directory
echo "🗑️  Removing baseline_models directory..."
rm -rf baseline_models/

if [ $? -eq 0 ]; then
    echo "✅ baseline_models directory successfully removed!"
    echo ""
    echo "📝 Cleanup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Verify the project still works: python bin/run_experiments.py"
    echo "  2. Commit the changes: git add -A && git commit -m 'Remove baseline_models after successful migration'"
    echo "  3. (Optional) Remove this script: rm cleanup_baseline_models.sh"
else
    echo "❌ Failed to remove baseline_models directory!"
    exit 1
fi
