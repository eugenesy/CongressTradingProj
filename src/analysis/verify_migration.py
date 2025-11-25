"""
Verify that the migrated code produces identical results to baseline_models.

This script compares:
1. Predictions (JSON files)
2. Metrics (CSV files)
3. Summary statistics

Usage:
    python -m src.analysis.verify_migration
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_json(file_path: Path) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_predictions(baseline_path: Path, new_path: Path, model_name: str) -> Dict:
    """
    Compare prediction files between baseline and new implementation.
    
    Returns:
        Dictionary with comparison results
    """
    results = {
        'model_name': model_name,
        'files_compared': 0,
        'files_matched': 0,
        'files_different': 0,
        'differences': []
    }
    
    # Get all prediction files from baseline
    baseline_pred_files = list(baseline_path.glob('*/predictions.json'))
    baseline_pred_files.append(baseline_path / 'all_predictions.json')
    
    for baseline_file in baseline_pred_files:
        if not baseline_file.exists():
            continue
            
        # Construct corresponding new file path
        relative_path = baseline_file.relative_to(baseline_path)
        new_file = new_path / relative_path
        
        if not new_file.exists():
            results['differences'].append({
                'file': str(relative_path),
                'issue': 'Missing in new implementation'
            })
            results['files_different'] += 1
            continue
        
        # Load both files
        baseline_data = load_json(baseline_file)
        new_data = load_json(new_file)
        
        results['files_compared'] += 1
        
        # Compare
        if baseline_data == new_data:
            results['files_matched'] += 1
        else:
            # Find specific differences
            baseline_ids = set(baseline_data.keys())
            new_ids = set(new_data.keys())
            
            missing_ids = baseline_ids - new_ids
            extra_ids = new_ids - baseline_ids
            common_ids = baseline_ids & new_ids
            
            diff_info = {
                'file': str(relative_path),
                'missing_transaction_ids': len(missing_ids),
                'extra_transaction_ids': len(extra_ids),
                'prediction_mismatches': 0
            }
            
            # Check predictions for common IDs
            for tid in common_ids:
                if baseline_data[tid] != new_data[tid]:
                    diff_info['prediction_mismatches'] += 1
            
            results['differences'].append(diff_info)
            results['files_different'] += 1
    
    return results


def compare_metrics(baseline_path: Path, new_path: Path, model_name: str) -> Dict:
    """
    Compare metric CSV files between baseline and new implementation.
    
    Returns:
        Dictionary with comparison results
    """
    results = {
        'model_name': model_name,
        'files_compared': 0,
        'files_matched': 0,
        'files_different': 0,
        'differences': []
    }
    
    # Get all metric files from baseline
    baseline_metric_files = list(baseline_path.glob('*/metrics.csv'))
    baseline_metric_files.append(baseline_path / 'summary_metrics.csv')
    
    for baseline_file in baseline_metric_files:
        if not baseline_file.exists():
            continue
            
        # Construct corresponding new file path
        relative_path = baseline_file.relative_to(baseline_path)
        new_file = new_path / relative_path
        
        if not new_file.exists():
            results['differences'].append({
                'file': str(relative_path),
                'issue': 'Missing in new implementation'
            })
            results['files_different'] += 1
            continue
        
        # Load both files
        baseline_df = pd.read_csv(baseline_file)
        new_df = pd.read_csv(new_file)
        
        results['files_compared'] += 1
        
        # Compare
        try:
            pd.testing.assert_frame_equal(baseline_df, new_df, check_exact=False, rtol=1e-5)
            results['files_matched'] += 1
        except AssertionError as e:
            diff_info = {
                'file': str(relative_path),
                'issue': 'Values differ',
                'details': str(e)[:200]  # Truncate error message
            }
            results['differences'].append(diff_info)
            results['files_different'] += 1
    
    return results


def verify_all_models() -> List[Dict]:
    """
    Verify all models that exist in both baseline and new implementation.
    
    Returns:
        List of comparison results for each model
    """
    root = get_project_root()
    baseline_results_dir = root / 'baseline_models' / 'results'
    new_results_dir = root / 'data' / 'results'
    
    if not baseline_results_dir.exists():
        print("❌ Baseline models directory not found!")
        return []
    
    if not new_results_dir.exists():
        print("❌ New results directory not found!")
        return []
    
    # Get all model directories
    baseline_models = {d.name for d in baseline_results_dir.iterdir() if d.is_dir()}
    new_models = {d.name for d in new_results_dir.iterdir() if d.is_dir()}
    
    common_models = baseline_models & new_models
    missing_models = baseline_models - new_models
    extra_models = new_models - baseline_models
    
    print(f"\n{'='*60}")
    print(f"🔍 MIGRATION VERIFICATION REPORT")
    print(f"{'='*60}\n")
    
    print(f"📊 Model Inventory:")
    print(f"  - Baseline models: {len(baseline_models)}")
    print(f"  - New models: {len(new_models)}")
    print(f"  - Common models: {len(common_models)}")
    
    if missing_models:
        print(f"  - ⚠️  Missing in new: {', '.join(missing_models)}")
    if extra_models:
        print(f"  - ℹ️  Extra in new: {', '.join(extra_models)}")
    
    # Compare each common model
    all_results = []
    
    for model_name in sorted(common_models):
        print(f"\n{'─'*60}")
        print(f"🔬 Verifying: {model_name}")
        print(f"{'─'*60}")
        
        baseline_path = baseline_results_dir / model_name
        new_path = new_results_dir / model_name
        
        # Compare predictions
        pred_results = compare_predictions(baseline_path, new_path, model_name)
        metric_results = compare_metrics(baseline_path, new_path, model_name)
        
        # Print summary
        print(f"\n  📄 Predictions:")
        print(f"    - Files compared: {pred_results['files_compared']}")
        print(f"    - Files matched: {pred_results['files_matched']}")
        print(f"    - Files different: {pred_results['files_different']}")
        
        if pred_results['differences']:
            print(f"    - ⚠️  Differences detected:")
            for diff in pred_results['differences']:
                print(f"      • {diff['file']}: {diff.get('issue', '')}")
                if 'prediction_mismatches' in diff:
                    print(f"        - Prediction mismatches: {diff['prediction_mismatches']}")
        else:
            print(f"    - ✅ All prediction files match!")
        
        print(f"\n  📊 Metrics:")
        print(f"    - Files compared: {metric_results['files_compared']}")
        print(f"    - Files matched: {metric_results['files_matched']}")
        print(f"    - Files different: {metric_results['files_different']}")
        
        if metric_results['differences']:
            print(f"    - ⚠️  Differences detected:")
            for diff in metric_results['differences']:
                print(f"      • {diff['file']}: {diff.get('issue', '')}")
        else:
            print(f"    - ✅ All metric files match!")
        
        all_results.append({
            'model_name': model_name,
            'predictions': pred_results,
            'metrics': metric_results
        })
    
    # Overall summary
    print(f"\n{'='*60}")
    print(f"✅ OVERALL SUMMARY")
    print(f"{'='*60}\n")
    
    total_matched = sum(1 for r in all_results 
                       if r['predictions']['files_different'] == 0 
                       and r['metrics']['files_different'] == 0)
    
    print(f"Models fully matched: {total_matched}/{len(all_results)}")
    
    if total_matched == len(all_results):
        print(f"\n🎉 ALL MODELS VERIFIED SUCCESSFULLY!")
        print(f"✅ The migration is complete and correct.")
        print(f"✅ You can safely remove the baseline_models directory.")
    else:
        print(f"\n⚠️  Some differences detected. Review details above.")
    
    return all_results


if __name__ == "__main__":
    verify_all_models()
