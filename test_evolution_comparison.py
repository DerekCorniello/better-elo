#!/usr/bin/env python3
"""
Test script to compare baseline vs improved evolutionary algorithm.
Tests blend crossover + constraint enforcement vs baseline two-point crossover.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from better_elo.data_generator import RealDataGenerator
from better_elo.evaluation import run_evaluation
from better_elo.ea import run_evolution
import json

def test_baseline_vs_improved():
    """Compare baseline algorithm with improved version."""
    
    print("=== Evolutionary Algorithm Comparison Test ===\n")
    
    # Load Magnus data for consistency with baseline
    username = "MagnusCarlsen"
    print(f"Loading data for {username}...")
    
    try:
        generator = RealDataGenerator(username)
        dataset = generator.generate_dataset(velocity_window=50)
        print(f"Loaded {len(dataset)} games\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Test configurations
    experiments = [
        {
            'name': 'Baseline (Two-Point Crossover)',
            'crossover': 'cxTwoPoint',
            'generations': 50,
            'constraints': False,
            'description': 'Original algorithm with two-point crossover, no constraints'
        },
        {
            'name': 'Improved (Blend Crossover + Constraints)',
            'crossover': 'cxBlend', 
            'generations': 50,
            'constraints': True,
            'description': 'New algorithm with blend crossover (α=0.3) and constraint enforcement'
        }
    ]
    
    results = {}
    
    for experiment in experiments:
        print(f"Running: {experiment['name']}")
        print(f"Description: {experiment['description']}")
        print(f"Generations: {experiment['generations']}")
        print("-" * 60)
        
        # Run evaluation with 3 seeds for statistical significance
        try:
            best, mses, baseline_mses, r2s = run_evaluation(
                dataset, 
                num_runs=3, 
                pop_size=100, 
                ngen=experiment['generations']
            )
            
            results[experiment['name']] = {
                'mse_mean': sum(mses) / len(mses),
                'mse_std': (sum((x - sum(mses)/len(mses))**2 for x in mses) / len(mses))**0.5,
                'r2_mean': sum(r2s) / len(r2s),
                'r2_std': (sum((x - sum(r2s)/len(r2s))**2 for x in r2s) / len(r2s))**0.5,
                'baseline_mse_mean': sum(baseline_mses) / len(baseline_mses),
                'improvement': sum(baseline_mses) / len(baseline_mses) - (sum(mses) / len(mses)),
                'best_weights': [float(w) for w in best],
                'config': experiment
            }
            
            print(f"Results:")
            print(f"  Evolved MSE: {results[experiment['name']]['mse_mean']:.1f} ± {results[experiment['name']]['mse_std']:.1f}")
            print(f"  Baseline MSE: {results[experiment['name']]['baseline_mse_mean']:.1f}")
            print(f"  Improvement: {results[experiment['name']]['improvement']:.1f}")
            print(f"  R²: {results[experiment['name']]['r2_mean']:.3f} ± {results[experiment['name']]['r2_std']:.3f}")
            print(f"  Best weights: {[f'{w:.3f}' for w in best]}")
            
        except Exception as e:
            print(f"Error running experiment: {e}")
            results[experiment['name']] = {'error': str(e)}
        
        print("\n" + "="*60 + "\n")
    
    # Save results
    output_file = 'evolution_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Summary comparison
    if len(results) >= 2 and 'error' not in list(results.values())[0] and 'error' not in list(results.values())[1]:
        baseline = list(results.values())[0]
        improved = list(results.values())[1]
        
        print("=== COMPARISON SUMMARY ===")
        print(f"Baseline MSE: {baseline['mse_mean']:.1f}")
        print(f"Improved MSE: {improved['mse_mean']:.1f}")
        improvement_pct = ((baseline['mse_mean'] - improved['mse_mean']) / baseline['mse_mean']) * 100
        print(f"Improvement: {improvement_pct:.1f}%")
        print(f"Baseline R²: {baseline['r2_mean']:.3f}")
        print(f"Improved R²: {improved['r2_mean']:.3f}")
        
        if improvement_pct > 0:
            print("✓ IMPROVED VERSION PERFORMS BETTER")
        else:
            print("✗ Baseline version performs better")

if __name__ == "__main__":
    test_baseline_vs_improved()