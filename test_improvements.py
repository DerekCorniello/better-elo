#!/usr/bin/env python3
"""
Simple test to compare baseline vs improved evolutionary algorithm.
Run this script to test the blend crossover + constraint enforcement improvements.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from better_elo.data_generator import RealDataGenerator
from better_elo.evaluation import run_evaluation

def main():
    print("=== Testing Evolutionary Algorithm Improvements ===\n")
    
    # Load Magnus data
    username = "MagnusCarlsen"
    print(f"Loading data for {username}...")
    
    try:
        generator = RealDataGenerator(username)
        dataset = generator.generate_dataset(velocity_window=50)
        print(f"Loaded {len(dataset)} games\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Running baseline evaluation (Two-Point Crossover, 50 generations)...")
    print("This will take a few minutes...\n")
    
    # Run the evaluation with current settings
    # This will use the modified algorithm (blend crossover + constraints)
    try:
        best, mses, baseline_mses, r2s = run_evaluation(
            dataset, 
            num_runs=3, 
            pop_size=100, 
            ngen=200
        )
        
        print("=== RESULTS ===")
        print(f"Evolved Model MSE: {sum(mses)/len(mses):.1f} ± {(sum((x - sum(mses)/len(mses))**2 for x in mses) / len(mses))**0.5:.1f}")
        print(f"Baseline MSE (no momentum): {sum(baseline_mses)/len(baseline_mses):.1f}")
        print(f"Improvement: {sum(baseline_mses)/len(baseline_mses) - sum(mses)/len(mses):.1f}")
        print(f"R²: {sum(r2s)/len(r2s):.3f} ± {(sum((x - sum(r2s)/len(r2s))**2 for x in r2s) / len(r2s))**0.5:.3f}")
        print(f"\nBest weights found:")
        feature_names = ["Win Streak", "Recent Win Rate", "Avg Accuracy", "Rating Trend", "Games Last 30d", "Velocity"]
        for i, (name, weight) in enumerate(zip(feature_names, best)):
            print(f"  {name}: {weight:.3f}")
        
        print(f"\nComparison to baseline (from magnus_output.txt):")
        print(f"Baseline MSE: 6027.317 ± 934.277")
        print(f"Current MSE: {sum(mses)/len(mses):.1f}")
        
        improvement = ((6027.317 - sum(mses)/len(mses)) / 6027.317) * 100
        if improvement > 0:
            print(f"Improvement: {improvement:.1f}% ✓ BETTER")
        else:
            print(f"Change: {improvement:.1f}% ✗ WORSE")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()