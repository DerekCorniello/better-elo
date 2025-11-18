from src.evaluation import run_evaluation, statistical_analysis, train_test_split, analyze_rating_cavities
from src.data_generator import RealDataGenerator
from src.momentum_rating import MomentumRating
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'src')))


def main():
    """Test the momentum-enhanced rating system"""
    # Get username from command line or default to MagnusCarlsen
    username = sys.argv[1] if len(sys.argv) > 1 else "AnnaCramling"
    velocity_window = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Generate real dataset with velocity window
    generator = RealDataGenerator(username=username)
    dataset = generator.generate_dataset(velocity_window=velocity_window)
    print(f"Loaded {len(dataset)} games for {username} with velocity window {velocity_window}")

    # Skip evaluation if dataset is empty
    if not dataset:
        print("No games available for evaluation after filtering.")
        return

    # Add opponent Elo and actual result to dataset (simplified for demo)
    for i, game in enumerate(dataset):
        # Simplified opponent Elo (average of nearby games)
        if i > 0 and i < len(dataset) - 1:
            game.opponent_elo = (dataset[i-1].post_game_elo + dataset[i+1].pre_game_elo) / 2
        else:
            game.opponent_elo = game.pre_game_elo  # Default to same rating
        
        # Simplified actual result based on Elo change
        if game.post_game_elo > game.pre_game_elo:
            game.actual_result = 1.0  # Win
        elif game.post_game_elo < game.pre_game_elo:
            game.actual_result = 0.0  # Loss
        else:
            game.actual_result = 0.5  # Draw

    # Run temporal evaluation (train on past, predict future)
    print("\n=== TEMPORAL EVALUATION: Future Prediction ===")
    best, mses, baseline_mses, r2s, additional_metrics = run_evaluation(
        dataset, num_runs=3, pop_size=50, ngen=25, temporal=True)
    
    print(f"Evolved MSEs: {mses}")
    print(f"Baseline MSEs: {baseline_mses}")

    # Statistical analysis
    results = statistical_analysis(mses, baseline_mses, r2s, additional_metrics)
    print("\n=== STATISTICAL RESULTS ===")
    print(f"Evolved Model MSE: {results['evolved_mean_mse']:.3f} ± {results['evolved_std_mse']:.3f}")
    print(f"Baseline MSE (traditional Elo): {results['baseline_mean_mse']:.3f} ± {results['baseline_std_mse']:.3f}")
    print(f"Evolved R²: {results['evolved_mean_r2']:.3f} ± {results['evolved_std_r2']:.3f}")
    print(f"Improvement (lower MSE): {results['improvement']:.3f}")
    print(f"t-statistic: {results['t_stat']:.3f}, p-value: {results['p_value']:.3f}")
    
    # Future prediction accuracy (temporal evaluation)
    if 'future_accuracy_mean' in results:
        print(f"Future Prediction Accuracy: {results['future_accuracy_mean']:.3f} ± {results['future_accuracy_std']:.3f}")
    
    if results['p_value'] < 0.05:
        print("✓ Statistically significant improvement!")
    else:
        print("✗ No significant improvement.")
    
    # Rating cavity analysis
    cavity_metrics = analyze_rating_cavities(dataset, best)
    print(f"\n=== RATING CAVITY ANALYSIS ===")
    print(f"Total cavities detected: {cavity_metrics['total_cavities']}")
    print(f"Average cavity duration: {cavity_metrics['avg_cavity_duration']:.1f} games")
    print(f"Cavity frequency: {cavity_metrics['cavity_frequency']:.3f}")

    # Print best weights for interpretation
    print(f"\n=== OPTIMAL MOMENTUM WEIGHTS ===")
    features = ['Win Streak', 'Recent Win Rate', 'Avg Accuracy',
                'Rating Trend', 'Games Last 30d', 'Velocity']
    for i, w in enumerate(best):
        print(f"  {features[i]}: {w:.3f}")
    
    print("\n=== MOMENTUM SYSTEM INSIGHTS ===")
    print("Higher positive weights indicate stronger momentum impact.")
    print("Rating Trend and Velocity typically dominate for preventing cavities.")
    
    # Demonstrate momentum-enhanced rating calculation
    print(f"\n=== MOMENTUM-ENHANCED RATING EXAMPLE ===")
    if len(dataset) > 50:
        sample_game = dataset[50]
        momentum_adj = sum(w * f for w, f in zip(best, sample_game.to_feature_vector()))
        momentum_elo = sample_game.pre_game_elo + momentum_adj
        
        # Calculate adaptive K-factor
        momentum_rating = MomentumRating(sample_game.pre_game_elo, momentum_adj)
        adaptive_K = momentum_rating.calculate_adaptive_K(sample_game.to_feature_vector(), best)
        
        print(f"Sample game pre-Elo: {sample_game.pre_game_elo:.1f}")
        print(f"Momentum adjustment: {momentum_adj:.1f}")
        print(f"Momentum-enhanced Elo: {momentum_elo:.1f}")
        print(f"Adaptive K-factor: {adaptive_K:.1f} (vs standard 32.0)")
        print(f"Traditional post-Elo: {sample_game.post_game_elo:.1f}")
        print(f"Momentum system would adjust {'faster' if adaptive_K > 32 else 'slower'} due to current form")
    
    print(f"\n=== KEY BENEFITS OF MOMENTUM SYSTEM ===")
    print("1. Faster rating adjustments for improving/declining players")
    print("2. Reduced time spent in rating cavities")
    print("3. More responsive to streaks and form changes")
    print("4. Better future game outcome predictions")
    print("5. Adaptive K-factors based on momentum indicators")


if __name__ == "__main__":
    main()