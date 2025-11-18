from src.evaluation import run_evaluation, statistical_analysis, train_test_split, predict_momentum_adjustment
from src.data_generator import RealDataGenerator
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'src')))


def plot_velocity_scatter(test_dataset, predictions, actuals, username):
    """Plot scatter of predicted vs actual velocities."""
    print(f"Velocity scatter: {len(predictions)} points, actual range: {min(actuals):.1f} to {
          max(actuals):.1f}, predicted range: {min(predictions):.1f} to {max(predictions):.1f}")
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, predictions, alpha=0.6, color='blue')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)],
             color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Velocity')
    plt.ylabel('Predicted Velocity')
    plt.title(f'Predicted vs Actual Velocity for {username}')
    plt.legend()
    plt.grid(True)
    plt.show()  # Display the plot in a window
    plt.savefig(f'velocity_scatter_{username}.png')
    plt.close()


def plot_elo_trajectory(test_dataset, predictions, actuals, username):
    """Plot Elo trajectory over time."""
    print(f"Elo trajectory: {len(actuals)} points")
    game_indices = list(range(len(test_dataset)))
    plt.figure(figsize=(10, 6))
    plt.plot(game_indices, actuals, label='Actual Elo',
             color='blue', linewidth=2)
    plt.plot(game_indices, predictions,
             label='Predicted Adjusted Elo', color='orange', linewidth=2)
    plt.xlabel('Game Index')
    plt.ylabel('Elo Rating')
    plt.title(f'Elo Trajectory for {username}')
    plt.legend()
    plt.grid(True)
    plt.show()  # Display the plot in a window
    plt.savefig(f'elo_trajectory_{username}.png')
    plt.close()


def plot_feature_weights(best_weights, username):
    """Plot bar chart of feature weights."""
    features = ['Win Streak', 'Recent Win Rate',
                'Avg Accuracy', 'Rating Trend', 'Games Last 30d', 'Velocity']
    plt.figure(figsize=(8, 6))
    plt.bar(features, best_weights, color='green')
    plt.xlabel('Features')
    plt.ylabel('Weights')
    plt.title(f'Feature Weights for {username}')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.show()  # Display the plot in a window
    plt.savefig(f'feature_weights_{username}.png')
    plt.close()


def plot_full_trajectory(dataset, best_weights, username):
    """Plot full Elo trajectory: actual vs predicted adjusted Elo, with difference highlights."""
    # Sort dataset by end_time for chronological order
    sorted_dataset = sorted(dataset, key=lambda g: g.end_time)

    # Compute predicted adjusted Elo for each game
    predicted_elos = []
    actual_elos = []
    differences = []
    for i, game in enumerate(sorted_dataset):
        adjustment = predict_momentum_adjustment(
            best_weights, game.to_feature_vector())
        predicted_elo = game.pre_game_elo + adjustment
        predicted_elos.append(predicted_elo)
        actual_elos.append(game.post_game_elo)
        differences.append(predicted_elo - game.post_game_elo)
        # Debug: print first 10
        if i < 10:
            print(f"Game {i}: predicted={predicted_elo:.1f}, actual={
                  game.post_game_elo:.1f}, diff={predicted_elo - game.post_game_elo:.1f}")

    game_indices = list(range(len(sorted_dataset)))

    # Convert to numpy arrays for operations
    actual_elos = np.array(actual_elos)
    predicted_elos = np.array(predicted_elos)
    differences = np.array(differences)

    # Plot 1: Trajectories
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(game_indices, actual_elos,
             label='Chess.com Elo (Actual)', color='blue', linewidth=2)
    plt.plot(game_indices, predicted_elos,
             label='Predicted True Elo (Adjusted)', color='orange', linewidth=2)
    plt.fill_between(game_indices, actual_elos, predicted_elos, where=(
        predicted_elos > actual_elos), color='green', alpha=0.3, label='Underrated (Predicted > Actual)')
    plt.fill_between(game_indices, actual_elos, predicted_elos, where=(
        predicted_elos < actual_elos), color='red', alpha=0.3, label='Overrated (Predicted < Actual)')
    plt.xlabel('Game Index (Chronological)')
    plt.ylabel('Elo Rating')
    plt.title(
        f'Elo Trajectories: Chess.com vs. Predicted True Elo for {username}')
    plt.legend()
    plt.grid(True)

    # Plot 2: Difference over time
    plt.subplot(2, 1, 2)
    plt.plot(game_indices, differences, color='purple', linewidth=1)
    plt.axhline(0, color='black', linestyle='--',
                linewidth=1, label='No Difference')
    plt.fill_between(game_indices, 0, differences, where=(
        differences > 0), color='green', alpha=0.3)
    plt.fill_between(game_indices, 0, differences, where=(
        differences < 0), color='red', alpha=0.3)
    plt.xlabel('Game Index (Chronological)')
    plt.ylabel('Elo Difference (Predicted - Actual)')
    plt.title(
        f'Elo Difference: Where Our Model Improves on Chess.com for {username}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'full_trajectory_{username}.png')
    plt.show()  # Display the plot
    plt.close()


def main():
    # Get username from command line or default to MagnusCarlsen
    username = sys.argv[1] if len(sys.argv) > 1 else "AnnaCramling"
    velocity_window = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Generate real dataset with velocity window
    generator = RealDataGenerator(username=username)
    dataset = generator.generate_dataset(velocity_window=velocity_window)
    print(f"Loaded {len(dataset)} games for {
          username} with velocity window {velocity_window}")

    # Skip evaluation if dataset is empty
    if not dataset:
        print("No games available for evaluation after filtering.")
        return

    # Run evaluation with multiple runs
    best, mses, baseline_mses, r2s = run_evaluation(
        dataset, num_runs=2, pop_size=50, ngen=25)
    print(f"Evolved MSEs: {mses}")
    print(f"Baseline MSEs: {baseline_mses}")

    # Statistical analysis
    results = statistical_analysis(mses, baseline_mses, r2s)
    print("\nStatistical Results:")
    print(f"Evolved Model MSE: {results['evolved_mean_mse']:.3f} ± {
          results['evolved_std_mse']:.3f}")
    print(f"Baseline MSE (predict velocity=0): {
          results['baseline_mean_mse']:.3f} ± {results['baseline_std_mse']:.3f}")
    print(f"Evolved R²: {results['evolved_mean_r2']:.3f} ± {
          results['evolved_std_r2']:.3f}")
    print(f"Improvement (lower MSE): {results['improvement']:.3f}")
    print(f"t-statistic: {results['t_stat']
          :.3f}, p-value: {results['p_value']:.3f}")
    if results['p_value'] < 0.05:
        print("Statistically significant improvement!")
    else:
        print("No significant improvement.")

    # Print best weights for interpretation
    print("Best weights (for momentum adjustment):")
    features = ['Win Streak', 'Recent Win Rate', 'Avg Accuracy',
                'Rating Trend', 'Games Last 30d', 'Velocity']
    for i, w in enumerate(best):
        print(f"  {features[i]}: {w:.3f}")
    print("Interpretation: Higher positive weights indicate stronger momentum impact.")

    # Generate plots for the last run's test set
    train, test = train_test_split(
        dataset, test_size=0.2, random_state=1)  # Use same seed as last run
    predictions = [game.pre_game_elo +
                   game.momentum_adjustment for game in test]
    actuals = [game.post_game_elo for game in test]
    print(f"Test set size: {len(test)}")
    print(f"Actual Elo range: {min(actuals):.1f} to {max(actuals):.1f}")
    print(f"Predicted Elo range: {
          min(predictions):.1f} to {max(predictions):.1f}")
    print(f"Sample actual vs predicted: {
          actuals[0]:.1f} vs {predictions[0]:.1f}")
    plot_velocity_scatter(test, predictions, actuals, username)
    plot_elo_trajectory(test, predictions, actuals, username)
    plot_feature_weights(best, username)
    plot_full_trajectory(dataset, best, username)
    print(
        "Plots saved: velocity_scatter_{username}.png, elo_trajectory_{username}.png, feature_weights_{username}.png, full_trajectory_{username}.png")


if __name__ == "__main__":
    main()
