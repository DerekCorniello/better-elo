import os
import sys

import numpy as np

from src.config import config
from src.data_generator import RealDataGenerator
from src.ea import evaluate_individual, run_evolution
from src.novel_momentum_system import (
    NovelMomentumRating,
    NovelMomentumSystem,
    NovelTemporalValidator,
    evaluate_future_prediction_accuracy,
    evaluate_direct_comparison,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)


def load_real_player_data(players: list, velocity_window: int = 10) -> dict:
    """
    Load and process real player datasets for momentum analysis.

    Inputs:
    - players: list of player usernames to load data for
    - velocity_window: int, window size for velocity calculation (default 10)

    Outputs:
    - dict: mapping of player names to their processed game datasets

    Expected behavior:
    Loads JSON game data for each player, processes it using RealDataGenerator,
    and returns a dictionary of datasets. Prints progress and errors.
    """
    datasets = {}

    for player in players:
        try:
            print(f"Loading data for {player}...")
            if RealDataGenerator is None:
                print(f"RealDataGenerator not available for {player}")
                datasets[player] = []
                continue
            generator = RealDataGenerator(username=player)
            raw_dataset = generator.generate_dataset(
                velocity_window=velocity_window
            )

            # games are already processed by RealDataGenerator with
            # opponent_elo and actual_result
            processed_games = raw_dataset

            datasets[player] = processed_games
            print(f"Processed {len(processed_games)} games for {player}")

        except Exception as e:
            print(f"Failed to load data for {player}: {e}")
            datasets[player] = []

    return datasets


def train_momentum_system(
    dataset, pop_size: int | None = None, ngen: int | None = None, num_runs: int | None = None
):
    """
    Train momentum system using multi-run evolutionary algorithms
        to prevent local minima.

    Inputs:
    - dataset: list of UserGameData objects for training
    - pop_size: int, population size for evolutionary algorithm
    - ngen: int, number of generations per run
    - num_runs: int, number of independent evolutionary runs

    Outputs:
    - list: best momentum weights found across all runs

    Expected behavior:
    Runs multiple evolutionary optimization runs, selects the best weights,
    and returns them. Prints training progress and final results.
    """
    # Use config defaults if not provided
    if pop_size is None:
        pop_size = config.evolution["population_size"]
    if ngen is None:
        ngen = config.evolution["generations"]
    if num_runs is None:
        num_runs = config.evolution["num_runs"]

    print(
        f"Training momentum system: {len(dataset)} games, "
        f"pop_size={pop_size}, ngen={ngen}, runs={num_runs}"
    )

    if len(dataset) < 20:
        print("Insufficient data for training")
        return [1.0, 0.5, 0.1, 2.0, -0.5, 1.5]

    best_overall_weights = None
    best_overall_fitness = float("inf")

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        try:
            weights = run_evolution(dataset, pop_size=pop_size, ngen=ngen)
            final_fitness = evaluate_individual(weights, dataset)[0]
            print(f"Run {run + 1} fitness: {final_fitness:.4f}")

            if final_fitness < best_overall_fitness:
                best_overall_fitness = final_fitness
                best_overall_weights = list(weights)
                print(f"New best fitness: {final_fitness:.4f}")

        except Exception as e:
            print(f"Run {run + 1} failed: {e}")
            continue

    if best_overall_weights is None:
        print("All evolutionary runs failed, using fallback weights")
        return [2.5, -0.8, 25.0, 8.5, -1.2, 45.0]

    print(f"Best fitness: {best_overall_fitness:.4f}")
    return best_overall_weights


def validate_temporal_prediction(
    dataset, momentum_weights, prediction_horizon: int = 50
):
    """
    Perform true temporal validation with prediction horizon.

    Inputs:
    - dataset: list of UserGameData objects
    - momentum_weights: list of 6 floats for momentum features
    - prediction_horizon: int, games to skip between sets (default 50)

    Outputs:
    - dict: validation metrics including accuracy, brier_score, total_games

    Expected behavior:
    Splits data temporally, trains on past games, validates on future games,
    and returns prediction metrics. Prints validation results.
    """
    print(f"Temporal validation: horizon {prediction_horizon}")

    if len(dataset) < prediction_horizon + 50:
        print("Insufficient data for temporal validation")
        return {"accuracy": 0.0, "brier_score": 1.0, "total_games": 0}

    train_data, future_test_data = (
        NovelTemporalValidator.create_prediction_horizon_split(
            dataset, horizon=prediction_horizon
        )
    )

    print(f"Train: {len(train_data)}, Test: {len(future_test_data)}")

    momentum_system = NovelMomentumSystem()
    momentum_system.momentum_weights = momentum_weights

    for game in train_data:
        momentum_system.update_after_game(
            game.username,
            "opponent",
            game.actual_result,
            game.to_feature_vector(),
            momentum_weights,
        )

    future_metrics = evaluate_future_prediction_accuracy(
        future_test_data, momentum_weights
    )

    print(
        f"Accuracy: {future_metrics['accuracy']:.1%}, "
        f"Brier: {future_metrics['brier_score']:.3f}"
    )
    return future_metrics


def validate_cross_player_transfer(
    train_datasets: dict, test_player: str, momentum_weights
):
    """
    Test if momentum patterns transfer across players.

    Inputs:
    - train_datasets: dict of player datasets for training
    - test_player: str, player name to test on
    - momentum_weights: list of 6 floats for momentum features

    Outputs:
    - dict: transfer validation metrics

    Expected behavior:
    Trains on multiple players, tests on unseen player, returns
        transfer accuracy.
    Prints transfer results.
    """
    print(
        f"Cross-player validation: train on {list(train_datasets.keys())}, "
        f"test on {test_player}"
    )

    if test_player not in train_datasets and len(train_datasets) == 0:
        print("Insufficient data for cross-player validation")
        return {"accuracy": 0.0, "brier_score": 1.0, "total_games": 0}

    momentum_system = NovelMomentumSystem()
    momentum_system.momentum_weights = momentum_weights

    total_train_games = 0
    for player, dataset in train_datasets.items():
        if player != test_player:
            for game in dataset[:100]:
                momentum_system.update_after_game(
                    game.username,
                    "opponent",
                    game.actual_result,
                    game.to_feature_vector(),
                    momentum_weights,
                )
                total_train_games += 1

    print(f"Trained on {total_train_games} games")

    test_dataset = train_datasets.get(test_player, [])
    if not test_dataset:
        print("No test data")
        return {"accuracy": 0.0, "brier_score": 1.0, "total_games": 0}

    transfer_metrics = evaluate_future_prediction_accuracy(
        test_dataset[:50], momentum_weights
    )

    print(f"Transfer accuracy: {transfer_metrics['accuracy']:.1%}")
    return transfer_metrics


def analyze_cavity_prevention(dataset, momentum_weights):
    """
    Analyze how well momentum system prevents rating cavities.

    Inputs:
    - dataset: list of UserGameData objects
    - momentum_weights: list of 6 floats for momentum features

    Outputs:
    - dict: cavity metrics including episodes, duration, frequency

    Expected behavior:
    Evaluates rating stability, detects cavity episodes, returns metrics.
    Prints cavity analysis results.
    """
    print("Cavity prevention analysis")

    if len(dataset) < 100:
        print("Insufficient data for cavity analysis")
        return {
            "cavity_episodes": 0,
            "avg_cavity_duration": 0.0,
            "cavity_frequency": 0.0,
        }

    cavity_metrics = NovelTemporalValidator.evaluate_cavity_prevention(
        dataset, momentum_weights
    )

    print(
        f"Cavities: {cavity_metrics['cavity_episodes']}, "
        f"Avg duration: {cavity_metrics['avg_cavity_duration']:.1f}, "
        f"Frequency: {cavity_metrics['cavity_frequency']:.3f}"
    )
    return cavity_metrics


def train_multi_player_momentum_system(
    train_datasets: dict, val_player: str
) -> dict:
    """
    Train momentum system on multiple players,
    validate on one held-out player
    """
    print(
        f"\nMULTI-PLAYER TRAINING: \
        {list(train_datasets.keys())} -> {val_player}"
    )

    # Combine all training games from multiple players
    all_train_games = []
    for player, dataset in train_datasets.items():
        all_train_games.extend(dataset)
        print(f"  {player}: {len(dataset)} games")

    print(f"  Total training games: {len(all_train_games)}")

    # Train evolutionary algorithm on combined multi-player data (multi-run)
    momentum_weights = train_momentum_system(all_train_games)

    # Validate on held-out player
    if val_player in train_datasets and len(train_datasets[val_player]) > 50:
        val_results = validate_temporal_prediction(
            train_datasets[val_player], momentum_weights, prediction_horizon=30
        )

        cavity_results = analyze_cavity_prevention(
            train_datasets[val_player], momentum_weights
        )

        return {
            "weights": momentum_weights,
            "validation_accuracy": val_results["accuracy"],
            "brier_score": val_results["brier_score"],
            "total_games_validated": val_results["total_games"],
            "cavity_episodes": cavity_results["cavity_episodes"],
            "cavity_frequency": cavity_results["cavity_frequency"],
            "avg_cavity_duration": cavity_results["avg_cavity_duration"],
        }
    else:
        return {
            "weights": momentum_weights,
            "validation_accuracy": 0.0,
            "error": f"Insufficient validation data for {val_player}",
        }


def run_player_specific_validation() -> dict:
    """
    Run player-specific momentum model training and validation.

    Inputs:
    - None (uses hardcoded players list)

    Outputs:
    - dict: results for each player including accuracy, weights, stats, cavity

    Expected behavior:
    For each player, loads data, trains model, validates temporally,
    computes stats and cavity metrics, returns comprehensive results.
    Prints progress and warnings.
    """
    print("Player-specific momentum model validation")
    print(
        "WARNING: This will take considerable time (20-40 hours) with "
        "enhanced parameters!"
    )

    players = ["MagnusCarlsen"]
    results = {}

    for player in players:
        print(f"Training {player} model")

        datasets = load_real_player_data([player], velocity_window=10)

        if not datasets.get(player) or len(datasets[player]) < 100:
            print(f"Insufficient data for {player}")
            continue

        player_games = datasets[player]
        print(f"Dataset: {len(player_games)} games")

        for game in player_games:
            if game.actual_result != 0.5:
                expected = 1 / (
                    1 + 10 ** ((game.opponent_elo - game.pre_game_elo) / 400)
                )
                actual = game.actual_result
                K = 20
                delta = K * (expected - actual)
                game.opponent_elo -= delta

        train_data, future_test_data = (
            NovelTemporalValidator.create_prediction_horizon_split(
                player_games, horizon=50
            )
        )

        momentum_weights = train_momentum_system(train_data)

        validation_results = evaluate_direct_comparison(
            future_test_data, momentum_weights
        )

        momentum_predictions = []
        elo_predictions = []
        actual_results = []

        for game in future_test_data:
            elo_prob = 1 / (
                1 + 10 ** ((game.opponent_elo - game.pre_game_elo) / 400)
            )
            features = game.to_feature_vector()
            momentum_adjustment = sum(
                w * f for w, f in zip(momentum_weights, features)
            )
            momentum_adjustment = max(-0.2, min(0.2, momentum_adjustment))
            enhanced_prob = elo_prob + momentum_adjustment
            enhanced_prob = max(0.01, min(0.99, enhanced_prob))
            momentum_predictions.append(enhanced_prob)
            elo_predictions.append(elo_prob)
            actual_results.append(game.actual_result)

        stats_results = calculate_statistical_significance(
            momentum_predictions, elo_predictions, actual_results
        )
        cavity_results = analyze_cavity_prevention(
            player_games, momentum_weights
        )

        results[player] = {
            "future_accuracy": validation_results["momentum_accuracy"],
            "elo_accuracy": validation_results["elo_accuracy"],
            "improvement": validation_results["improvement"],
            "relative_improvement": validation_results["relative_improvement"],
            "total_games_validated": validation_results["total_games"],
            "momentum_correct": validation_results["momentum_correct"],
            "elo_correct": validation_results["elo_correct"],
            "weights": momentum_weights,
            "stats": stats_results,
            "cavity": cavity_results,
        }

        print(f"{player} model trained and validated")

    return results


def calculate_statistical_significance(
    momentum_predictions, elo_predictions, actual_results
):
    """
    Calculate statistical significance using McNemar's test.

    Inputs:
    - momentum_predictions: list of predicted probabilities
        from momentum system
    - elo_predictions: list of predicted probabilities from Elo system
    - actual_results: list of actual game outcomes (0/1)

    Outputs:
    - dict: statistical test results including p-value, confidence intervals

    Expected behavior:
    Performs McNemar's test on binary predictions, computes bootstrap CI,
    returns significance metrics.
    """
    # Convert to binary predictions
    momentum_binary = [1 if p > 0.5 else 0 for p in momentum_predictions]
    elo_binary = [1 if p > 0.5 else 0 for p in elo_predictions]
    actual_binary = [1 if r > 0.5 else 0 for r in actual_results]

    # Create contingency table for McNemar's test
    momentum_correct = [m == a for m, a in zip(momentum_binary, actual_binary)]
    elo_correct = [e == a for e, a in zip(elo_binary, actual_binary)]

    # Count cases
    b = sum(
        1 for m, e in zip(momentum_correct, elo_correct) if m and not e
    )  # Momentum right, Elo wrong
    c = sum(
        1 for m, e in zip(momentum_correct, elo_correct) if not m and e
    )  # Elo right, Momentum wrong

    # Simple significance test
    if b + c > 0:
        # Exact binomial test for small samples
        n = b + c
        k = min(b, c)
        # Calculate two-sided p-value
        from math import comb

        p_value = 2 * sum(comb(n, i) * (0.5**n) for i in range(k + 1))
        p_value = min(p_value, 1.0)
    else:
        p_value = 1.0

    # Simple bootstrap CI for accuracy difference
    n_bootstrap = 1000
    accuracy_diffs = []
    n = len(actual_results)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        momentum_acc = (
            sum(momentum_binary[i] == actual_binary[i] for i in indices) / n
        )
        elo_acc = sum(elo_binary[i] == actual_binary[i] for i in indices) / n
        accuracy_diffs.append(momentum_acc - elo_acc)

    accuracy_diffs.sort()
    ci_lower = accuracy_diffs[int(0.025 * n_bootstrap)]
    ci_upper = accuracy_diffs[int(0.975 * n_bootstrap)]
    mean_diff = np.mean(accuracy_diffs)

    return {
        "mcnemar_p_value": p_value,
        "accuracy_diff_ci_lower": ci_lower,
        "accuracy_diff_ci_upper": ci_upper,
        "accuracy_diff_mean": mean_diff,
        "statistically_significant": p_value < 0.05,
    }


def aggregate_player_specific_results(results_dict: dict) -> dict:
    """
    Aggregate and analyze results across all players.

    Inputs:
    - results_dict: dict of individual player results

    Outputs:
    - dict: aggregated metrics including mean accuracy,
            improvement, significance

    Expected behavior:
    Computes averages and statistics across players, filters valid results,
    returns summary metrics.
    """
    if not results_dict:
        return {}

    # Extract valid results (exclude errors)
    valid_results = [
        r
        for r in results_dict.values()
        if "future_accuracy" in r and r["future_accuracy"] > 0
    ]

    if not valid_results:
        return {"error": "No valid results to aggregate"}

    accuracies = [r["future_accuracy"] for r in valid_results]
    elo_accuracies = [r.get("elo_accuracy", 0) for r in valid_results]
    improvements = [r.get("improvement", 0) for r in valid_results]
    significant_players = (
        len(
            [
                r
                for r in valid_results
                if r.get("statistical_significance", {}).get(
                    "statistically_significant", False
                )
            ]
        ),
    )
    significance_rate = (
        significant_players / len(valid_results) if valid_results else 0
    )

    return {
        "players_tested": len(valid_results),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_elo_accuracy": float(np.mean(elo_accuracies)),
        "mean_improvement": float(np.mean(improvements)),
        "std_improvement": float(np.std(improvements)),
        "statistically_significant_players": significant_players,
        "significance_rate": significance_rate,
    }


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--player-specific":
        results = run_player_specific_validation()
        aggregated = aggregate_player_specific_results(results)

        if results:
            print("\nPlayer-specific results:")
            for player, data in results.items():
                print(
                    f"{player}: Accuracy {data['future_accuracy']:.1%}, "
                    f"Improvement {data.get('improvement', 0):.1%}"
                )

            if aggregated and "mean_accuracy" in aggregated:
                print(
                    f"Mean accuracy: {aggregated['mean_accuracy']:.1%}, "
                    f"Mean improvement: {aggregated['mean_improvement']:.1%}"
                )
        else:
            print("Player-specific validation failed")

    else:
        players = ["MagnusCarlsen"]
        datasets = load_real_player_data(players, velocity_window=10)

        if not datasets or len(datasets.get("MagnusCarlsen", [])) < 100:
            print("Insufficient Magnus Carlsen data.")
            return

        momentum_weights = train_momentum_system(datasets["MagnusCarlsen"])

        temporal_results = validate_temporal_prediction(
            datasets["MagnusCarlsen"], momentum_weights, prediction_horizon=50
        )

        cavity_results = analyze_cavity_prevention(
            datasets["MagnusCarlsen"], momentum_weights
        )

        print(
            f"Accuracy: {temporal_results['accuracy']:.1%}, "
            f"Cavity freq: {cavity_results['cavity_frequency']:.3f}"
        )
        improvement = (temporal_results["accuracy"] - 0.5) / 0.5 * 100
        print(f"Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
