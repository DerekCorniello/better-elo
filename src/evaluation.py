import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

from ea import evaluate_individual, run_evolution
from models import UserGameData

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def train_test_split(
    dataset: List[UserGameData], test_size: float = 0.2, random_state: int = 0
) -> Tuple[List[UserGameData], List[UserGameData]]:
    """
    Split dataset into train and test sets.

    Inputs:
    - dataset: list of UserGameData
    - test_size: float, fraction for test set (default 0.2)
    - random_state: int, random seed for reproducibility

    Outputs:
    - tuple: (train_data, test_data)

    Expected behavior:
    Shuffles and splits the dataset into training and testing portions.
    """
    if random_state is not None:
        random.seed(random_state)
    data = dataset[:]
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    train = data[:split_idx]
    test = data[split_idx:]
    return train, test


def evaluate_baseline(test_dataset: List[UserGameData]) -> float:
    """
    Evaluate baseline MSE: predict no momentum adjustment.

    Inputs:
    - test_dataset: list of UserGameData for testing

    Outputs:
    - float: mean squared error of baseline predictions

    Expected behavior:
    Calculates MSE assuming no Elo change (post_elo = pre_elo).
    """
    if not test_dataset:
        return 0.0
    mse = float(
        np.mean(
            [
                (game.post_game_elo - game.pre_game_elo) ** 2
                for game in test_dataset
            ]
        )
    )
    return mse


def run_evaluation(
    dataset: List[UserGameData],
    num_runs: int = 3,
    pop_size: int = 100,
    ngen: int = 50,
) -> Tuple[Any, List[float], List[float], List[float]]:
    """
    Run multiple evolutionary runs and collect metrics.

    Inputs:
    - dataset: list of UserGameData
    - num_runs: int, number of evolutionary runs
    - pop_size: int, population size
    - ngen: int, generations per run

    Outputs:
    - tuple: (best_individual, mses, baseline_mses, r2s)

    Expected behavior:
    Runs evolution multiple times, evaluates on test splits,
    returns best model and performance metrics.
    """
    mses = []
    baseline_mses = []
    r2s = []
    best = None
    for seed in range(num_runs):
        train, test = train_test_split(
            dataset, test_size=0.2, random_state=seed
        )

        # run evolution on train
        best = run_evolution(train, pop_size=pop_size, ngen=ngen)

        # evaluate on test
        mse = evaluate_individual(best, test)[0]
        mses.append(mse)

        # compute R2
        predictions = [
            game.pre_game_elo + game.momentum_adjustment for game in test
        ]
        actuals = [game.post_game_elo for game in test]
        r2 = compute_r2(predictions, actuals)
        r2s.append(r2)

        # baseline
        base_mse = evaluate_baseline(test)
        baseline_mses.append(base_mse)
    return best, mses, baseline_mses, r2s


def compute_r2(predictions: List[float], actuals: List[float]) -> float:
    """
    Compute R-squared for predictions.

    Inputs:
    - predictions: list of predicted values
    - actuals: list of actual values

    Outputs:
    - float: R-squared value

    Expected behavior:
    Calculates coefficient of determination for regression quality.
    """
    ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
    ss_tot = np.sum((np.array(actuals) - np.mean(actuals)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


def statistical_analysis(
    mses: List[float], baseline_mses: List[float], r2s: List[float]
) -> Dict:
    """
    Perform statistical analysis on MSE results.

    Inputs:
    - mses: list of momentum system MSEs
    - baseline_mses: list of baseline MSEs
    - r2s: list of R-squared values

    Outputs:
    - dict: statistical metrics including p-values, improvements

    Expected behavior:
    Computes means, stds, t-tests, and improvement statistics.
    """
    mse_mean = np.mean(mses)
    mse_std = np.std(mses)
    base_mean = np.mean(baseline_mses)
    base_std = np.std(baseline_mses)
    improvement = base_mean - mse_mean  # lower MSE is better
    # paired t-test for statistical significance

    t_stat, p_value = stats.ttest_rel(mses, baseline_mses)
    r2_mean = np.mean(r2s) if r2s else None
    r2_std = np.std(r2s) if r2s else None
    return {
        "evolved_mean_mse": mse_mean,
        "evolved_std_mse": mse_std,
        "baseline_mean_mse": base_mean,
        "baseline_std_mse": base_std,
        "improvement": improvement,
        "evolved_mean_r2": r2_mean,
        "evolved_std_r2": r2_std,
        "t_stat": t_stat,
        "p_value": p_value,
    }
