import math
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MomentumRating:
    """
    Momentum-enhanced rating system that prevents rating cavities.

    Attributes:
    - base_elo: float, traditional Elo rating
    - momentum_adjustment: float, adjustment based on form
    - adaptive_K: float, adaptive K-factor for rating updates
    """

    base_elo: float
    momentum_adjustment: float = 0.0
    adaptive_K: float = 32.0

    @property
    def momentum_elo(self) -> float:
        """
        Get the momentum-enhanced rating.

        Returns:
        - float: base_elo + momentum_adjustment
        """
        return self.base_elo + self.momentum_adjustment

    def calculate_win_probability(self, opponent_elo: float) -> float:
        """
        Calculate win probability using momentum-enhanced rating.

        Inputs:
        - opponent_elo: float, opponent's rating

        Returns:
        - float: win probability (0-1)
        """
        return 1.0 / (
            1.0 + 10.0 ** ((opponent_elo - self.momentum_elo) / 400.0)
        )

    def calculate_adaptive_K(
        self, features: List[float], weights: List[float]
    ) -> float:
        """
        Calculate adaptive K-factor based on momentum indicators.

        Inputs:
        - features: list of momentum features
        - weights: list of feature weights

        Returns:
        - float: adaptive K-factor

        Expected behavior:
        Adjusts K-factor based on rating trend, velocity, and win streaks.
        Higher momentum indicators result in faster rating adjustments.
        """

        K = 32.0
        momentum_multiplier = 1.0

        # positive rating trend increases K
        #   (faster adjustment for improving players)
        rating_trend_weight = weights[3] if len(weights) > 3 else 0.0
        if rating_trend_weight > 0 and features[3] > 0:  # positive trend
            momentum_multiplier += (
                abs(features[3] * rating_trend_weight) / 100.0
            )

        # high velocity increases K
        #   (consistent performance gets faster adjustment)
        velocity_weight = weights[5] if len(weights) > 5 else 0.0
        if velocity_weight > 0 and abs(features[5]) > 1:
            momentum_multiplier += abs(features[5] * velocity_weight) / 50.0

        # win streaks affect K
        #   (hot/cold streaks get faster adjustment)
        win_streak_weight = weights[0] if len(weights) > 0 else 0.0
        if abs(features[0]) >= 3:  # Significant streak
            momentum_multiplier += abs(features[0] * win_streak_weight) / 20.0

        # cap K-factor to prevent extreme adjustments
        return min(max(K * momentum_multiplier, 16.0), 64.0)


@dataclass
class RatingCavityMetrics:
    """
    Metrics to detect and measure rating cavities.

    Attributes:
    - cavity_duration: int, games spent in cavity
    - cavity_depth: float, how far rating is from true performance
    - recovery_time: int, games to escape cavity
    - plateau_episodes: int, number of plateau periods
    """

    cavity_duration: int = 0  # games spent in cavity
    cavity_depth: float = 0.0  # how far rating is from true performance
    recovery_time: int = 0  # games to escape cavity
    plateau_episodes: int = 0  # number of plateau periods

    def is_in_cavity(
        self, performance_gap: float, threshold: float = 50.0
    ) -> bool:
        """
        Check if player is currently in a rating cavity.

        Inputs:
        - performance_gap: float, gap between actual and expected performance
        - threshold: float, minimum gap to consider cavity (default 50.0)

        Returns:
        - bool: True if in cavity

        Expected behavior:
        Compares absolute performance gap against threshold.
        """
        return abs(performance_gap) > threshold

    def calculate_performance_gap(
        self, actual_win_rate: float, expected_win_rate: float
    ) -> float:
        """
        Calculate performance gap between actual and expected win rates.

        Inputs:
        - actual_win_rate: float, observed win rate
        - expected_win_rate: float, expected win rate

        Returns:
        - float: Elo difference equivalent to the gap
        """
        if expected_win_rate == 0 or expected_win_rate == 1:
            return 0.0
        elo_diff = -400 * math.log10(
            (1 / expected_win_rate - 1) / (1 / actual_win_rate - 1)
        )
        return elo_diff


class TemporalValidator:
    """
    Temporal cross-validation for future prediction evaluation.

    Provides methods for time-based dataset splitting and evaluation.
    """

    @staticmethod
    def temporal_split(dataset: List[Any], train_ratio: float = 0.7) -> tuple:
        """
        Split dataset by time (train on past, test on future).

        Inputs:
        - dataset: list of game data
        - train_ratio: float, fraction for training (default 0.7)

        Outputs:
        - tuple: (train_data, test_data)

        Expected behavior:
        Sorts by end_time and splits chronologically.
        """
        sorted_data = sorted(dataset, key=lambda g: g.end_time)
        split_idx = int(len(sorted_data) * train_ratio)
        return sorted_data[:split_idx], sorted_data[split_idx:]

    @staticmethod
    def calculate_future_prediction_accuracy(
        test_dataset: List[Any], momentum_weights: List[float]
    ) -> Dict[str, float]:
        """
        Evaluate how well momentum system predicts future game outcomes.

        Inputs:
        - test_dataset: list of game data
        - momentum_weights: list of weights

        Returns:
        - Dict[str, float]: accuracy and brier score metrics

        Expected behavior:
        Computes prediction accuracy and calibration on test data.
        """
        correct_predictions = 0
        total_games = len(test_dataset)
        brier_score = 0.0

        for game in test_dataset:
            # calculate momentum-enhanced win probability
            momentum_rating = MomentumRating(game.pre_game_elo)
            momentum_rating.momentum_adjustment = sum(
                w * f
                for w, f in zip(momentum_weights, game.to_feature_vector())
            )

            predicted_prob = momentum_rating.calculate_win_probability(
                game.opponent_elo
            )

            actual_result = game.actual_result
            predicted_win = 1 if predicted_prob > 0.5 else 0
            actual_win = 1 if actual_result > 0.5 else 0

            if predicted_win == actual_win:
                correct_predictions += 1

            # Brier score for probability calibration
            brier_score += (predicted_prob - actual_result) ** 2

        accuracy = correct_predictions / total_games if total_games > 0 else 0.0
        brier_score = brier_score / total_games if total_games > 0 else 0.0

        return {
            "accuracy": accuracy,
            "brier_score": brier_score,
            "total_games": total_games,
        }

    @staticmethod
    def detect_rating_cavities(
        dataset: List[Any], momentum_weights: List[float]
    ) -> List[RatingCavityMetrics]:
        """
        Detect periods where players are stuck in rating cavities.

        Inputs:
        - dataset: list of game data
        - momentum_weights: list of weights

        Returns:
        - List[RatingCavityMetrics]: detected cavities

        Expected behavior:
        Analyzes performance gaps to identify rating stagnation.
        """
        cavities = []
        current_cavity = None

        for i, game in enumerate(dataset):
            momentum_rating = MomentumRating(game.pre_game_elo)
            momentum_rating.momentum_adjustment = sum(
                w * f
                for w, f in zip(momentum_weights, game.to_feature_vector())
            )

            expected_win_rate = momentum_rating.calculate_win_probability(
                game.opponent_elo
            )

            # use recent performance to estimate true skill level
            recent_games = dataset[max(0, i - 10) : i]
            if len(recent_games) >= 5:
                recent_actual = sum(
                    g.actual_result for g in recent_games
                ) / len(recent_games)
                performance_gap = abs(recent_actual - expected_win_rate)

                if performance_gap > 0.1:  # significant performance gap
                    if current_cavity is None:
                        current_cavity = RatingCavityMetrics()
                        current_cavity.cavity_duration = 1
                    else:
                        current_cavity.cavity_duration += 1
                else:
                    if current_cavity is not None:
                        cavities.append(current_cavity)
                        current_cavity = None

        if current_cavity is not None:
            cavities.append(current_cavity)

        return cavities
