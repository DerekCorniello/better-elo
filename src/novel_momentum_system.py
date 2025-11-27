from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import math
import numpy as np


@dataclass
class NovelMomentumRating:
    """
    Independent momentum-based rating system that prevents rating cavities.

    Attributes:
    - player_id: str, unique player identifier
    - base_rating: float, traditional rating component
    - momentum_score: float, momentum adjustment
    - rating_history: List[float], historical ratings
    """
    player_id: str
    base_rating: float = 1500.0
    momentum_score: float = 0.0
    rating_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.rating_history:
            self.rating_history = [self.base_rating]

    @property
    def momentum_rating(self) -> float:
        """
        Get the momentum-enhanced rating.

        Returns:
        - float: base_rating + momentum_score
        """
        return self.base_rating + self.momentum_score

    def calculate_win_probability(self, opponent_rating: float) -> float:
        """
        Calculate win probability using momentum-enhanced rating.

        Inputs:
        - opponent_rating: float, opponent's rating

        Returns:
        - float: win probability (0-1)
        """
        # Bound the exponent to prevent overflow
        exponent = (opponent_rating - self.momentum_rating) / 400.0
        exponent = max(min(exponent, 10.0), -10.0)  # Bound between -10 and 10
        return 1.0 / (1.0 + 10.0 ** exponent)

    def update_momentum(self, momentum_features: List[float],
                        weights: List[float]) -> None:
        """
        Update momentum score based on features and weights.

        Inputs:
        - momentum_features: list of 6 feature values
        - weights: list of 6 weight values

        Outputs:
        - None (updates momentum_score)
        """
        momentum_change = sum(
            w * f for w, f in zip(weights, momentum_features))
        self.momentum_score = momentum_change

    def update_rating(self, opponent_rating: float, actual_result: float,
                      momentum_features: List[float],
                      weights: List[float]) -> None:
        expected_result = self.calculate_win_probability(opponent_rating)
        adaptive_K = self.calculate_adaptive_K(momentum_features, weights)
        rating_change = adaptive_K * (actual_result - expected_result)
        self.base_rating += rating_change
        self.update_momentum(momentum_features, weights)
        self.rating_history.append(self.momentum_rating)

    def calculate_adaptive_K(self, momentum_features: List[float],
                             weights: List[float]) -> float:
        """
        Calculate adaptive K-factor based on momentum.

        Inputs:
        - momentum_features: list of features
        - weights: list of weights

        Returns:
        - float: adaptive K-factor
        """

        base_K = 32.0
        momentum_multiplier = 1.0

        # Positive rating trend increases K
        # (faster adjustment for improving players)
        rating_trend_weight = weights[3] if len(weights) > 3 else 0.0
        if rating_trend_weight > 0 and momentum_features[3] > 0:
            momentum_multiplier += abs(
                momentum_features[3] * rating_trend_weight) / 100.0

        # High velocity increases K
        # (consistent performance gets faster adjustment)
        velocity_weight = weights[5] if len(weights) > 5 else 0.0
        if velocity_weight > 0 and abs(momentum_features[5]) > 1:
            momentum_multiplier += abs(
                momentum_features[5] * velocity_weight) / 50.0

        # Win streaks affect K
        # (hot/cold streaks get faster adjustment)
        win_streak_weight = weights[0] if len(weights) > 0 else 0.0
        if abs(momentum_features[0]) >= 3:
            momentum_multiplier += abs(
                momentum_features[0] * win_streak_weight) / 20.0

        # Cap K-factor to prevent extreme adjustments
        return min(max(base_K * momentum_multiplier, 16.0), 64.0)


class NovelMomentumSystem:
    """
    Complete momentum-based rating system that competes with traditional Elo
    """

    def __init__(self):
        self.players: Dict[str, NovelMomentumRating] = {}
        self.momentum_weights: List[float] = [0.0] * 6
        self.prediction_horizon: int = 30

    def add_player(self, player_id: str,
                   initial_rating: float = 1500.0) -> None:
        if player_id not in self.players:
            self.players[player_id] = NovelMomentumRating(
                player_id, initial_rating)

    def predict_future_trajectory(self, player_id: str,
                                  games_ahead: int = 10) -> List[float]:
        """
        Predict how player's rating will change over next N games.

        Inputs:
        - player_id: str, player to predict for
        - games_ahead: int, number of future games to predict

        Outputs:
        - List[float]: predicted rating after each game

        Expected behavior:
        Simulates future games with current momentum
        and predicts rating trajectory.
        """
        if player_id not in self.players:
            return []

        player = self.players[player_id]
        current_rating = player.momentum_rating
        trajectory = [current_rating]
        momentum_trend = player.momentum_score * 0.1  # decay factor

        for _ in range(games_ahead):
            # project next rating based on current momentum
            next_rating = trajectory[-1] + momentum_trend
            trajectory.append(next_rating)

            # momentum decay over time
            momentum_trend *= 0.95

        return trajectory[1:]  # exclude current rating

    def predict_game_outcome(self, player1_id: str,
                             player2_id: str) -> Tuple[float, float]:
        """
        Predict outcome between two players.

        Inputs:
        - player1_id: str, first player ID
        - player2_id: str, second player ID

        Returns:
        - Tuple[float, float]: (p1_win_prob, p2_win_prob)
        """
        if player1_id not in self.players or player2_id not in self.players:
            return 0.5, 0.5

        p1 = self.players[player1_id]
        p2 = self.players[player2_id]

        p1_win_prob = p1.calculate_win_probability(p2.momentum_rating)
        p2_win_prob = 1.0 - p1_win_prob

        return p1_win_prob, p2_win_prob

    def update_after_game(self, player1_id: str, player2_id: str,
                          player1_result: float, player1_features: List[float],
                          player2_features: List[float]) -> None:
        """
        Update ratings after a game.

        Inputs:
        - player1_id: str, first player ID
        - player2_id: str, second player ID
        - player1_result: float, 1.0 win, 0.5 draw, 0.0 loss for player1
        - player1_features: list of momentum features for player1
        - player2_features: list of momentum features for player2

        Outputs:
        - None (updates player ratings in self.players)

        Expected behavior:
        Calculates Elo updates and adjusts momentum scores
        based on game outcome.
        """
        # ensure players exist
        self.add_player(player1_id)
        self.add_player(player2_id)

        p1 = self.players[player1_id]
        p2 = self.players[player2_id]

        # update both players' ratings
        p1.update_rating(p2.momentum_rating, player1_result,
                         player1_features, self.momentum_weights)
        p2_result = 1.0 - player1_result  # Two-player zero-sum
        p2.update_rating(p1.momentum_rating, p2_result,
                         player2_features, self.momentum_weights)


def evaluate_future_prediction_accuracy(test_games: List[Any],
                                        weights: list) -> Dict[str, float]:
    """
    Evaluate prediction accuracy on future games.

    Inputs:
    - test_games: list of game data
    - weights: list of momentum weights

    Returns:
    - Dict[str, float]: accuracy metrics
    """
    if not test_games:
        return {'accuracy': 0.0, 'brier_score': 0.0, 'total_games': 0}

    total_games = len(test_games)
    predictions = []
    actual_results = []
    elo_predictions = []
    correct_predictions = 0
    brier_score = 0.0
    log_loss_sum = 0.0

    for game in test_games:
        # this is the way elo is usually predicted
        elo_expected = 1 / \
            (1 + 10 ** ((game.opponent_elo - game.pre_game_elo) / 400))

        # calculate momentum adjustment
        features = game.to_feature_vector()
        momentum_adjustment = sum(w * f for w, f in zip(weights, features))

        # dont allow the elo to drastically change the
        # game winner, clamp to 1-99%
        momentum_adjustment = max(-0.2, min(0.2, momentum_adjustment))
        enhanced_prob = elo_expected + momentum_adjustment
        enhanced_prob = max(0.01, min(0.99, enhanced_prob))
        predictions.append(enhanced_prob)
        elo_predictions.append(elo_expected)
        actual_results.append(game.actual_result)

        # binary prediction (0.5 threshold)
        predicted_win = 1 if enhanced_prob > 0.5 else 0
        actual_win = 1 if game.actual_result > 0.5 else 0

        if predicted_win == actual_win:
            correct_predictions += 1

        # Brier score
        brier_score += (enhanced_prob - game.actual_result) ** 2

        # log loss
        enhanced_prob_clipped = max(
            1e-15, min(1 - 1e-15, enhanced_prob))  # Prevent log(0)
        log_loss_sum += -(game.actual_result *
                          math.log(enhanced_prob_clipped) +
                          (1 - game.actual_result) *
                          math.log(1 - enhanced_prob_clipped))

    # Basic metrics
    accuracy = correct_predictions / total_games
    brier_score = brier_score / total_games
    log_loss = log_loss_sum / total_games

    # Elo baseline metrics
    elo_correct = sum(1 for i, pred in enumerate(elo_predictions) if
                      (pred > 0.5 and actual_results[i] > 0.5) or
                      (pred <= 0.5 and actual_results[i] <= 0.5))
    elo_accuracy = elo_correct / total_games

    elo_brier = (sum((pred - actual) ** 2 for pred,
                     actual in zip(elo_predictions, actual_results))
                 / total_games)

    try:
        auc_roc = roc_auc_score(actual_results, predictions)
        elo_auc = roc_auc_score(actual_results, elo_predictions)
    except Exception:
        auc_roc = 0.5
        elo_auc = 0.5

    # calibration metrics
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actual_results, predictions, n_bins=10)
        calibration_error = np.mean(
            np.abs(fraction_of_positives - mean_predicted_value))
    except Exception:
        calibration_error = 0.0

    # different threshold accuracies
    threshold_accuracies = {}
    for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
        thresh_correct = sum(1 for pred, actual in
                             zip(predictions, actual_results)
                             if (pred > threshold and actual > 0.5) or
                             (pred <= threshold and actual <= 0.5))
        threshold_accuracies[f'threshold_{
            threshold}'] = thresh_correct / total_games

    # confidence-weighted accuracy
    confidence_weighted_acc = sum(
        abs(pred - 0.5) * 2 * ((pred > 0.5 and actual > 0.5)
                               or (pred <= 0.5 and actual <= 0.5))
        for pred, actual in zip(predictions, actual_results)
    ) / sum(abs(pred - 0.5) * 2 for pred in predictions) if predictions else 0

    return {
        'accuracy': accuracy,
        'brier_score': brier_score,
        'log_loss': log_loss,
        'auc_roc': auc_roc,
        'calibration_error': calibration_error,
        'confidence_weighted_accuracy': confidence_weighted_acc,
        'total_games': total_games,

        # baseline comparisons
        'elo_accuracy': elo_accuracy,
        'elo_brier_score': elo_brier,
        'elo_auc_roc': elo_auc,

        # improvements
        'accuracy_improvement': accuracy - elo_accuracy,
        'brier_improvement': elo_brier - brier_score,  # Lower is better
        'auc_improvement': auc_roc - elo_auc,

        **threshold_accuracies,

        # Precision: True Positives / (True Positives + False Positives)
        # Measures accuracy of positive predictions,
        #   predicted wins that were wins
        'precision': (sum(1 for pred, actual in zip(predictions,
                                                    actual_results)
                          if pred > 0.5 and actual > 0.5) /
                      sum(1 for pred in predictions if pred > 0.5)
                      if sum(1 for pred in predictions if pred > 0.5) >
                      0 else 0),
        # Recall: True Positives / (True Positives + False Negatives)
        # Measures completeness of positive predictions,
        #   actual wins that were predicted
        'recall': (sum(1 for pred, actual in zip(predictions, actual_results)
                       if pred > 0.5 and actual > 0.5) /
                   sum(1 for actual in actual_results if actual > 0.5)
                   if sum(1 for actual in actual_results if actual > 0.5)
                   > 0 else 0),
        'f1_score': 0.0  # Will be calculated below
    }


def detect_cavities(momentum_weights: list, game_history: List[Any],
                    threshold: float = 0.15) -> List[dict]:
    """
    Detect rating cavities in game history.

    Inputs:
    - momentum_weights: list of weights
    - game_history: list of game data
    - threshold: float, cavity detection threshold

    Returns:
    - List[dict]: detected cavities
    """
    cavities = []

    for i in range(20, len(game_history)):  # need 20 games history
        # calculate recent actual performance
        recent_games = game_history[i-20:i]
        actual_win_rate = sum(g.actual_result for g in recent_games) / 20
        current_game = game_history[i]

        # calculate momentum-enhanced prediction
        elo_expected = (1 /
                        (1 + 10 **
                         ((current_game.opponent_elo -
                           current_game.pre_game_elo)
                          / 400)))
        features = current_game.to_feature_vector()
        momentum_adjustment = sum(
            w * f for w, f in zip(momentum_weights, features))
        momentum_adjustment = max(-0.2, min(0.2, momentum_adjustment))
        expected_win_rate = elo_expected + momentum_adjustment
        expected_win_rate = max(0.01, min(0.99, expected_win_rate))

        # detect cavity
        performance_gap = abs(actual_win_rate - expected_win_rate)
        if performance_gap > threshold:
            cavities.append({
                'game_index': i,
                'performance_gap': performance_gap,
                'actual_rate': actual_win_rate,
                'expected_rate': expected_win_rate,
                'type': 'underrated' if (actual_win_rate
                                         > expected_win_rate) else 'overrated'
            })

    return cavities


def calculate_cavity_metrics(momentum_weights: list,
                             game_history: List[Any]) -> dict:
    """
    Calculate comprehensive cavity metrics for fitness evaluation.

    Inputs:
    - momentum_weights: list of momentum weights
    - game_history: list of game data

    Outputs:
    - dict: cavity metrics including frequency, duration, etc.

    Expected behavior:
    Detects cavities and computes statistical metrics for rating stability.
    """
    cavities = detect_cavities(momentum_weights, game_history)

    if not cavities:
        return {
            'frequency': 0.0,
            'avg_duration': 0.0,
            'max_gap': 0.0,
            'total_episodes': 0
        }

    # calculate cavity duration (consecutive cavity games)
    cavity_episodes = []
    current_episode = []

    for i, cavity in enumerate(cavities):
        if i == 0 or cavity['game_index'] != cavities[i-1]['game_index'] + 1:
            # New cavity episode
            if current_episode:
                cavity_episodes.append(current_episode)
            current_episode = [cavity]
        else:
            # continuation of current episode
            current_episode.append(cavity)

    # add final episode
    if current_episode:
        cavity_episodes.append(current_episode)

    # calculate some metrics
    total_games = len(game_history)
    cavity_frequency = len(cavities) / total_games
    avg_duration = sum(len(episode) for episode in cavity_episodes) / \
        len(cavity_episodes) if cavity_episodes else 0
    max_gap = max(cavity['performance_gap'] for cavity in cavities)

    return {
        'frequency': cavity_frequency,
        'avg_duration': avg_duration,
        'max_gap': max_gap,
        'total_episodes': len(cavity_episodes),
        'cavities': cavities
    }


def evaluate_direct_comparison(test_games: List[Any],
                               weights: list) -> Dict[str, Any]:
    """
    Direct comparison: Momentum system vs traditional Elo.

    Inputs:
    - test_games: list of game data for evaluation
    - weights: list of momentum weights

    Outputs:
    - Dict[str, Any]: comparison metrics including accuracies, improvements

    Expected behavior:
    Evaluates both systems on the same games and computes improvement metrics.
    """
    if not test_games:
        return {'momentum_accuracy': 0.0, 'elo_accuracy': 0.0,
                'improvement': 0.0, 'total_games': 0}

    momentum_correct = 0
    elo_correct = 0
    total_games = len(test_games)

    for game in test_games:
        # calculate momentum prediction
        elo_expected = 1 / \
            (1 + 10 ** ((game.opponent_elo - game.pre_game_elo) / 400))
        features = game.to_feature_vector()
        momentum_adjustment = sum(w * f for w, f in zip(weights, features))
        momentum_adjustment = max(-0.2, min(0.2, momentum_adjustment))
        enhanced_prob = elo_expected + momentum_adjustment
        enhanced_prob = max(0.01, min(0.99, enhanced_prob))

        elo_prob = elo_expected
        momentum_win = 1 if enhanced_prob > 0.5 else 0
        elo_win = 1 if elo_prob > 0.5 else 0
        actual_win = 1 if game.actual_result > 0.5 else 0

        if momentum_win == actual_win:
            momentum_correct += 1
        if elo_win == actual_win:
            elo_correct += 1

    # Calculate accuracies and improvement
    momentum_accuracy = momentum_correct / total_games
    elo_accuracy = elo_correct / total_games
    improvement = momentum_accuracy - elo_accuracy

    return {
        'momentum_accuracy': float(momentum_accuracy),
        'elo_accuracy': float(elo_accuracy),
        'improvement': float(improvement),
        'relative_improvement': float((improvement / elo_accuracy * 100)
                                      if elo_accuracy > 0 else 0),
        'total_games': float(total_games),
        'momentum_correct': float(momentum_correct),
        'elo_correct': float(elo_correct)
    }


class NovelTemporalValidator:
    """Advanced temporal validation for true future prediction"""

    @staticmethod
    def create_prediction_horizon_split(dataset: List[Any],
                                        horizon: int = 30) -> Tuple[List[Any],
                                                                    List[Any]]:
        """
        Split dataset with prediction horizon.

        Inputs:
        - dataset: list of game data
        - horizon: int, games to skip

        Returns:
        - Tuple[List, List]: (train_data, test_data)
        """
        sorted_data = sorted(dataset, key=lambda g: g.end_time)

        # train on early games (80%), test on games far in future (20%)
        train_end_idx = int(len(sorted_data) * 0.8)
        test_start_idx = train_end_idx + horizon

        return sorted_data[:train_end_idx], sorted_data[test_start_idx:]

    @staticmethod
    def evaluate_cavity_prevention(dataset: List[Any],
                                   momentum_weights: List[float]
                                   ) -> Dict[str, float]:
        """
        Evaluate how well the system prevents cavities.

        Inputs:
        - dataset: list of game data
        - momentum_weights: list of weights

        Returns:
        - Dict[str, float]: cavity prevention metrics
        """
        momentum_system = NovelMomentumSystem()
        momentum_system.momentum_weights = momentum_weights

        cavity_episodes = 0
        total_cavity_duration = 0
        current_cavity_start = None

        for i, game in enumerate(dataset):
            momentum_system.update_after_game(
                game.username, "opponent", game.actual_result,
                game.to_feature_vector(), momentum_weights
            )

            # check for cavity (performance vs rating mismatch)
            if i >= 20:
                recent_games = dataset[i-20:i]
                recent_win_rate = sum(
                    g.actual_result for g in recent_games) / len(recent_games)

                player = momentum_system.players.get(game.username)
                if player:
                    # expected win rate based on momentum rating
                    expected_win_rate = player.calculate_win_probability(
                        game.opponent_elo)

                    # cavity detection,significant performance gap of 20%
                    if abs(recent_win_rate - expected_win_rate) > 0.2:
                        if current_cavity_start is None:
                            current_cavity_start = i
                    else:
                        if current_cavity_start is not None:
                            cavity_episodes += 1
                            total_cavity_duration += (i - current_cavity_start)
                            current_cavity_start = None

        return {
            'cavity_episodes': cavity_episodes,
            'avg_cavity_duration': (total_cavity_duration /
                                    max(cavity_episodes, 1)),
            'cavity_frequency': (cavity_episodes / len(dataset)
                                 if dataset else 0.0)
        }
