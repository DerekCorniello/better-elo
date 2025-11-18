from dataclasses import dataclass
from typing import List, Dict, Any
import math


@dataclass
class MomentumRating:
    """Momentum-enhanced rating system that prevents rating cavities"""
    base_elo: float
    momentum_adjustment: float = 0.0
    adaptive_K: float = 32.0
    
    @property
    def momentum_elo(self) -> float:
        """Get the momentum-enhanced rating"""
        return self.base_elo + self.momentum_adjustment
    
    def calculate_win_probability(self, opponent_elo: float) -> float:
        """Calculate win probability using momentum-enhanced rating"""
        return 1.0 / (1.0 + 10.0 ** ((opponent_elo - self.momentum_elo) / 400.0))
    
    def calculate_adaptive_K(self, features: List[float], weights: List[float]) -> float:
        """Calculate adaptive K-factor based on momentum indicators"""
        # Base K-factor
        K = 32.0
        
        # Momentum multiplier based on feature weights
        momentum_multiplier = 1.0
        
        # Positive rating trend increases K (faster adjustment for improving players)
        rating_trend_weight = weights[3] if len(weights) > 3 else 0.0
        if rating_trend_weight > 0 and features[3] > 0:  # Positive trend
            momentum_multiplier += abs(features[3] * rating_trend_weight) / 100.0
        
        # High velocity increases K (consistent performance gets faster adjustment)
        velocity_weight = weights[5] if len(weights) > 5 else 0.0
        if velocity_weight > 0 and abs(features[5]) > 1:
            momentum_multiplier += abs(features[5] * velocity_weight) / 50.0
        
        # Win streaks affect K (hot/cold streaks get faster adjustment)
        win_streak_weight = weights[0] if len(weights) > 0 else 0.0
        if abs(features[0]) >= 3:  # Significant streak
            momentum_multiplier += abs(features[0] * win_streak_weight) / 20.0
        
        # Cap K-factor to prevent extreme adjustments
        return min(max(K * momentum_multiplier, 16.0), 64.0)


@dataclass
class RatingCavityMetrics:
    """Metrics to detect and measure rating cavities"""
    cavity_duration: int = 0  # Games spent in cavity
    cavity_depth: float = 0.0  # How far rating is from true performance
    recovery_time: int = 0  # Games to escape cavity
    plateau_episodes: int = 0  # Number of plateau periods
    
    def is_in_cavity(self, performance_gap: float, threshold: float = 50.0) -> bool:
        """Check if player is currently in a rating cavity"""
        return abs(performance_gap) > threshold
    
    def calculate_performance_gap(self, actual_win_rate: float, expected_win_rate: float) -> float:
        """Calculate gap between actual and expected performance"""
        # Convert win rate difference to approximate Elo difference
        if expected_win_rate == 0 or expected_win_rate == 1:
            return 0.0
        elo_diff = -400 * math.log10((1/expected_win_rate - 1) / (1/actual_win_rate - 1))
        return elo_diff


class TemporalValidator:
    """Temporal cross-validation for future prediction evaluation"""
    
    @staticmethod
    def temporal_split(dataset: List[Any], train_ratio: float = 0.7) -> tuple:
        """Split dataset by time (train on past, test on future)"""
        sorted_data = sorted(dataset, key=lambda g: g.end_time)
        split_idx = int(len(sorted_data) * train_ratio)
        return sorted_data[:split_idx], sorted_data[split_idx:]
    
    @staticmethod
    def calculate_future_prediction_accuracy(test_dataset: List[Any], 
                                           momentum_weights: List[float]) -> Dict[str, float]:
        """Evaluate how well momentum system predicts future game outcomes"""
        correct_predictions = 0
        total_games = len(test_dataset)
        brier_score = 0.0
        
        for game in test_dataset:
            # Calculate momentum-enhanced win probability
            momentum_rating = MomentumRating(game.pre_game_elo)
            momentum_rating.momentum_adjustment = sum(
                w * f for w, f in zip(momentum_weights, game.to_feature_vector())
            )
            
            predicted_prob = momentum_rating.calculate_win_probability(game.opponent_elo)
            actual_result = game.actual_result
            
            # Binary prediction (win if prob > 0.5)
            predicted_win = 1 if predicted_prob > 0.5 else 0
            actual_win = 1 if actual_result > 0.5 else 0
            
            if predicted_win == actual_win:
                correct_predictions += 1
            
            # Brier score for probability calibration
            brier_score += (predicted_prob - actual_result) ** 2
        
        accuracy = correct_predictions / total_games if total_games > 0 else 0.0
        brier_score = brier_score / total_games if total_games > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'brier_score': brier_score,
            'total_games': total_games
        }
    
    @staticmethod
    def detect_rating_cavities(dataset: List[Any], momentum_weights: List[float]) -> List[RatingCavityMetrics]:
        """Detect periods where players are stuck in rating cavities"""
        cavities = []
        current_cavity = None
        
        for i, game in enumerate(dataset):
            # Calculate momentum-enhanced rating
            momentum_rating = MomentumRating(game.pre_game_elo)
            momentum_rating.momentum_adjustment = sum(
                w * f for w, f in zip(momentum_weights, game.to_feature_vector())
            )
            
            # Calculate expected vs actual performance
            expected_win_rate = momentum_rating.calculate_win_probability(game.opponent_elo)
            actual_win_rate = game.actual_result
            
            # Use recent performance to estimate true skill level
            recent_games = dataset[max(0, i-10):i]
            if len(recent_games) >= 5:
                recent_actual = sum(g.actual_result for g in recent_games) / len(recent_games)
                performance_gap = abs(recent_actual - expected_win_rate)
                
                if performance_gap > 0.1:  # Significant performance gap
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