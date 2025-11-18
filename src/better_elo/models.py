from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class PlayerFeatures:
    username: str
    current_elo: float
    win_streak: int
    recent_win_rate: float  # 0.0 to 1.0
    avg_accuracy: float  # 0.0 to 100.0
    rating_trend: float
    games_last_30d: int
    velocity: float = 0.0  # Elo change per game over window

    def to_feature_vector(self) -> List[float]:
        return [
            self.win_streak,
            self.recent_win_rate,
            self.avg_accuracy,
            self.rating_trend,
            self.games_last_30d,
            self.velocity
        ]


@dataclass
class UserGameData:
    username: str
    pre_game_elo: float
    post_game_elo: float
    features: PlayerFeatures  # Features computed from history before this game
    velocity: float = 0.0  # Elo change per game over window
    momentum_adjustment: float = 0.0  # Predicted adjustment for true Elo
    end_time: int = 0  # Timestamp of game end for sorting

    def to_feature_vector(self) -> List[float]:
        # Features for adjustment, pre_elo handled separately
        return self.features.to_feature_vector()


@dataclass
class MatchData:
    player1: PlayerFeatures
    player2: PlayerFeatures
    player1_won: bool
    was_draw: bool = False

    @classmethod
    def from_api_response(cls, game_dict: Dict[str, Any], p1_history: List[Dict[str, Any]], p2_history: List[Dict[str, Any]]) -> 'MatchData':
        # Extract basic info
        white_username = game_dict['white']['username']
        black_username = game_dict['black']['username']
        white_rating = game_dict['white']['rating']
        black_rating = game_dict['black']['rating']
        white_result = game_dict['white']['result']
        black_result = game_dict['black']['result']
        accuracies = game_dict.get('accuracies', {})
        white_accuracy = accuracies.get('white', 0.0)
        black_accuracy = accuracies.get('black', 0.0)
        end_time = game_dict['end_time']

        # Determine player1 and player2 (arbitrarily white as p1)
        p1_username = white_username
        p2_username = black_username
        p1_elo = white_rating
        p2_elo = black_rating
        p1_won = white_result == 'win'
        was_draw = white_result == 'draw' or black_result == 'draw'

        # Calculate features from history
        p1_features = cls._calculate_features(p1_username, p1_elo, p1_history, end_time)
        p2_features = cls._calculate_features(p2_username, p2_elo, p2_history, end_time)

        return cls(
            player1=p1_features,
            player2=p2_features,
            player1_won=p1_won,
            was_draw=was_draw
        )

    @classmethod
    def from_mock(cls, p1_elo: float, p2_elo: float, **kwargs) -> 'MatchData':
        # Generate mock features
        import random

        p1_features = PlayerFeatures(
            username=f"player_{random.randint(1000, 9999)}",
            current_elo=p1_elo,
            win_streak=random.randint(-5, 5),
            recent_win_rate=random.uniform(0.3, 0.7),
            avg_accuracy=random.uniform(70.0, 95.0),
            rating_trend=random.uniform(-50, 50),
            games_last_30d=random.randint(10, 50)
        )

        p2_features = PlayerFeatures(
            username=f"player_{random.randint(1000, 9999)}",
            current_elo=p2_elo,
            win_streak=random.randint(-5, 5),
            recent_win_rate=random.uniform(0.3, 0.7),
            avg_accuracy=random.uniform(70.0, 95.0),
            rating_trend=random.uniform(-50, 50),
            games_last_30d=random.randint(10, 50)
        )

        # Determine outcome based on adjusted ratings (simplified)
        adjusted_p1 = p1_elo + sum(p1_features.to_feature_vector()[i] * random.uniform(-1, 1) for i in range(5))
        adjusted_p2 = p2_elo + sum(p2_features.to_feature_vector()[i] * random.uniform(-1, 1) for i in range(5))
        p1_won = adjusted_p1 > adjusted_p2
        if random.random() < 0.2:  # 20% upset probability
            p1_won = not p1_won

        return cls(
            player1=p1_features,
            player2=p2_features,
            player1_won=p1_won,
            was_draw=False
        )

    @staticmethod
    def _did_player_win(username: str, game: Dict[str, Any]) -> bool:
        if game['white']['username'] == username:
            return game['white']['result'] == 'win'
        elif game['black']['username'] == username:
            return game['black']['result'] == 'win'
        return False

    @staticmethod
    def _get_player_accuracy(username: str, game: Dict[str, Any]) -> float:
        accuracies = game.get('accuracies', {})
        if game['white']['username'] == username:
            return accuracies.get('white', 0.0)
        elif game['black']['username'] == username:
            return accuracies.get('black', 0.0)
        return 0.0

    @staticmethod
    def _get_player_rating(username: str, game: Dict[str, Any]) -> float:
        if game['white']['username'] == username:
            return game['white']['rating']
        elif game['black']['username'] == username:
            return game['black']['rating']
        return 1500.0  # default

    @staticmethod
    def _calculate_features(username: str, current_elo: float, history: List[Dict[str, Any]], match_end_time: int) -> PlayerFeatures:
        # Assume history is sorted by end_time ascending (earliest first)
        N = 10  # window size for recent metrics
        window_games = history[-N:] if len(history) >= N else history

        # Win streak
        win_streak = 0
        for game in reversed(history):  # most recent first
            won = MatchData._did_player_win(username, game)
            if win_streak == 0:
                win_streak = 1 if won else -1
            elif (win_streak > 0 and won) or (win_streak < 0 and not won):
                win_streak += 1 if won else -1
            else:
                break

        # Recent win rate
        if window_games:
            wins = sum(1 for game in window_games if MatchData._did_player_win(username, game))
            recent_win_rate = wins / len(window_games)
        else:
            recent_win_rate = 0.5

        # Average accuracy
        accuracies = []
        for game in window_games:
            acc = MatchData._get_player_accuracy(username, game)
            if acc > 0:
                accuracies.append(acc)
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 80.0

        # Rating trend
        if window_games:
            past_rating = MatchData._get_player_rating(username, window_games[0])
            rating_trend = current_elo - past_rating
        else:
            rating_trend = 0.0

        # Games last 30 days
        thirty_days_ago = match_end_time - 30 * 24 * 3600
        games_last_30d = sum(1 for game in history if game['end_time'] >= thirty_days_ago)

        return PlayerFeatures(
            username=username,
            current_elo=current_elo,
            win_streak=win_streak,
            recent_win_rate=recent_win_rate,
            avg_accuracy=avg_accuracy,
            rating_trend=rating_trend,
            games_last_30d=games_last_30d
        )