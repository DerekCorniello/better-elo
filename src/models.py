from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class PlayerFeatures:
    """
    Represents player features for momentum analysis.

    Attributes:
    - username: str, player username
    - current_elo: float, current Elo rating
    - win_streak: int, current win/loss streak
    - recent_win_rate: float, win rate in recent games (0.0 to 1.0)
    - avg_accuracy: float, average move accuracy (0.0 to 100.0)
    - rating_trend: float, rating change over recent games
    - games_last_30d: int, games played in last 30 days
    - velocity: float, Elo change per game over window
    """
    username: str
    current_elo: float
    win_streak: int
    recent_win_rate: float   # 0.0 to 1.0
    avg_accuracy: float      # 0.0 to 100.0
    rating_trend: float
    games_last_30d: int
    velocity: float = 0.0    # Elo change per game over window

    def to_feature_vector(self) -> List[float]:
        """
        Convert features to numerical vector for ML.

        Returns:
        - List[float]: 6-element feature vector
        """
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
    """
    Represents a single game with pre/post Elo and features.

    Attributes:
    - username: str, player username
    - pre_game_elo: float, Elo before game
    - post_game_elo: float, Elo after game
    - features: PlayerFeatures, momentum features
    - velocity: float, Elo change per game
    - momentum_adjustment: float, predicted adjustment
    - opponent_elo: float, opponent's rating
    - actual_result: float, game outcome (1.0 win, 0.5 draw, 0.0 loss)
    - end_time: int, game end timestamp
    """
    username: str
    pre_game_elo: float
    post_game_elo: float
    features: PlayerFeatures  # features computed from history before this game
    velocity: float = 0.0  # elo change per game over window
    momentum_adjustment: float = 0.0  # predicted adjustment for true Elo
    opponent_elo: float = 0.0  # opponent's rating for win probability calc
    actual_result: float = 0.0  # 1.0 for win, 0.5 for draw, 0.0 for loss
    end_time: int = 0  # timestamp of game end for sorting

    def to_feature_vector(self) -> List[float]:
        """
        Get feature vector from PlayerFeatures.

        Returns:
        - List[float]: feature vector
        """
        # features for adjustment, pre_elo handled separately
        return self.features.to_feature_vector()


@dataclass
class MatchData:
    """
    Represents a match between two players.

    Attributes:
    - player1: PlayerFeatures, first player features
    - player2: PlayerFeatures, second player features
    - player1_won: bool, whether player1 won
    - was_draw: bool, whether game was draw
    """
    player1: PlayerFeatures
    player2: PlayerFeatures
    player1_won: bool
    was_draw: bool = False

    @classmethod
    def from_api_response(cls, game_dict: Dict[str, Any],
                          p1_history: List[Dict[str, Any]],
                          p2_history: List[Dict[str, Any]]) -> 'MatchData':
        """
        Create MatchData from Chess.com API response.

        Inputs:
        - game_dict: dict, game data from API
        - p1_history: list, player1 game history
        - p2_history: list, player2 game history

        Returns:
        - MatchData: constructed match data
        """
        white_username = game_dict['white']['username']
        black_username = game_dict['black']['username']
        white_rating = game_dict['white']['rating']
        black_rating = game_dict['black']['rating']
        white_result = game_dict['white']['result']
        black_result = game_dict['black']['result']
        end_time = game_dict['end_time']

        # determine player1 and player2 (arbitrarily white as p1)
        p1_username = white_username
        p2_username = black_username
        p1_elo = white_rating
        p2_elo = black_rating
        p1_won = white_result == 'win'
        was_draw = white_result == 'draw' or black_result == 'draw'

        # calculate features from history
        p1_features = cls._calculate_features(p1_username, p1_elo,
                                              p1_history, end_time)
        p2_features = cls._calculate_features(p2_username, p2_elo,
                                              p2_history, end_time)

        return cls(
            player1=p1_features,
            player2=p2_features,
            player1_won=p1_won,
            was_draw=was_draw
        )

    @staticmethod
    def _did_player_win(username: str, game: Dict[str, Any]) -> bool:
        """
        Check if player won the game.

        Inputs:
        - username: str, player username
        - game: dict, game data

        Returns:
        - bool: True if player won
        """
        if game['white']['username'] == username:
            return game['white']['result'] == 'win'
        elif game['black']['username'] == username:
            return game['black']['result'] == 'win'
        return False

    @staticmethod
    def _get_player_accuracy(username: str, game: Dict[str, Any]) -> float:
        """
        Get player's accuracy from game.

        Inputs:
        - username: str, player username
        - game: dict, game data

        Returns:
        - float: accuracy percentage
        """
        accuracies = game.get('accuracies', {})
        if game['white']['username'] == username:
            return accuracies.get('white', 0.0)
        elif game['black']['username'] == username:
            return accuracies.get('black', 0.0)
        return 0.0

    @staticmethod
    def _get_player_rating(username: str, game: Dict[str, Any]) -> float:
        """
        Get player's rating from game.

        Inputs:
        - username: str, player username
        - game: dict, game data

        Returns:
        - float: player rating
        """
        if game['white']['username'] == username:
            return game['white']['rating']
        elif game['black']['username'] == username:
            return game['black']['rating']
        return 1500.0  # default, shouldnt hit this tho

    @staticmethod
    def _calculate_features(username: str, current_elo: float,
                            history: List[Dict[str, Any]],
                            match_end_time: int) -> PlayerFeatures:
        """
        Calculate player features from game history.

        Inputs:
        - username: str, player username
        - current_elo: float, current rating
        - history: list, game history
        - match_end_time: int, match timestamp

        Returns:
        - PlayerFeatures: calculated features
        """
        N = 10  # window size for recent metrics
        window_games = history[-N:] if len(history) >= N else history

        win_streak = 0
        for game in reversed(history):  # most recent first
            won = MatchData._did_player_win(username, game)
            if win_streak == 0:
                win_streak = 1 if won else -1
            elif (win_streak > 0 and won) or (win_streak < 0 and not won):
                win_streak += 1 if won else -1
            else:
                break

        if window_games:
            wins = sum(1 for game in window_games
                       if MatchData._did_player_win(username, game))
            recent_win_rate = wins / len(window_games)
        else:
            recent_win_rate = 0.5

        accuracies = []
        for game in window_games:
            acc = MatchData._get_player_accuracy(username, game)
            if acc > 0:
                accuracies.append(acc)
        avg_accuracy = sum(accuracies) / \
            len(accuracies) if accuracies else 80.0

        if window_games:
            past_rating = MatchData._get_player_rating(
                username, window_games[0])
            rating_trend = current_elo - past_rating
        else:
            rating_trend = 0.0

        thirty_days_ago = match_end_time - 30 * 24 * 3600
        games_last_30d = sum(
            1 for game in history if game['end_time'] >= thirty_days_ago)

        return PlayerFeatures(
            username=username,
            current_elo=current_elo,
            win_streak=win_streak,
            recent_win_rate=recent_win_rate,
            avg_accuracy=avg_accuracy,
            rating_trend=rating_trend,
            games_last_30d=games_last_30d
        )
