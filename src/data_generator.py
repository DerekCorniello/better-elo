import json
import os
import sys
from typing import List

from .config import config
from .logging_config import get_logger
from .models import MatchData, PlayerFeatures, UserGameData

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logger = get_logger(__name__)


class RealDataGenerator:
    """Generates training datasets from real Chess.com game data.

    Attributes:
        username: Player username to generate data for.
    """

    def __init__(self, username: str):
        """
        Initialize data generator for a player.

        Inputs:
        - username: str, Chess.com username

        Outputs:
        - None

        Expected behavior:
        Sets up generator for the specified player.
        """
        self.username = username

    def generate_dataset(
        self, velocity_window: int | None = None
    ) -> List[UserGameData]:
        """Generate dataset of UserGameData from player's games.

        Args:
            velocity_window: Games to look back for velocity calculation.

        Returns:
            List[UserGameData]: Processed game data for training.

        Expected behavior:
            Loads JSON games, filters to blitz, computes features
            and Elo adjustments. Returns list of game data objects.
        """
        if velocity_window is None:
            velocity_window = config.model["velocity_window"]
        # Load games from JSON data we extracted
        filepath = f"data/{self.username}/games.json"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Games file not found: {filepath}")

        try:
            with open(filepath, "r") as f:
                games = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load games from %s: %s", filepath, e)
            raise

        # Filter to blitz time controls for consistency
        time_control = config.data["time_control"]
        games = [
            g
            for g in games
            if g.get("time_control", "").startswith(time_control)
        ]
        logger.info(
            "Filtered to %d blitz games for %s", len(games), self.username
        )
        games.sort(key=lambda g: g["end_time"])

        user_games = []
        history = []

        for i, game in enumerate(games):
            # Determine if user is white or black
            if game["white"]["username"].lower() == self.username.lower():
                post_rating = game["white"]["rating"]
                user_result = game["white"]["result"]
            elif game["black"]["username"].lower() == self.username.lower():
                post_rating = game["black"]["rating"]
                user_result = game["black"]["result"]
            else:
                # Skip game without user in it
                logger.warning("Skipping a game without the user in it")
                continue

            # compute velocity change over last N games
            if i >= velocity_window:
                pre_window_game = games[i - velocity_window]
                if pre_window_game["white"]["username"] == self.username:
                    pre_rating = pre_window_game["white"]["rating"]
                else:
                    pre_rating = pre_window_game["black"]["rating"]
                velocity = (post_rating - pre_rating) / velocity_window
            else:
                # not enough history for velocity window
                continue

            # compute features from history before this game
            features = self._calculate_features(
                history, post_rating, game["end_time"]
            )
            features.velocity = velocity

            if game["white"]["username"].lower() == self.username.lower():
                opponent_elo = game["black"]["rating"]
            else:
                opponent_elo = game["white"]["rating"]

            if user_result == "win":
                actual_result = 1.0  # player won
            elif user_result in [
                "resigned",
                "checkmated",
                "timeout",
                "abandoned",
            ]:
                actual_result = 0.0  # player lost
            else:
                actual_result = 0.5  # draw or other result

            # adjust opponent Elo to approximate pre-game
            # rating using Elo formula, since the data
            # we get is post game only...
            if actual_result != 0.5:  # only adjust for decisive games
                expected = 1 / (1 + 10 ** ((opponent_elo - pre_rating) / 400))
                actual = actual_result
                K = config.model["k_factor"]
                delta = K * (expected - actual)
                opponent_elo -= delta  # get pre-game Elo

            user_game = UserGameData(
                username=self.username,
                pre_game_elo=pre_rating,
                post_game_elo=post_rating,
                features=features,
                end_time=game["end_time"],  # for sorting
                velocity=velocity,
                opponent_elo=opponent_elo,
                actual_result=actual_result,
            )

            user_games.append(user_game)
            history.append(game)
        return user_games

    def _calculate_features(
        self, history: List[dict], current_elo: float, match_end_time: int
    ) -> PlayerFeatures:
        """Wrapper to calculate player features from game history.

        Args:
            history: List of game dictionaries.
            current_elo: Current player rating.
            match_end_time: Timestamp of current match.

        Returns:
            PlayerFeatures: Computed features.

        Expected behavior:
            Delegates to MatchData._calculate_features for feature computation.
        """
        return MatchData._calculate_features(
            self.username, current_elo, history, match_end_time
        )
