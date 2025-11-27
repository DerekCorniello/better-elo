import json
import os
from typing import List
import sys
from models import UserGameData, PlayerFeatures, MatchData

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class RealDataGenerator:
    """
    Generates training datasets from real Chess.com game data.

    Attributes:
    - username: str, player username to generate data for
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

    def generate_dataset(self, velocity_window: int = 10
                         ) -> List[UserGameData]:
        """
        Generate dataset of UserGameData from player's games.

        Inputs:
        - velocity_window: int, games to look back for velocity calculation

        Outputs:
        - List[UserGameData]: processed game data for training

        Expected behavior:
        Loads JSON games, filters to blitz, computes features
            and Elo adjustments.
        Returns list of game data objects.
        """
        # Load games from JSON data we extracted
        filepath = f'data/{self.username}/games.json'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Games file not found: {filepath}")

        with open(filepath, 'r') as f:
            games = json.load(f)

        # filter to blitz time controls (180, 180+1, 180+2) for consistency
        # NOTE: you can filter anything if you want, but blitz is the most
        # common with the largest amount of data for the users we have
        games = [g for g in games if g.get(
            'time_control', '').startswith('180')]
        print(f"Filtered to {len(games)} blitz games for {self.username}")
        games.sort(key=lambda g: g['end_time'])

        user_games = []
        history = []

        for i, game in enumerate(games):
            # determine if user is white or black
            # TODO: do we need the user result anymore?
            if game['white']['username'].lower() == self.username.lower():
                post_rating = game['white']['rating']
                user_result = game['white']['result']
            elif game['black']['username'].lower() == self.username.lower():
                post_rating = game['black']['rating']
                user_result = game['black']['result']
            else:
                # shouldnt hit this but...
                print("Skipping a game without the user in it \
                        (investigate the data)")
                continue

            # compute velocity change over last N games
            if i >= velocity_window:
                pre_window_game = games[i - velocity_window]
                if pre_window_game['white']['username'] == self.username:
                    pre_rating = pre_window_game['white']['rating']
                else:
                    pre_rating = pre_window_game['black']['rating']
                velocity = (post_rating - pre_rating) / velocity_window
            else:
                # not enough history for velocity window
                continue

            # compute features from history before this game
            features = self._calculate_features(
                history, post_rating, game['end_time'])
            features.velocity = velocity

            if game['white']['username'].lower() == self.username.lower():
                opponent_elo = game['black']['rating']
            else:
                opponent_elo = game['white']['rating']

            if user_result == 'win':
                actual_result = 1.0  # player won
            elif user_result in ['resigned', 'checkmated',
                                 'timeout', 'abandoned']:
                actual_result = 0.0  # player lost
            else:
                actual_result = 0.5  # draw or other result

            # adjust opponent Elo to approximate pre-game
            # rating using Elo formula, since the data
            # we get is post game only...
            if actual_result != 0.5:  # only adjust for decisive games
                expected = 1 / (1 + 10 ** ((opponent_elo - pre_rating) / 400))
                actual = actual_result
                K = 20  # Standard K-factor
                delta = K * (expected - actual)
                opponent_elo -= delta  # get pre-game Elo

            user_game = UserGameData(
                username=self.username,
                pre_game_elo=pre_rating,
                post_game_elo=post_rating,
                features=features,
                end_time=game['end_time'],  # for sorting
                velocity=velocity,
                opponent_elo=opponent_elo,
                actual_result=actual_result
            )

            user_games.append(user_game)
            history.append(game)
        return user_games

    def _calculate_features(self, history: List[dict], current_elo: float,
                            match_end_time: int) -> PlayerFeatures:
        """
        Wrapper to calculate player features from game history.

        Inputs:
        - history: list of game dicts
        - current_elo: float, current player rating
        - match_end_time: int, timestamp of current match

        Outputs:
        - PlayerFeatures: computed features

        Expected behavior:
        Delegates to MatchData._calculate_features for feature computation.
        """
        return MatchData._calculate_features(self.username, current_elo,
                                             history, match_end_time)
