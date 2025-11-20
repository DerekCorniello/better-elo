import json
import os
import numpy as np
from typing import List
import sys
import os
# Add current directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import UserGameData, PlayerFeatures, MatchData


class RealDataGenerator:
    def __init__(self, username: str):
        self.username = username

    def generate_dataset(self, velocity_window: int = 10) -> List[UserGameData]:
        # Load games from JSON data we extracted
        filepath = f'data/{self.username}/games.json'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Games file not found: {filepath}")

        with open(filepath, 'r') as f:
            games = json.load(f)

        # filter to blitz time controls (180, 180+1, 180+2) for consistency
        # NOTE: you can filter anything if you want, but blitz is the most
        # common with the largest amount of data
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
                print("Skipping a game without the user in it (investigate the data)")
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

            # Extract opponent Elo
            if game['white']['username'].lower() == self.username.lower():
                opponent_elo = game['black']['rating']
            else:
                opponent_elo = game['white']['rating']

            # Determine actual result correctly
            if user_result == 'win':
                actual_result = 1.0  # Player won
            elif user_result in ['resigned', 'checkmated', 'timeout', 'abandoned']:
                actual_result = 0.0  # Player lost
            else:
                actual_result = 0.5  # Draw or other result

            # Adjust opponent Elo to approximate pre-game rating
            if actual_result == 1.0:
                opponent_elo -= 7  # Opponent was weaker, pre-game Elo lower
            elif actual_result == 0.0:
                opponent_elo += 7  # Opponent was stronger, pre-game Elo higher
            # For draws, no adjustment

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

        # Keep original features - normalization destroys momentum patterns
        # The evolutionary algorithm needs the raw, meaningful relationships

        return user_games

    # TODO: this is messy, lets fix
    def _calculate_features(self, history: List[dict], current_elo: float, match_end_time: int) -> PlayerFeatures:
        return MatchData._calculate_features(self.username, current_elo, history, match_end_time)
