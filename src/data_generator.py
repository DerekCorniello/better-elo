import json
import os
import numpy as np
from typing import List
from .models import UserGameData, PlayerFeatures, MatchData


class RealDataGenerator:
    def __init__(self, username: str):
        self.username = username

    def generate_dataset(self, velocity_window: int = 10) -> List[UserGameData]:
        # Load games from JSON
        filepath = f'data/{self.username}/games.json'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Games file not found: {filepath}")

        with open(filepath, 'r') as f:
            games = json.load(f)

        # Filter to blitz time controls (180, 180+1, 180+2, etc.) for consistency
        games = [g for g in games if g.get('time_control', '').startswith('180')]
        print(f"Filtered to {len(games)} blitz games for {self.username}")
        # Debug: print unique time controls
        unique_controls = set(g.get('time_control', '') for g in games)
        print(f"Unique time controls: {unique_controls}")

        # Sort games by end_time ascending
        games.sort(key=lambda g: g['end_time'])

        user_games = []
        history = []

        for i, game in enumerate(games):
            # Debug: print first few games
            if i < 5:
                print(f"Game {i}: white={game['white']['username']}, black={game['black']['username']}, target={self.username}")
            # Determine if user is white or black (case-insensitive)
            if game['white']['username'].lower() == self.username.lower():
                post_rating = game['white']['rating']
                user_result = game['white']['result']
            elif game['black']['username'].lower() == self.username.lower():
                post_rating = game['black']['rating']
                user_result = game['black']['result']
            else:
                continue  # Skip if not the user

            # Compute velocity: Elo change over last N games
            if i >= velocity_window:
                pre_window_game = games[i - velocity_window]
                if pre_window_game['white']['username'] == self.username:
                    pre_rating = pre_window_game['white']['rating']
                else:
                    pre_rating = pre_window_game['black']['rating']
                velocity = (post_rating - pre_rating) / velocity_window
            else:
                # Not enough history for velocity window
                print(f"Skipping game {i} for {self.username}: i={i}, velocity_window={velocity_window}")
                continue

            # Compute features from history before this game
            features = self._calculate_features(history, post_rating, game['end_time'])
            features.velocity = velocity  # Set velocity in features

            user_game = UserGameData(
                username=self.username,
                pre_game_elo=pre_rating,  # Elo before the current game
                post_game_elo=post_rating,  # Elo after the current game
                features=features,
                end_time=game['end_time']  # Timestamp for sorting
            )
            # Store velocity as a separate attribute for training
            user_game.velocity = velocity
            user_games.append(user_game)

            # Update history
            history.append(game)

        # Normalize features across the dataset
        if user_games:
            feature_vectors = [game.to_feature_vector() for game in user_games]
            feature_array = np.array(feature_vectors)
            means = np.mean(feature_array, axis=0)
            stds = np.std(feature_array, axis=0)
            stds = np.where(stds == 0, 1, stds)  # Avoid division by zero
            normalized_features = (feature_array - means) / stds

            for i, game in enumerate(user_games):
                game.features = PlayerFeatures(
                    username=game.username,
                    current_elo=game.features.current_elo,  # Keep original Elo
                    win_streak=normalized_features[i][0],
                    recent_win_rate=normalized_features[i][1],
                    avg_accuracy=normalized_features[i][2],
                    rating_trend=normalized_features[i][3],
                    games_last_30d=normalized_features[i][4],
                    velocity=normalized_features[i][5]
                )

        return user_games

    def _calculate_features(self, history: List[dict], current_elo: float, match_end_time: int) -> PlayerFeatures:
        # Use MatchData's static method
        return MatchData._calculate_features(self.username, current_elo, history, match_end_time)