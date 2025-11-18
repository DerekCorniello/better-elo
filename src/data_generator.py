import json
import os
import numpy as np
from typing import List
from .models import UserGameData, PlayerFeatures, MatchData


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

            user_game = UserGameData(
                username=self.username,
                pre_game_elo=pre_rating,
                post_game_elo=post_rating,
                features=features,
                end_time=game['end_time'],  # for sorting
                velocity=velocity
            )

            user_games.append(user_game)
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

    # TODO: this is messy, lets fix
    def _calculate_features(self, history: List[dict], current_elo: float, match_end_time: int) -> PlayerFeatures:
        return MatchData._calculate_features(self.username, current_elo, history, match_end_time)
