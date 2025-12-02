import json
import os
import sys
import time
from datetime import datetime

import requests

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from analyze_game import analyze_game
    from config import config
    from logging_config import get_logger
except ImportError:
    # Fallback for when running as script
    from analyze_game import analyze_game
    from config import config
    from logging_config import get_logger

logger = get_logger(__name__)


def get_user_games(username, months=None):
    """Fetch and process the last N months of blitz games for a user.

    Args:
        username: Chess.com username to fetch games for.
        months: Number of months to look back (uses config if None).

    Returns:
        list: Processed game data with accuracies.
    """
    if months is None:
        months = config.api["months"]
    base_url = f"https://api.chess.com/pub/player/{username}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Get archives
    archives_url = f"{base_url}/games/archives"
    try:
        response = requests.get(archives_url, headers=headers, timeout=30)
        if response.status_code != 200:
            logger.error(
                "Failed to get archives: %d:\n%s",
                response.status_code,
                response.text,
            )
            return []

        archives = response.json().get("archives", [])
    except requests.RequestException as e:
        logger.error("Failed to fetch archives: %s", e)
        return []

    # Filter to last months
    now = datetime.now()
    recent_archives = []
    for archive in archives:
        parts = archive.split("/")
        year = int(parts[-2])
        month = int(parts[-1])
        months_diff = (now.year - year) * 12 + now.month - month
        if months_diff < months:
            recent_archives.append(archive)

    games = []
    for archive in recent_archives:
        logger.info("Fetching %s", archive)
        try:
            response = requests.get(archive, headers=headers, timeout=30)
            if response.status_code != 200:
                logger.error(
                    "Failed to get games from %s: %d", archive, response.status_code
                )
                continue

            data = response.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error("Failed to process archive %s: %s", archive, e)
            continue
        for game in data.get("games", []):
            if game.get("time_class") != "blitz":
                continue

            pgn = game["pgn"]
            accuracies = game.get("accuracies", {})

            white_username = game["white"]["username"]
            black_username = game["black"]["username"]

            user_accuracies = None
            if username == white_username:
                user_accuracies = accuracies.get("white")
            elif username == black_username:
                user_accuracies = accuracies.get("black")

            computed_accuracy = None
            if user_accuracies is None:
                logger.info("Computing accuracy for game %s", game["url"])
                result = analyze_game(pgn, username)
                if "error" in result:
                    logger.error("Error: %s", result["error"])
                    # Skip this game if analysis fails
                    continue
                computed_accuracy = result["accuracy"]

            game_data = {
                "url": game["url"],
                "end_time": game["end_time"],
                "time_control": game.get("time_control", ""),
                "rules": game.get("rules", "chess"),
                "eco": game.get("eco", ""),
                "fen": game.get("fen", ""),
                "white": game["white"],
                "black": game["black"],
                "result": f"{game['white']['result']}-{game['black']['result']}",
                "accuracies": accuracies,
                "computed_accuracy": computed_accuracy,
                "pgn": pgn,
            }
            games.append(game_data)

        time.sleep(1)  # Rate limit

    # Save to file
    try:
        os.makedirs(f"data/{username}", exist_ok=True)
        with open(f"data/{username}/games.json", "w") as f:
            json.dump(games, f, indent=2)
    except (OSError, IOError) as e:
        logger.error("Failed to save games to file: %s", e)
        return []

    logger.info("Saved %d games for %s", len(games), username)
    return games


if __name__ == "__main__":
    # List of usernames to fetch data for
    usernames = ["MagnusCarlsen"]

    for username in usernames:
        logger.info("Fetching data for %s...", username)
        try:
            get_user_games(username)
        except Exception as e:
            logger.error("Error fetching for %s: %s", username, e)
        time.sleep(5)  # Delay between users to avoid rate limits
