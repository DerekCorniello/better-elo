import requests
import json
import os
import time
from datetime import datetime
from analyze_game import analyze_game


def get_user_games(username, months=3):
    """Fetch and process the last N months of blitz games for a user."""
    base_url = f"https://api.chess.com/pub/player/{username}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Get archives
    archives_url = f"{base_url}/games/archives"
    response = requests.get(archives_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to get archives: {
              response.status_code}:\n{response.text}")
        return []

    archives = response.json().get('archives', [])

    # Filter to last months
    now = datetime.now()
    recent_archives = []
    for archive in archives:
        parts = archive.split('/')
        year = int(parts[-2])
        month = int(parts[-1])
        months_diff = (now.year - year) * 12 + now.month - month
        if months_diff < months:
            recent_archives.append(archive)

    games = []
    for archive in recent_archives:
        print(f"Fetching {archive}")
        response = requests.get(archive, headers=headers)
        if response.status_code != 200:
            print(f"Failed to get games from {
                  archive}: {response.status_code}")
            continue

        data = response.json()
        for game in data.get('games', []):
            if game.get('time_class') != 'blitz':
                continue

            pgn = game['pgn']
            accuracies = game.get('accuracies', {})

            white_username = game['white']['username']
            black_username = game['black']['username']

            user_accuracies = None
            if username == white_username:
                user_accuracies = accuracies.get('white')
            elif username == black_username:
                user_accuracies = accuracies.get('black')

            computed_accuracy = None
            if user_accuracies is None:
                print(f"Computing accuracy for game {game['url']}")
                result = analyze_game(pgn, username)
                if 'error' in result:
                    print(f"Error: {result['error']}")
                    # Skip this game if analysis fails
                    continue
                computed_accuracy = result['accuracy']

            game_data = {
                'url': game['url'],
                'end_time': game['end_time'],
                'time_control': game.get('time_control', ''),
                'rules': game.get('rules', 'chess'),
                'eco': game.get('eco', ''),
                'fen': game.get('fen', ''),
                'white': game['white'],
                'black': game['black'],
                'result': f"{game['white']['result']}-{game['black']['result']}",
                'accuracies': accuracies,
                'computed_accuracy': computed_accuracy,
                'pgn': pgn
            }
            games.append(game_data)

        time.sleep(1)  # Rate limit

    # Save to file
    os.makedirs(f'data/{username}', exist_ok=True)
    with open(f'data/{username}/games.json', 'w') as f:
        json.dump(games, f, indent=2)

    print(f"Saved {len(games)} games for {username}")
    return games


if __name__ == "__main__":
    # List of usernames to fetch data for
    usernames = [
        "hikaru",
        "GothamChess",
        "MagnusCarlsen",
        "FabianoCaruana",
        "WesleySo",
        "AnishGiri",
        "AnnaCramling"
    ]
    months = 24  # Fetch last 24 months for more data

    for username in usernames:
        print(f"\nFetching data for {username}...")
        try:
            get_user_games(username, months)
        except Exception as e:
            print(f"Error fetching for {username}: {e}")
        time.sleep(5)  # Delay between users to avoid rate limits
