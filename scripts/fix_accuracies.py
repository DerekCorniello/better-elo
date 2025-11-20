#!/usr/bin/env python3

import json
import os
import subprocess
import tempfile

def fix_accuracies():
    data_dir = 'data'
    script_path = 'scripts/analyze_game.py'

    print("Starting fix_accuracies")

    # Only process MagnusCarlsen for now
    user_dir = 'MagnusCarlsen'
    user_path = os.path.join(data_dir, user_dir)
    print(f"Checking {user_path}")
    if not os.path.isdir(user_path):
        print("Not a dir")
        return

    username = user_dir  # The player whose data this is
    games_file = os.path.join(user_path, 'games.json')
    print(f"Checking {games_file}")
    if not os.path.exists(games_file):
        print("File not exists")
        return

    print(f"Processing {games_file} for {username}")

    with open(games_file, 'r') as f:
        games = json.load(f)

    updated = False
    processed_count = 0

    for game in games:
            accuracies = game.get('accuracies', {})
            if accuracies and 'white' in accuracies and 'black' in accuracies:
                continue  # Both accuracies available, skip
            computed_acc = game.get('computed_accuracy')
            if computed_acc is not None and (computed_acc == 0.0 or computed_acc == 100.0):

                pgn = game['pgn']
                if username.lower() == game['white']['username'].lower():
                    opponent = game['black']['username']
                else:
                    opponent = game['white']['username']
                print(f"Fixing accuracy for {username.lower()} vs {opponent.lower()} in game {game['url']}")

                # Write PGN to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as temp_pgn:
                    temp_pgn.write(pgn)
                    temp_pgn_path = temp_pgn.name

                try:
                    # Run analyze_game.py
                    result = subprocess.run(
                        ['python', script_path, temp_pgn_path, username],
                        capture_output=True, text=True, check=True
                    )
                    # Parse output to get accuracy
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines:
                        if line.startswith('Accuracy for'):
                            accuracy_str = line.split(':')[1].strip().split('%')[0]
                            new_accuracy = float(accuracy_str)
                            game['computed_accuracy'] = new_accuracy
                            updated = True
                            processed_count += 1
                            print(f"Updated accuracy to {new_accuracy}%")
                            break
                except subprocess.CalledProcessError as e:
                    print(f"Error running analyze_game: {e}")
                finally:
                    os.unlink(temp_pgn_path)

    if updated:
            with open(games_file, 'w') as f:
                json.dump(games, f, indent=2)
            print(f"Updated {processed_count} games in {games_file}")

if __name__ == "__main__":
    fix_accuracies()