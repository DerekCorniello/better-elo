#!/usr/bin/env python3

import io
import os
import sys

import chess
import chess.pgn
from stockfish import Stockfish

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from config import config
    from logging_config import get_logger
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from config import config
    from logging_config import get_logger

logger = get_logger(__name__)


def normalize_eval(eval_obj, side_to_move_is_white):
    """
    Convert Stockfish evaluation to a centipawn number from the POV
    of the player who is about to move in the *original* position.
    We also map mate scores to large cp values so they still produce
    sensible "loss" numbers.
    """
    if eval_obj["type"] == "cp":
        return eval_obj["value"]

    # Mate scores: positive = winning, negative = losing
    # Map mate in N to +/- 100000 - N so closer mates are slightly bigger.
    if eval_obj["type"] == "mate":
        mate_in = eval_obj["value"]
        if mate_in is None:
            return 0
        base = 100000
        if mate_in > 0:
            return base - mate_in
        else:
            return -base - mate_in

    return 0


def analyze_game(pgn_text, username=None):
    """
    Analyze a PGN game with Stockfish and compute accuracies for specified user
    """
    if username is None:
        return {"error": "Username required"}

    # Initialize Stockfish with configuration parameters
    stockfish = Stockfish(
        path=config.analysis["stockfish_path"],
        parameters={
            "Threads": 12,
            "Hash": 12288,  # 12GB RAM
            "Contempt": 0,
        },
    )
    stockfish.set_depth(config.analysis["depth"])

    pgn = chess.pgn.read_game(io.StringIO(pgn_text))
    if pgn is None:
        return {"error": "Invalid PGN"}

    white_player = pgn.headers.get("White")
    black_player = pgn.headers.get("Black")

    if white_player and username.lower() == white_player.lower().strip():
        user_color = chess.WHITE
    elif black_player and username.lower() == black_player.lower().strip():
        user_color = chess.BLACK
    else:
        return {"error": "Username not found in game"}

    board = pgn.board()
    losses = []
    move_count = 0

    for move in pgn.mainline_moves():
        player_color = board.turn
        # only analyze moves for the target user
        if player_color == user_color:
            # evaluate best move line
            stockfish.set_fen_position(board.fen())
            best_move = stockfish.get_best_move()

            if best_move:
                # eval after best move
                stockfish.make_moves_from_current_position([best_move])
                best_after = stockfish.get_evaluation()

                # evaluate the move taken
                board.push(move)
                stockfish.set_fen_position(board.fen())
                your_after = stockfish.get_evaluation()

                # the best measure of accuracy is centipawn loss:
                # https://www.chess.com/blog/raync910/average-centipawn-loss-chess-acpl
                best_cp = normalize_eval(
                    best_after,
                    side_to_move_is_white=(user_color == chess.WHITE),
                )
                your_cp = normalize_eval(
                    your_after,
                    side_to_move_is_white=(user_color == chess.WHITE),
                )

                # from white's pov, higher cp is better
                # from black's pov, we flip the sign to keep things consistent
                if user_color == chess.WHITE:
                    loss = best_cp - your_cp
                else:
                    loss = your_cp - best_cp

                losses.append(loss)
            else:
                board.push(move)
        else:
            board.push(move)

    # Compute accuracies: % of moves with |loss| < threshold cp
    # This is a standard measurement for accurate moves
    threshold = config.analysis["accuracy_threshold"]
    accurate_moves = sum(1 for loss in losses if abs(loss) < threshold)
    accuracy = (accurate_moves / len(losses)) * 100.0 if losses else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "num_moves": len(losses),
    }


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            pgn_text = f.read()
            username = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        pgn_text = sys.stdin.read()
        username = sys.argv[2] if len(sys.argv) > 2 else None

    result = analyze_game(pgn_text, username)
    if "error" in result:
        logger.error(result["error"])
    else:
        logger.info("Accuracy for %s: %.1f%%", username, result["accuracy"])
        logger.info("Average Centipawn Loss: %.1f cp", result["avg_loss"])
        logger.info("Moves analyzed: %d", result["num_moves"])


if __name__ == "__main__":
    main()
