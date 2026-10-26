import argparse
import csv

import chess.pgn
from tqdm import tqdm


def extract_examples(game):
    board = game.board()
    for move in game.mainline_moves():
        yield board.fen(), move.uci()
        board.push(move)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn")
    parser.add_argument("--out", default="data/positions.csv")
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    games = 0
    examples = 0
    with open(args.pgn, encoding="utf-8") as pgn, open(args.out, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["fen", "move", "game"])
        progress = tqdm(total=args.max_games, unit="games")
        while args.max_games is None or games < args.max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            for fen, move in extract_examples(game):
                writer.writerow([fen, move, games])
                examples += 1
            games += 1
            progress.update(1)
        progress.close()

    print(f"{games} games, {examples} examples -> {args.out}")


if __name__ == "__main__":
    main()
