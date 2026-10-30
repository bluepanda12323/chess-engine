import argparse
import csv
import io
import json
import urllib.request

import zstandard
from tqdm import tqdm

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def best_score(evals):
    deepest = max(evals, key=lambda e: e["depth"])
    pv = deepest["pvs"][0]
    return pv.get("cp"), pv.get("mate"), pv["line"].split()[0]


def square_index(square):
    return (8 - int(square[1])) * 8 + (ord(square[0]) - ord("a"))


def is_quiet(fen, move):
    if len(move) == 5:
        return False

    squares = []
    for row in fen.split(" ")[0].split("/"):
        for char in row:
            if char.isdigit():
                squares.extend("." * int(char))
            else:
                squares.append(char)

    from_square, to_square = square_index(move[:2]), square_index(move[2:4])
    if squares[to_square] != ".":
        return False
    if squares[from_square] in "Pp" and move[0] != move[2]:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/evals_quiet.csv")
    parser.add_argument("--max-positions", type=int, default=30000000)
    parser.add_argument("--all-positions", action="store_true")
    args = parser.parse_args()

    positions, skipped = 0, 0
    with urllib.request.urlopen(URL) as response, open(args.out, "w", newline="", encoding="utf-8") as out:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(response), encoding="utf-8")
        writer = csv.writer(out)
        writer.writerow(["fen", "cp", "mate", "move"])
        progress = tqdm(total=args.max_positions, unit="positions")
        for line in reader:
            if positions >= args.max_positions:
                break
            data = json.loads(line)
            if not data["evals"]:
                continue
            cp, mate, move = best_score(data["evals"])
            if not args.all_positions and not is_quiet(data["fen"], move):
                skipped += 1
                continue
            writer.writerow([data["fen"], cp, mate, move])
            positions += 1
            progress.update(1)
        progress.close()

    print(f"{positions} positions -> {args.out} ({skipped} skipped because the best move was a capture or promotion)")


if __name__ == "__main__":
    main()
