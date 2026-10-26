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
    return pv.get("cp"), pv.get("mate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/evals.csv")
    parser.add_argument("--max-positions", type=int, default=3_000_000)
    args = parser.parse_args()

    positions = 0
    with urllib.request.urlopen(URL) as response, open(args.out, "w", newline="", encoding="utf-8") as out:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(response), encoding="utf-8")
        writer = csv.writer(out)
        writer.writerow(["fen", "cp", "mate"])
        progress = tqdm(total=args.max_positions, unit="positions")
        for line in reader:
            if positions >= args.max_positions:
                break
            data = json.loads(line)
            if not data["evals"]:
                continue
            cp, mate = best_score(data["evals"])
            writer.writerow([data["fen"], cp, mate])
            positions += 1
            progress.update(1)
        progress.close()

    print(f"{positions} positions -> {args.out}")


if __name__ == "__main__":
    main()
