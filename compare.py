import argparse
import math
import os
import shutil
import subprocess
import time

import chess
import chess.engine
import chess.pgn

RUST_BUILD = os.path.join("engine-rs", "target", "release", "engine-rs.exe")
WEIGHTS_FILE = os.path.join("models", "nnue.bin")
WEIGHTS_NAMES = ["nnue.bin", "value_weights.bin"]
STOCKFISH_PATH = os.path.join("stockfish", "stockfish.exe")
CARGO = shutil.which("cargo") or os.path.expanduser(os.path.join("~", ".cargo", "bin", "cargo.exe"))
FASTCHESS = os.path.join("tools", "fastchess", "fastchess-windows-x86-64", "fastchess.exe")
OPENING_BOOK = os.path.join("tools", "books", "8moves_v3.pgn")
VERSIONS_DIR = "versions"
MATCHES_DIR = "matches"
MAX_PLIES = 400

OPENINGS = [
    "e4 e5 Nf3 Nc6 Bb5 a6",
    "e4 e5 Nf3 Nc6 Bc4 Bc5",
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4",
    "e4 e5 Nf3 Nf6 Nxe5 d6",
    "e4 e5 Nc3 Nf6 f4 d5",
    "e4 e5 f4 exf4 Nf3 g5",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4",
    "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7",
    "e4 c5 c3 Nf6 e5 Nd5",
    "e4 e6 d4 d5 Nc3 Nf6",
    "e4 c6 d4 d5 e5 Bf5",
    "e4 d5 exd5 Qxd5 Nc3 Qa5",
    "e4 Nf6 e5 Nd5 d4 d6",
    "e4 d6 d4 Nf6 Nc3 g6",
    "e4 g6 d4 Bg7 Nc3 d6",
    "d4 d5 c4 e6 Nc3 Nf6",
    "d4 d5 c4 c6 Nf3 Nf6",
    "d4 d5 c4 dxc4 Nf3 Nf6",
    "d4 d5 Bf4 Nf6 e3 c5",
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6",
    "d4 Nf6 c4 e6 Nc3 Bb4",
    "d4 Nf6 c4 e6 Nf3 b6",
    "d4 Nf6 c4 c5 d5 b5",
    "d4 Nf6 Nf3 e6 Bg5 c5",
    "d4 f5 g3 Nf6 Bg2 g6",
    "c4 e5 Nc3 Nf6 Nf3 Nc6",
    "c4 c5 Nc3 Nc6 g3 g6",
    "c4 Nf6 g3 e6 Bg2 d5",
    "Nf3 d5 g3 Nf6 Bg2 c6",
]


def engine_command(name):
    name, _, extra = name.partition(":")
    options = {}
    for pair in filter(None, extra.split(",")):
        key, value = pair.split("=", 1)
        options[key] = value == "true" if value in ("true", "false") else int(value) if value.lstrip("-").isdigit() else value
    if name == "current":
        command = [os.path.abspath(RUST_BUILD), os.path.abspath(WEIGHTS_FILE)]
    elif name == "stockfish" or name.startswith("stockfish-"):
        command = [os.path.abspath(STOCKFISH_PATH)]
        options["Threads"] = 1
        if name != "stockfish":
            options["UCI_LimitStrength"] = True
            options["UCI_Elo"] = int(name.split("-")[1])
    elif os.path.isdir(os.path.join(VERSIONS_DIR, name)):
        folder = os.path.join(VERSIONS_DIR, name)
        weights = next(os.path.join(folder, w) for w in WEIGHTS_NAMES if os.path.exists(os.path.join(folder, w)))
        command = [os.path.abspath(os.path.join(folder, "engine-rs.exe")), os.path.abspath(weights)]
    else:
        raise SystemExit(f"Unknown player: {name} (see: python compare.py list)")

    if not os.path.exists(command[0]):
        raise SystemExit(f"Not found: {command[0]}")
    return command, options


class Player:

    def __init__(self, name):
        self.name = name
        command, options = engine_command(name)
        self.engine = chess.engine.SimpleEngine.popen_uci(command, timeout=60)
        if options:
            self.engine.configure(options)
        self.moves = 0
        self.total_time = 0.
        self.total_depth = 0

    def play(self, board, limit, game_id):
        start = time.time()
        result = self.engine.play(board, limit, game=game_id, info=chess.engine.INFO_BASIC)
        self.total_time += time.time() - start
        self.total_depth += result.info.get("depth", 0)
        self.moves += 1
        return result.move

    def summary(self):
        if self.moves == 0:
            return f"{self.name}: no moves played"
        return f"{self.name}: average depth {self.total_depth / self.moves:.1f}, {self.total_time / self.moves:.3f} s per move"

    def close(self):
        self.engine.close()


def play_game(white, black, opening, limit, game_id):
    board = chess.Board()
    for san in opening.split():
        board.push_san(san)

    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        player = white if board.turn == chess.WHITE else black
        board.push(player.play(board, limit, game_id))

    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"

    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = f"{white.name} vs {black.name}"
    game.headers["White"] = white.name
    game.headers["Black"] = black.name
    game.headers["Result"] = result
    game.headers["Opening"] = opening
    return result, game


def elo_difference(score):
    score = min(max(score, 0.001), 0.999)
    return -400 * math.log10(1 / score - 1)


def print_results(player_a, player_b, wins, draws, losses):
    games = wins + draws + losses
    if games == 0:
        return

    score = (wins + 0.5 * draws) / games
    variance = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score ** 2) / games
    margin = 1.96 * math.sqrt(variance / games)
    elo = elo_difference(score)
    low, high = elo_difference(score - margin), elo_difference(score + margin)

    print()
    print(f"Result {player_a.name} vs {player_b.name} after {games} games")
    print(f"  {player_a.name}: {wins} wins, {draws} draws, {losses} losses (score {score:.1%})")
    if variance == 0:
        print("  Every game had the same result, so the Elo difference cannot be estimated")
    else:
        print(f"  Elo difference: {round(elo):+d} (95% sure between {round(low):+d} and {round(high):+d})")
        if low > 0:
            print(f"  {player_a.name} is significantly stronger")
        elif high < 0:
            print(f"  {player_b.name} is significantly stronger")
        else:
            print("  Not significant yet: play more games")
    print(f"  {player_a.summary()}")
    print(f"  {player_b.summary()}")


def run_match(name_a, name_b, num_games, movetime, depth):
    engine_command(name_a)
    engine_command(name_b)
    os.makedirs(MATCHES_DIR, exist_ok=True)
    pgn_path = os.path.join(MATCHES_DIR, f"{name_a}_vs_{name_b}_{time.strftime('%Y%m%d_%H%M%S')}.pgn")

    player_a = Player(name_a)
    player_b = Player(name_b)
    wins, draws, losses = 0, 0, 0
    limit = chess.engine.Limit(depth=depth) if depth else chess.engine.Limit(time=movetime)
    print(f"{name_a} vs {name_b}: {num_games} games, " + (f"depth {depth} per move" if depth else f"{movetime} s per move"))

    try:
        for i in range(num_games):
            opening = OPENINGS[(i // 2) % len(OPENINGS)]
            white, black = (player_a, player_b) if i % 2 == 0 else (player_b, player_a)
            result, game = play_game(white, black, opening, limit, i)

            with open(pgn_path, "a", encoding="utf-8") as f:
                print(game, file=f, end="\n\n")

            if result == "1/2-1/2":
                draws += 1
            elif (result == "1-0") == (white is player_a):
                wins += 1
            else:
                losses += 1
            print(f"Game {i + 1}/{num_games}: {white.name} - {black.name} {result}   ({name_a}: +{wins} ={draws} -{losses})", flush=True)
    except KeyboardInterrupt:
        print("Match stopped")
    finally:
        player_a.close()
        player_b.close()

    print_results(player_a, player_b, wins, draws, losses)
    print(f"  Games saved in {pgn_path}")


def label(name):
    return name.replace(":", "_").replace("=", "-").replace(",", "_")


def run_sprt(name_a, name_b, tc, concurrency, elo1, rounds):
    arguments = [os.path.abspath(FASTCHESS)]
    for name in (name_a, name_b):
        command, options = engine_command(name)
        arguments += ["-engine", f"cmd={command[0]}", f"name={label(name)}"]
        if len(command) > 1:
            arguments.append(f"args={' '.join(command[1:])}")
        arguments += [f"option.{key}={str(value).lower()}" for key, value in options.items()]

    os.makedirs(MATCHES_DIR, exist_ok=True)
    base = os.path.join(MATCHES_DIR, f"sprt_{label(name_a)}_vs_{label(name_b)}_{time.strftime('%Y%m%d_%H%M%S')}")
    arguments += [
        "-each", "proto=uci", f"tc={tc}",
        "-openings", f"file={os.path.abspath(OPENING_BOOK)}", "format=pgn", "order=random",
        "-rounds", str(rounds), "-games", "2", "-repeat",
        "-sprt", "elo0=0", f"elo1={elo1}", "alpha=0.05", "beta=0.05", "model=logistic",
        "-concurrency", str(concurrency), "-ratinginterval", "20",
        "-pgnout", f"file={base}.pgn",
        "-config", f"outname={base}.json",
    ]
    print(f"SPRT {name_a} vs {name_b}: {tc} per game, H0 = {name_a} is not better, H1 = {name_a} is {elo1} Elo better")
    subprocess.run(arguments)


def save_version(name, weights):
    folder = os.path.join(VERSIONS_DIR, name)
    if os.path.exists(folder):
        raise SystemExit(f"Version {name} already exists")

    subprocess.run([CARGO, "build", "--release"], cwd="engine-rs", check=True)
    os.makedirs(folder)
    shutil.copy(RUST_BUILD, os.path.join(folder, "engine-rs.exe"))
    shutil.copy(weights, os.path.join(folder, os.path.basename(weights)))
    print(f"Saved {name} in {folder}")


def list_players():
    print("Rust versions:")
    if os.path.isdir(VERSIONS_DIR):
        for name in sorted(os.listdir(VERSIONS_DIR)):
            print(f"  {name}")
    print("  current (the latest build of engine-rs)")
    print("Stockfish:")
    print("  stockfish (full strength)")
    print("  stockfish-1500 (limited to that Elo, minimum 1320)")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    match_parser = commands.add_parser("match")
    match_parser.add_argument("player_a")
    match_parser.add_argument("player_b")
    match_parser.add_argument("--games", type=int, default=20)
    match_parser.add_argument("--movetime", type=float, default=0.1)
    match_parser.add_argument("--depth", type=int, default=None)

    sprt_parser = commands.add_parser("sprt")
    sprt_parser.add_argument("player_a")
    sprt_parser.add_argument("player_b")
    sprt_parser.add_argument("--tc", default="8+0.08")
    sprt_parser.add_argument("--concurrency", type=int, default=8)
    sprt_parser.add_argument("--elo1", type=float, default=10)
    sprt_parser.add_argument("--rounds", type=int, default=10000)

    save_parser = commands.add_parser("save")
    save_parser.add_argument("name")
    save_parser.add_argument("--weights", default=WEIGHTS_FILE)

    commands.add_parser("list")

    args = parser.parse_args()
    if args.command == "match":
        run_match(args.player_a, args.player_b, args.games, args.movetime, args.depth)
    elif args.command == "sprt":
        run_sprt(args.player_a, args.player_b, args.tc, args.concurrency, args.elo1, args.rounds)
    elif args.command == "save":
        save_version(args.name, args.weights)
    else:
        list_players()


if __name__ == "__main__":
    main()
