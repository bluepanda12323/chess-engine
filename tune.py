import argparse
import json
import os
import random
import threading
import time

import chess
import chess.engine
import chess.pgn

from compare import MATCHES_DIR, OPENING_BOOK, engine_command

MAX_PLIES = 300


def read_openings(path, count):
    openings = []
    with open(path, encoding="utf-8") as f:
        while len(openings) < count:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            openings.append(list(game.mainline_moves()))
    return openings


def spin_options(name):
    command, options = engine_command(name)
    engine = chess.engine.SimpleEngine.popen_uci(command)
    spins = {key: (option.default, option.min, option.max) for key, option in engine.options.items() if option.type == "spin"}
    engine.quit()
    return spins


def play_game(white, black, opening, base, increment):
    board = chess.Board()
    for move in opening:
        board.push(move)
    clocks = {chess.WHITE: base, chess.BLACK: base}
    for engine in (white, black):
        engine.ping()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        engine = white if board.turn == chess.WHITE else black
        limit = chess.engine.Limit(white_clock=clocks[chess.WHITE], black_clock=clocks[chess.BLACK], white_inc=increment, black_inc=increment)
        start = time.time()
        result = engine.play(board, limit, game=id(board))
        clocks[board.turn] -= time.time() - start
        if clocks[board.turn] < 0:
            return 0.0 if board.turn == chess.WHITE else 1.0
        clocks[board.turn] += increment
        board.push(result.move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


class Tuner:

    def __init__(self, names, spins, iterations, learning_rate):
        self.names = names
        self.spins = spins
        self.theta = {name: float(spins[name][0]) for name in names}
        self.iterations = iterations
        self.done = 0
        self.lock = threading.Lock()
        self.a_iterations = 0.1 * iterations
        self.c_end = {name: max(1.0, (spins[name][2] - spins[name][1]) / 20) for name in names}
        self.r_end = learning_rate

    def next_pair(self):
        with self.lock:
            if self.done >= self.iterations:
                return None
            k = self.done + 1
            plus, minus, signs, steps = {}, {}, {}, {}
            for name in self.names:
                c = self.c_end[name] * (self.iterations / k) ** 0.101
                sign = random.choice((-1, 1))
                low, high = self.spins[name][1], self.spins[name][2]
                plus[name] = round(min(high, max(low, self.theta[name] + c * sign)))
                minus[name] = round(min(high, max(low, self.theta[name] - c * sign)))
                signs[name], steps[name] = sign, c
            return k, plus, minus, signs, steps

    def update(self, k, result, signs, steps):
        with self.lock:
            for name in self.names:
                c = steps[name]
                a_end = self.r_end * self.c_end[name] ** 2
                a = a_end * (self.a_iterations + self.iterations) ** 0.602 / (self.a_iterations + k) ** 0.602
                low, high = self.spins[name][1], self.spins[name][2]
                self.theta[name] = min(high, max(low, self.theta[name] + a / c * result * signs[name]))
            self.done += 1

    def snapshot(self):
        with self.lock:
            return {name: round(value, 2) for name, value in self.theta.items()}


def worker(tuner, name, openings, base, increment, log):
    command, options = engine_command(name)
    engines = [chess.engine.SimpleEngine.popen_uci(command) for _ in range(2)]
    try:
        while (pair := tuner.next_pair()) is not None:
            k, plus, minus, signs, steps = pair
            engines[0].configure(options | plus)
            engines[1].configure(options | minus)
            opening = random.choice(openings)
            first = play_game(engines[0], engines[1], opening, base, increment)
            second = 1.0 - play_game(engines[1], engines[0], opening, base, increment)
            tuner.update(k, (first + second) - 1.0, signs, steps)
            if k % 20 == 0:
                log(f"pair {k}/{tuner.iterations}: {tuner.snapshot()}")
    finally:
        for engine in engines:
            engine.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("player")
    parser.add_argument("--params", default="")
    parser.add_argument("--pairs", type=int, default=3000)
    parser.add_argument("--tc", default="5+0.05")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()

    spins = spin_options(args.player)
    names = args.params.split(",") if args.params else [name for name in spins if name not in ("Hash", "Threads")]
    base, increment = (float(part) for part in args.tc.split("+"))
    openings = read_openings(OPENING_BOOK, 5000)
    tuner = Tuner(names, spins, args.pairs, args.learning_rate)

    os.makedirs(MATCHES_DIR, exist_ok=True)
    path = os.path.join(MATCHES_DIR, f"spsa_{args.player.replace(':', '_').replace('=', '-')}_{time.strftime('%Y%m%d_%H%M%S')}")
    start = time.time()

    def log(text):
        line = f"[{(time.time() - start) / 60:6.1f} min] {text}"
        print(line, flush=True)
        with open(path + ".log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"SPSA on {args.player}: {len(names)} parameters, {args.pairs} game pairs at {args.tc}, start {tuner.snapshot()}")
    threads = [threading.Thread(target=worker, args=(tuner, args.player, openings, base, increment, log)) for _ in range(args.concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    final = {name: round(value) for name, value in tuner.theta.items()}
    log(f"Done. Tuned values: {final}")
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({"start": {name: spins[name][0] for name in names}, "tuned": final, "pairs": args.pairs, "tc": args.tc}, f, indent=2)


if __name__ == "__main__":
    main()
