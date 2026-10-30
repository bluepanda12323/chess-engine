import os

import chess
import chess.engine

MOVE_TIME = 1.0
ENGINE_PATH = os.path.abspath(os.path.join("engine-rs", "target", "release", "engine-rs.exe"))
WEIGHTS_PATH = os.path.abspath(os.path.join("models", "nnue.bin"))

engine = chess.engine.SimpleEngine.popen_uci([ENGINE_PATH, WEIGHTS_PATH])


def engine_player(board):
    result = engine.play(board, chess.engine.Limit(time=MOVE_TIME), info=chess.engine.INFO_SCORE)
    if "score" in result.info:
        print("Engine eval:", result.info["score"].white())
    return result.move


def human_player(board):
    while True:
        text = input("Your move: ")
        try:
            return board.parse_san(text)
        except ValueError:
            print("Invalid move, try again.")


def play_game(white, black):
    board = chess.Board()
    while not board.is_game_over():
        print(board, "\n")
        player = white if board.turn == chess.WHITE else black
        move = player(board)
        print("Played:", board.san(move), "\n")
        board.push(move)

    print(board)
    print("Result:", board.result())
    return board.result()


if __name__ == "__main__":
    try:
        play_game(engine_player, human_player)
    finally:
        engine.quit()
