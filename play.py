import os
import random

import chess
import chess.engine
import torch

from ai.encoding import board_to_tensor, move_to_index
from ai.model import ChessModel, ValueModel
from engine import search

SEARCH_DEPTH = 4
RUST_DEPTH = 6
RUST_ENGINE_PATH = os.path.abspath(os.path.join("engine-rs", "target", "release", "engine-rs.exe"))

model = ChessModel(num_inputs=448, num_hidden=400, num_layers=5, num_outputs=4096)
model.load_state_dict(torch.load("models/best_model.tar", weights_only=True, map_location="cpu"))
model.eval()

value_model = ValueModel(num_inputs=448, num_hidden=400, num_layers=5)
value_model.load_state_dict(torch.load("models/value_model.tar", weights_only=True, map_location="cpu"))
value_model.eval()

rust_engine = chess.engine.SimpleEngine.popen_uci([RUST_ENGINE_PATH, "models/value_weights.bin"])


def alphabeta_player(board):
    return search.best_move(board, SEARCH_DEPTH, value_model)


def rust_player(board):
    return rust_engine.play(board, chess.engine.Limit(depth=RUST_DEPTH)).move


def network_player(board):
    x = board_to_tensor(board).unsqueeze(0)
    with torch.no_grad():
        scores = model(x)[0]

    best_move, best_score = None, None
    for move in board.legal_moves:
        if move.promotion not in (None, chess.QUEEN):
            continue
        score = scores[move_to_index(move)].item()
        if best_score is None or score > best_score:
            best_move, best_score = move, score
    return best_move


def random_player(board):
    return random.choice(list(board.legal_moves))


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
        play_game(rust_player, human_player)
    finally:
        rust_engine.quit()
