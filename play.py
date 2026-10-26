import random

import chess
import torch

from ai.encoding import board_to_tensor, move_to_index
from ai.model import ChessModel

model = ChessModel(num_inputs=448, num_hidden=400, num_layers=5, num_outputs=4096)
model.load_state_dict(torch.load("models/best_model.tar", weights_only=True))
model.eval()


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
    play_game(human_player, network_player)
