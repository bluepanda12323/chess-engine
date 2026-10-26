import chess
import torch

from ai.encoding import board_to_tensor


def evaluate(board, model):
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    if board.is_game_over():
        return 0.0
    with torch.no_grad():
        return model(board_to_tensor(board).unsqueeze(0)).item()
