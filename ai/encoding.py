import chess
import torch


def board_to_tensor(board):
    tensor = torch.zeros((7, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        layer = piece.piece_type - 1
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        if piece.color == chess.WHITE:
            tensor[layer, row, col] = 1
        else:
            tensor[layer, row, col] = -1

    if board.turn == chess.WHITE:
        tensor[6] = 1

    return tensor


def move_to_index(move):
    return move.from_square * 64 + move.to_square


def index_to_move(index, board):
    from_square = index // 64
    to_square = index % 64

    promotion = None
    piece = board.piece_at(from_square)
    if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
        promotion = chess.QUEEN

    return chess.Move(from_square, to_square, promotion=promotion)
