import math

import chess

from engine.evaluate import evaluate


def ordered_moves(board):
    moves = list(board.legal_moves)
    moves.sort(key=board.is_capture, reverse=True)
    return moves


def minimax(board, depth, model):
    if depth == 0 or board.is_game_over():
        return evaluate(board, model)

    if board.turn == chess.WHITE:
        best = -math.inf
        for move in list(board.legal_moves):
            board.push(move)
            best = max(best, minimax(board, depth - 1, model))
            board.pop()
        return best
    else:
        best = math.inf
        for move in list(board.legal_moves):
            board.push(move)
            best = min(best, minimax(board, depth - 1, model))
            board.pop()
        return best


def alphabeta(board, depth, alpha, beta, model):
    if depth == 0 or board.is_game_over():
        return evaluate(board, model)

    if board.turn == chess.WHITE:
        best = -math.inf
        for move in ordered_moves(board):
            board.push(move)
            best = max(best, alphabeta(board, depth - 1, alpha, beta, model))
            board.pop()
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best
    else:
        best = math.inf
        for move in ordered_moves(board):
            board.push(move)
            best = min(best, alphabeta(board, depth - 1, alpha, beta, model))
            board.pop()
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best


def best_move(board, depth, model):
    white = board.turn == chess.WHITE
    alpha, beta = -math.inf, math.inf
    best, best_score = None, None

    for move in ordered_moves(board):
        board.push(move)
        score = alphabeta(board, depth - 1, alpha, beta, model)
        board.pop()
        if best is None or (score > best_score if white else score < best_score):
            best, best_score = move, score
        if white:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)

    return best
