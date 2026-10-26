use std::time::Instant;

use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, Move, MoveList, Position, Role};

use crate::encoding::encode;
use crate::network::Network;

pub fn evaluate(pos: &Chess, network: &Network) -> f32 {
    if pos.legal_moves().is_empty() {
        if pos.is_check() {
            return if pos.turn() == Color::White { -1.0 } else { 1.0 };
        }
        return 0.0;
    }
    if pos.is_insufficient_material() {
        return 0.0;
    }
    network.evaluate(&encode(pos))
}

fn piece_value(role: Role) -> i32 {
    match role {
        Role::Pawn => 1,
        Role::Knight => 3,
        Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 0,
    }
}

fn move_priority(m: &Move) -> i32 {
    match m.capture() {
        Some(victim) => 100 + 10 * piece_value(victim) - piece_value(m.role()),
        None => 0,
    }
}

fn ordered_moves(pos: &Chess) -> MoveList {
    let mut moves = pos.legal_moves();
    moves.sort_by_key(|m| -move_priority(m));
    moves
}

fn alphabeta(pos: &Chess, depth: u32, mut alpha: f32, mut beta: f32, network: &Network, nodes: &mut u64) -> f32 {
    let moves = ordered_moves(pos);
    if depth == 0 || moves.is_empty() || pos.is_insufficient_material() {
        *nodes += 1;
        return evaluate(pos, network);
    }

    if pos.turn() == Color::White {
        let mut best = f32::NEG_INFINITY;
        for m in moves {
            let mut child = pos.clone();
            child.play_unchecked(m);
            best = best.max(alphabeta(&child, depth - 1, alpha, beta, network, nodes));
            alpha = alpha.max(best);
            if alpha >= beta {
                break;
            }
        }
        best
    } else {
        let mut best = f32::INFINITY;
        for m in moves {
            let mut child = pos.clone();
            child.play_unchecked(m);
            best = best.min(alphabeta(&child, depth - 1, alpha, beta, network, nodes));
            beta = beta.min(best);
            if alpha >= beta {
                break;
            }
        }
        best
    }
}

fn search_root(pos: &Chess, depth: u32, network: &Network, previous: Option<Move>, nodes: &mut u64) -> (Option<Move>, f32) {
    let white = pos.turn() == Color::White;
    let mut alpha = f32::NEG_INFINITY;
    let mut beta = f32::INFINITY;
    let mut best: Option<Move> = None;
    let mut best_score = 0.0;

    let mut moves: Vec<Move> = ordered_moves(pos).into_iter().collect();
    if let Some(index) = previous.and_then(|p| moves.iter().position(|&m| m == p)) {
        let m = moves.remove(index);
        moves.insert(0, m);
    }

    for m in moves {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let score = alphabeta(&child, depth - 1, alpha, beta, network, nodes);
        let better = if white { score > best_score } else { score < best_score };
        if best.is_none() || better {
            best = Some(m);
            best_score = score;
        }
        if white {
            alpha = alpha.max(score);
        } else {
            beta = beta.min(score);
        }
    }

    (best, best_score)
}

fn value_to_centipawns(value: f32, turn: Color) -> i32 {
    let clamped = value.clamp(-0.9999, 0.9999);
    let white_cp = (clamped.atanh() / 0.00184104).round() as i32;
    if turn == Color::White { white_cp } else { -white_cp }
}

pub fn best_move(pos: &Chess, max_depth: u32, network: &Network) -> Option<Move> {
    let start = Instant::now();
    let mut nodes = 0;
    let mut best = None;

    for depth in 1..=max_depth {
        let (m, score) = search_root(pos, depth, network, best, &mut nodes);
        best = m;

        let millis = start.elapsed().as_millis().max(1);
        let pv = m.map(|m| UciMove::from_move(m, CastlingMode::Standard).to_string()).unwrap_or_default();
        println!(
            "info depth {} score cp {} nodes {} nps {} time {} pv {}",
            depth,
            value_to_centipawns(score, pos.turn()),
            nodes,
            nodes as u128 * 1000 / millis,
            millis,
            pv
        );
        println!("info string value {:.6}", score);
    }

    best
}
