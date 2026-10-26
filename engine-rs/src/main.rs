mod encoding;
mod network;
mod search;

use std::io::{self, BufRead};

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position, PositionError};

use network::Network;

const DEFAULT_DEPTH: u32 = 5;

fn parse_position(args: &[&str]) -> Option<Chess> {
    let (mut pos, rest): (Chess, &[&str]) = match args.first() {
        Some(&"startpos") => (Chess::default(), &args[1..]),
        Some(&"fen") => {
            let end = args.iter().position(|&a| a == "moves").unwrap_or(args.len());
            let fen: Fen = args[1..end].join(" ").parse().ok()?;
            let pos = fen
                .into_position(CastlingMode::Standard)
                .or_else(PositionError::ignore_invalid_castling_rights)
                .or_else(PositionError::ignore_invalid_ep_square)
                .ok()?;
            (pos, &args[end..])
        }
        _ => return None,
    };

    if rest.first() == Some(&"moves") {
        for text in &rest[1..] {
            let uci: UciMove = text.parse().ok()?;
            let m = uci.to_move(&pos).ok()?;
            pos.play_unchecked(m);
        }
    }

    Some(pos)
}

fn main() {
    let weights_path = std::env::args().nth(1).unwrap_or_else(|| "models/value_weights.bin".to_string());
    let network = Network::load(&weights_path).expect("could not load the weights file");
    let mut pos = Chess::default();

    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();

        match parts.first() {
            Some(&"uci") => {
                println!("id name ChessEngineRS");
                println!("id author bluepanda12323");
                println!("uciok");
            }
            Some(&"isready") => println!("readyok"),
            Some(&"ucinewgame") => pos = Chess::default(),
            Some(&"position") => match parse_position(&parts[1..]) {
                Some(new_pos) => pos = new_pos,
                None => println!("info string invalid position: {}", line),
            },
            Some(&"go") => {
                let depth = parts
                    .iter()
                    .position(|&p| p == "depth")
                    .and_then(|i| parts.get(i + 1))
                    .and_then(|d| d.parse().ok())
                    .unwrap_or(DEFAULT_DEPTH);
                match search::best_move(&pos, depth, &network) {
                    Some(m) => println!("bestmove {}", UciMove::from_move(m, CastlingMode::Standard)),
                    None => println!("bestmove 0000"),
                }
            }
            Some(&"eval") => println!("value {:.6}", search::evaluate(&pos, &network)),
            Some(&"quit") => break,
            _ => {}
        }
    }
}
