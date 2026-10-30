mod datagen;
mod nnue;
mod search;
mod time;
mod tt;

use std::io::{self, BufRead};
use std::str::FromStr;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, Position, PositionError};

use nnue::Nnue;
use search::{SearchState, Shared};

const DEFAULT_DEPTH: u32 = 5;
const MAX_DEPTH: u32 = 64;
const MAX_HASH: usize = 4096;
const MAX_THREADS: usize = 64;

fn parse_value<T: FromStr>(parts: &[&str], name: &str) -> Option<T> {
    parts
        .iter()
        .position(|&p| p == name)
        .and_then(|i| parts.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_position(args: &[&str]) -> Option<(Chess, Vec<u64>)> {
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

    let mut game = Vec::new();
    if rest.first() == Some(&"moves") {
        for text in &rest[1..] {
            let uci: UciMove = text.parse().ok()?;
            let m = uci.to_move(&pos).ok()?;
            game.push(search::hash(&pos));
            pos.play_unchecked(m);
        }
    }

    Some((pos, game))
}

fn default_weights_path() -> String {
    let next_to_exe = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("nnue.bin")));
    match next_to_exe {
        Some(path) if path.exists() => path.to_string_lossy().to_string(),
        _ => "models/nnue.bin".to_string(),
    }
}

fn halt(shared: &Shared, search: &mut Option<JoinHandle<Vec<SearchState>>>, workers: &mut Vec<SearchState>) {
    if let Some(handle) = search.take() {
        shared.stop();
        *workers = handle.join().expect("the search thread crashed");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("datagen") {
        datagen::run(&args[2..], &default_weights_path());
        return;
    }
    let weights_path = args.get(1).cloned().unwrap_or_else(default_weights_path);
    let nnue = Nnue::load(&weights_path).expect("could not load the NNUE file");
    let mut shared = Arc::new(Shared::new(Arc::new(nnue)));
    let mut workers = vec![SearchState::new()];
    let mut search: Option<JoinHandle<Vec<SearchState>>> = None;
    let mut pos = Chess::default();
    let mut game = Vec::new();
    if std::path::Path::new("tablebases").is_dir() {
        Arc::get_mut(&mut shared).unwrap().set_syzygy_path("tablebases");
    }

    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();

        match parts.first() {
            Some(&"uci") => {
                println!("id name ChessEngineRS");
                println!("id author bluepanda12323");
                println!("option name Hash type spin default {} min 1 max {}", search::DEFAULT_HASH, MAX_HASH);
                println!("option name Threads type spin default 1 min 1 max {}", MAX_THREADS);
                println!("option name Ponder type check default false");
                println!("option name SyzygyPath type string default <empty>");
                println!("option name FallingEval type check default {}", search::Params::new().falling_eval);
                println!("option name Singular type check default {}", search::Params::new().singular);
                for (name, value, min, max) in search::Params::new().spin_options() {
                    println!("option name {} type spin default {} min {} max {}", name, value, min, max);
                }
                println!("uciok");
            }
            Some(&"setoption") => {
                halt(&shared, &mut search, &mut workers);
                if let Some((name, value)) = line.split_once(" value ") {
                    let name = name.trim_start_matches("setoption").trim().trim_start_matches("name").trim();
                    let value = value.trim();
                    match name {
                        "SyzygyPath" => {
                            let path = if value == "<empty>" { "" } else { value };
                            let tables = Arc::get_mut(&mut shared).unwrap().set_syzygy_path(path);
                            println!("info string {} tablebase files found", tables);
                        }
                        "Hash" => {
                            if let Ok(megabytes) = value.parse::<usize>() {
                                Arc::get_mut(&mut shared).unwrap().set_hash(megabytes.clamp(1, MAX_HASH));
                            }
                        }
                        "Threads" => {
                            if let Ok(threads) = value.parse::<usize>() {
                                workers.resize_with(threads.clamp(1, MAX_THREADS), SearchState::new);
                            }
                        }
                        _ => {
                            Arc::get_mut(&mut shared).unwrap().params.set(name, value);
                        }
                    }
                }
            }
            Some(&"isready") => println!("readyok"),
            Some(&"ucinewgame") => {
                halt(&shared, &mut search, &mut workers);
                shared.clear();
                workers.iter_mut().for_each(SearchState::clear);
                pos = Chess::default();
                game.clear();
            }
            Some(&"position") => {
                halt(&shared, &mut search, &mut workers);
                match parse_position(&parts[1..]) {
                    Some((new_pos, new_game)) => {
                        pos = new_pos;
                        game = new_game;
                    }
                    None => println!("info string invalid position: {}", line),
                }
            }
            Some(&"go") => {
                halt(&shared, &mut search, &mut workers);
                let ponder = parts.contains(&"ponder");
                let movetime: Option<u64> = parse_value(&parts, "movetime");
                let (time_name, increment_name) = if pos.turn() == Color::White { ("wtime", "winc") } else { ("btime", "binc") };
                let time_left: Option<i64> = parse_value(&parts, time_name);
                let increment: i64 = parse_value(&parts, increment_name).unwrap_or(0);

                let (soft, hard) = if let Some(ms) = movetime {
                    (search::NO_LIMIT, ms)
                } else if let Some(time_left) = time_left {
                    time::limits(time_left.max(0) as u64, increment.max(0) as u64)
                } else {
                    (search::NO_LIMIT, search::NO_LIMIT)
                };

                let unlimited = hard == search::NO_LIMIT && !ponder && !parts.contains(&"infinite");
                let depth = parse_value(&parts, "depth").unwrap_or(if unlimited { DEFAULT_DEPTH } else { MAX_DEPTH });
                shared.prepare(soft, hard, ponder);
                let (shared, pos, game, mut workers) = (shared.clone(), pos.clone(), game.clone(), std::mem::take(&mut workers));
                let handle = thread::Builder::new()
                    .stack_size(search::STACK_SIZE)
                    .spawn(move || {
                        search::think(&pos, &game, depth, &shared, &mut workers);
                        workers
                    })
                    .expect("could not start the search thread");
                search = Some(handle);
            }
            Some(&"ponderhit") => shared.ponderhit(),
            Some(&"stop") => halt(&shared, &mut search, &mut workers),
            Some(&"eval") => println!("value {:.6}", search::evaluate(&pos, &shared.nnue)),
            Some(&"quit") => break,
            _ => {}
        }
    }

    halt(&shared, &mut search, &mut workers);
}
