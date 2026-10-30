use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use shakmaty::{Chess, Color, Position};

use crate::nnue::Nnue;
use crate::search::{self, SearchState, Shared};

const RECORD_SIZE: usize = 36;
const RANDOM_PLIES: usize = 8;
const MAX_OPENING_SCORE: i32 = 400;
const WIN_SCORE: i32 = 2000;
const WIN_PLIES: u32 = 6;
const DRAW_SCORE: i32 = 10;
const DRAW_PLIES: u32 = 12;
const DRAW_START: usize = 80;
const MAX_PLIES: usize = 400;
const MAX_DEPTH: u32 = 30;
const HASH_MB: usize = 2;
const REPORT_SECONDS: u64 = 60;

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct Sample {
    board: [u8; 32],
    white: bool,
    score: i16,
}

fn pack(pos: &Chess) -> [u8; 32] {
    let mut board = [0u8; 32];
    for (square, piece) in pos.board() {
        let code = piece.role as u8 + if piece.color == Color::White { 0 } else { 6 };
        let index = square as usize;
        board[index / 2] |= code << (4 * (index % 2));
    }
    board
}

fn is_threefold(pos: &Chess, game: &[u64]) -> bool {
    let key = search::hash(pos);
    game.iter().rev().take(pos.halfmoves() as usize).filter(|&&seen| seen == key).count() >= 2
}

fn play_game(shared: &Shared, state: &mut SearchState, random: &mut Random) -> Vec<u8> {
    shared.clear();
    state.clear();
    let mut pos = Chess::default();
    let mut game = Vec::new();
    for _ in 0..RANDOM_PLIES + random.below(2) {
        let moves = pos.legal_moves();
        if moves.is_empty() {
            return Vec::new();
        }
        game.push(search::hash(&pos));
        pos.play_unchecked(moves[random.below(moves.len())]);
    }
    if pos.legal_moves().is_empty() {
        return Vec::new();
    }
    let (_, score) = search::analyse(&pos, &game, MAX_DEPTH, shared, state);
    if search::centipawns(score).abs() > MAX_OPENING_SCORE {
        return Vec::new();
    }

    let mut samples = Vec::new();
    let (mut white_ahead, mut black_ahead, mut quiet) = (0, 0, 0);
    let result = loop {
        if pos.legal_moves().is_empty() {
            break if !pos.is_check() { 1 } else if pos.turn() == Color::White { 0 } else { 2 };
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 || is_threefold(&pos, &game) || game.len() >= MAX_PLIES {
            break 1;
        }
        if let Some(wdl) = search::tablebase_result(&pos, shared) {
            let white_wdl = if pos.turn() == Color::White { wdl } else { -wdl };
            break (white_wdl + 1) as u8;
        }

        let (m, score) = search::analyse(&pos, &game, MAX_DEPTH, shared, state);
        let Some(m) = m else {
            break 1;
        };
        let cp = search::centipawns(score);
        let white_cp = if pos.turn() == Color::White { cp } else { -cp };
        if !pos.is_check() && !m.is_capture() && !m.is_promotion() && !search::is_decisive(score) {
            samples.push(Sample { board: pack(&pos), white: pos.turn() == Color::White, score: white_cp as i16 });
        }

        white_ahead = if white_cp >= WIN_SCORE { white_ahead + 1 } else { 0 };
        black_ahead = if white_cp <= -WIN_SCORE { black_ahead + 1 } else { 0 };
        quiet = if game.len() >= DRAW_START && cp.abs() <= DRAW_SCORE { quiet + 1 } else { 0 };
        if white_ahead >= WIN_PLIES {
            break 2;
        }
        if black_ahead >= WIN_PLIES {
            break 0;
        }
        if quiet >= DRAW_PLIES {
            break 1;
        }

        game.push(search::hash(&pos));
        pos.play_unchecked(m);
    };

    let mut records = Vec::with_capacity(samples.len() * RECORD_SIZE);
    for sample in samples {
        records.extend_from_slice(&sample.board);
        records.push(u8::from(sample.white));
        records.extend_from_slice(&sample.score.to_le_bytes());
        records.push(result);
    }
    records
}

pub fn run(args: &[String], weights_path: &str) {
    let output = args.first().map_or("selfplay.bin", String::as_str);
    let threads: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(1);
    let nodes: u64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5000);
    let target: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);

    let nnue = Arc::new(Nnue::load(weights_path).expect("could not load the NNUE file"));
    let file = OpenOptions::new().create(true).append(true).open(output).expect("could not open the output file");
    let writer = Mutex::new(BufWriter::new(file));
    let positions = AtomicU64::new(0);
    let games = AtomicU64::new(0);
    let running = AtomicU64::new(threads as u64);
    let start = Instant::now();
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    println!("datagen: {} threads, {} nodes per move, output {}", threads, nodes, output);

    thread::scope(|scope| {
        for id in 0..threads {
            let (writer, positions, games, running, nnue) = (&writer, &positions, &games, &running, nnue.clone());
            thread::Builder::new()
                .stack_size(search::STACK_SIZE)
                .spawn_scoped(scope, move || {
                    let mut shared = Shared::new(nnue);
                    shared.silent = true;
                    shared.node_limit = nodes;
                    shared.set_hash(HASH_MB);
                    if std::path::Path::new("tablebases").is_dir() {
                        shared.set_syzygy_path("tablebases");
                    }
                    let mut state = SearchState::new();
                    let mut random = Random(seed ^ (id as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
                    while positions.load(Ordering::Relaxed) < target {
                        let records = play_game(&shared, &mut state, &mut random);
                        if records.is_empty() {
                            continue;
                        }
                        let mut writer = writer.lock().unwrap();
                        writer.write_all(&records).expect("could not write the output file");
                        writer.flush().expect("could not write the output file");
                        positions.fetch_add((records.len() / RECORD_SIZE) as u64, Ordering::Relaxed);
                        games.fetch_add(1, Ordering::Relaxed);
                    }
                    running.fetch_sub(1, Ordering::Relaxed);
                })
                .expect("could not start a datagen thread");
        }

        let mut next_report = REPORT_SECONDS;
        while running.load(Ordering::Relaxed) > 0 {
            thread::sleep(Duration::from_millis(200));
            let seconds = start.elapsed().as_secs();
            if seconds >= next_report {
                next_report += REPORT_SECONDS;
                let count = positions.load(Ordering::Relaxed);
                println!(
                    "{:.1} h: {} games, {} positions, {:.0} positions/s",
                    seconds as f64 / 3600.0,
                    games.load(Ordering::Relaxed),
                    count,
                    count as f64 / seconds as f64
                );
            }
        }
    });

    println!("datagen done: {} games, {} positions in {:.1} h", games.load(Ordering::Relaxed), positions.load(Ordering::Relaxed), start.elapsed().as_secs_f64() / 3600.0);
}
