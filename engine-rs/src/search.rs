use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{Bitboard, CastlingMode, Chess, Color, EnPassantMode, Move, MoveList, Position, Role, Square};
use shakmaty_syzygy::{Tablebase, Wdl};

use crate::nnue::{Accumulator, Nnue};
use crate::tt::{Bound, Entry, Table};

pub const DEFAULT_HASH: usize = 64;
pub const STACK_SIZE: usize = 32 << 20;
pub const NO_LIMIT: u64 = u64::MAX;
const MATE: f32 = 1000.0;
const MATE_STEP: f32 = 1.0;
const MATE_BOUND: f32 = 500.0;
const TB_WIN: f32 = 200.0;
const TB_BOUND: f32 = 100.0;
const CP_SCALE: f32 = 0.00184104;
const MAX_PLY: usize = 128;
const WINDOW: f32 = 0.0001;
const RFP_DEPTH: u32 = 6;
const FUTILITY_DEPTH: u32 = 3;
const LMP_DEPTH: u32 = 4;
const IIR_DEPTH: u32 = 4;
const SEE_DEPTH: u32 = 8;
const HISTORY_DEPTH: u32 = 3;
const SINGULAR_DEPTH: u32 = 8;
const MAX_HISTORY: i32 = 16384;
const PIECE_SQUARES: usize = 12 * 64;
const CORRECTION_BITS: u32 = 14;
const CORRECTION_LIMIT: f32 = 64.0 * CP_SCALE;
const STABILITY_SCALE: [f32; 6] = [2.0, 1.5, 1.2, 1.0, 0.9, 0.8];
const FALLING_EVAL: f32 = 100.0 * CP_SCALE;
const HARD_NODE_FACTOR: u64 = 16;
const MAX_CENTIPAWNS: f32 = 30000.0;

#[derive(Clone, Copy)]
pub struct Params {
    pub falling_eval: bool,
    pub singular: bool,
    pub singular_margin: i32,
    pub rfp_margin: i32,
    pub futility_margin: i32,
    pub lmp_base: i32,
    pub lmr_base: i32,
    pub lmr_divisor: i32,
    pub nmp_reduction: i32,
    pub quiet_see_margin: i32,
    pub capture_see_margin: i32,
    pub history_margin: i32,
    pub history_lmr: i32,
    pub aspiration: i32,
}

impl Params {
    pub fn new() -> Params {
        Params {
            falling_eval: true,
            singular: true,
            singular_margin: 2,
            rfp_margin: 80,
            futility_margin: 120,
            lmp_base: 3,
            lmr_base: 75,
            lmr_divisor: 225,
            nmp_reduction: 3,
            quiet_see_margin: 70,
            capture_see_margin: 30,
            history_margin: 3000,
            history_lmr: 8192,
            aspiration: 25,
        }
    }

    pub fn spin_options(&self) -> [(&'static str, i32, i32, i32); 12] {
        [
            ("SingularMargin", self.singular_margin, 1, 8),
            ("RfpMargin", self.rfp_margin, 20, 200),
            ("FutilityMargin", self.futility_margin, 40, 300),
            ("LmpBase", self.lmp_base, 1, 8),
            ("LmrBase", self.lmr_base, 0, 200),
            ("LmrDivisor", self.lmr_divisor, 100, 400),
            ("NmpReduction", self.nmp_reduction, 2, 5),
            ("QuietSeeMargin", self.quiet_see_margin, 20, 150),
            ("CaptureSeeMargin", self.capture_see_margin, 10, 100),
            ("HistoryMargin", self.history_margin, 500, 8000),
            ("HistoryLmr", self.history_lmr, 2048, 16384),
            ("Aspiration", self.aspiration, 10, 60),
        ]
    }

    pub fn set(&mut self, name: &str, value: &str) -> bool {
        if name == "FallingEval" || name == "Singular" {
            let enabled = value == "true";
            if name == "FallingEval" {
                self.falling_eval = enabled;
            } else {
                self.singular = enabled;
            }
            return true;
        }
        let Some((_, _, min, max)) = self.spin_options().into_iter().find(|option| option.0 == name) else {
            return false;
        };
        let Ok(value) = value.parse::<i32>() else {
            return false;
        };
        let value = value.clamp(min, max);
        let field = match name {
            "SingularMargin" => &mut self.singular_margin,
            "RfpMargin" => &mut self.rfp_margin,
            "FutilityMargin" => &mut self.futility_margin,
            "LmpBase" => &mut self.lmp_base,
            "LmrBase" => &mut self.lmr_base,
            "LmrDivisor" => &mut self.lmr_divisor,
            "NmpReduction" => &mut self.nmp_reduction,
            "QuietSeeMargin" => &mut self.quiet_see_margin,
            "CaptureSeeMargin" => &mut self.capture_see_margin,
            "HistoryMargin" => &mut self.history_margin,
            "HistoryLmr" => &mut self.history_lmr,
            _ => &mut self.aspiration,
        };
        *field = value;
        true
    }
}

pub struct Shared {
    pub nnue: Arc<Nnue>,
    pub params: Params,
    pub silent: bool,
    pub node_limit: u64,
    tt: Table,
    tablebase: Tablebase<Chess>,
    stop: AtomicBool,
    pondering: AtomicBool,
    nodes: AtomicU64,
    epoch: Instant,
    start: AtomicU64,
    soft: AtomicU64,
    hard: AtomicU64,
}

impl Shared {
    pub fn new(nnue: Arc<Nnue>) -> Shared {
        Shared {
            nnue,
            params: Params::new(),
            silent: false,
            node_limit: NO_LIMIT,
            tt: Table::new(DEFAULT_HASH),
            tablebase: Tablebase::new(),
            stop: AtomicBool::new(false),
            pondering: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
            epoch: Instant::now(),
            start: AtomicU64::new(0),
            soft: AtomicU64::new(NO_LIMIT),
            hard: AtomicU64::new(NO_LIMIT),
        }
    }

    pub fn set_hash(&mut self, megabytes: usize) {
        self.tt = Table::new(megabytes);
    }

    pub fn set_syzygy_path(&mut self, paths: &str) -> usize {
        self.tablebase = Tablebase::new();
        let mut tables = 0;
        for path in paths.split(';').filter(|p| !p.is_empty()) {
            tables += self.tablebase.add_directory(path).unwrap_or(0);
        }
        tables
    }

    pub fn clear(&self) {
        self.tt.clear();
    }

    pub fn prepare(&self, soft: u64, hard: u64, ponder: bool) {
        self.tt.new_search();
        self.stop.store(false, Ordering::Relaxed);
        self.nodes.store(0, Ordering::Relaxed);
        self.start.store(self.now(), Ordering::Relaxed);
        self.soft.store(soft, Ordering::Relaxed);
        self.hard.store(hard, Ordering::Relaxed);
        self.pondering.store(ponder, Ordering::Relaxed);
    }

    pub fn ponderhit(&self) {
        self.start.store(self.now(), Ordering::Relaxed);
        self.pondering.store(false, Ordering::Relaxed);
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    fn now(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    fn elapsed(&self) -> u64 {
        self.now().saturating_sub(self.start.load(Ordering::Relaxed))
    }

    fn hard_limit_reached(&self) -> bool {
        !self.pondering.load(Ordering::Relaxed) && self.elapsed() >= self.hard.load(Ordering::Relaxed)
    }

    fn soft_limit_reached(&self, scale: f32) -> bool {
        !self.pondering.load(Ordering::Relaxed) && self.elapsed() as f32 >= self.soft.load(Ordering::Relaxed) as f32 * scale
    }
}

pub struct SearchState {
    id: usize,
    nodes: u64,
    stopped: bool,
    timed: bool,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    continuation: Vec<i32>,
    capture_history: Vec<i32>,
    correction: Vec<f32>,
    stack: [Option<usize>; MAX_PLY + 1],
    evals: [f32; MAX_PLY + 1],
    tb_hits: u64,
    root_depth: u32,
    last_score: Option<(Color, f32)>,
    path: [u64; MAX_PLY + 1],
    game: Vec<u64>,
    reductions: [[u32; 64]; 64],
}

impl SearchState {
    pub fn new() -> SearchState {
        SearchState {
            id: 0,
            nodes: 0,
            stopped: false,
            timed: false,
            killers: [[None; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            continuation: vec![0; PIECE_SQUARES * PIECE_SQUARES],
            capture_history: vec![0; PIECE_SQUARES * 6],
            correction: vec![0.0; 2 << CORRECTION_BITS],
            stack: [None; MAX_PLY + 1],
            evals: [0.0; MAX_PLY + 1],
            tb_hits: 0,
            root_depth: 0,
            last_score: None,
            path: [0; MAX_PLY + 1],
            game: Vec::new(),
            reductions: reduction_table(&Params::new()),
        }
    }

    pub fn clear(&mut self) {
        self.history = [[[0; 64]; 64]; 2];
        self.continuation.fill(0);
        self.capture_history.fill(0);
        self.correction.fill(0.0);
        self.last_score = None;
    }
}

fn reduction_table(params: &Params) -> [[u32; 64]; 64] {
    let (base, divisor) = (params.lmr_base as f32 / 100.0, params.lmr_divisor as f32 / 100.0);
    let mut table = [[0; 64]; 64];
    for (depth, row) in table.iter_mut().enumerate().skip(1) {
        for (index, reduction) in row.iter_mut().enumerate().skip(1) {
            *reduction = (base + (depth as f32).ln() * (index as f32).ln() / divisor) as u32;
        }
    }
    table
}

fn in_tablebase(pos: &Chess, shared: &Shared) -> bool {
    pos.board().occupied().count() <= shared.tablebase.max_pieces() && pos.castles().is_empty()
}

fn probe_tablebase(pos: &Chess, ply: usize, shared: &Shared, state: &mut SearchState) -> Option<f32> {
    if pos.halfmoves() != 0 || !in_tablebase(pos, shared) {
        return None;
    }
    let wdl = shared.tablebase.probe_wdl_after_zeroing(pos).ok()?;
    state.tb_hits += 1;
    Some(match wdl {
        Wdl::Win => TB_WIN - ply as f32,
        Wdl::Loss => -TB_WIN + ply as f32,
        _ => 0.0,
    })
}

pub fn evaluate(pos: &Chess, nnue: &Nnue) -> f32 {
    if pos.legal_moves().is_empty() {
        if pos.is_check() {
            return if pos.turn() == Color::White { -MATE } else { MATE };
        }
        return 0.0;
    }
    if pos.is_insufficient_material() {
        return 0.0;
    }
    let value = nnue.evaluate(&nnue.refresh(pos.board()), pos.turn());
    if pos.turn() == Color::White { value } else { -value }
}

fn mated_score(ply: usize) -> f32 {
    -MATE + ply as f32 * MATE_STEP
}

fn score_to_tt(score: f32, ply: usize) -> f32 {
    if score > MATE_BOUND {
        score + ply as f32 * MATE_STEP
    } else if score < -MATE_BOUND {
        score - ply as f32 * MATE_STEP
    } else {
        score
    }
}

fn score_from_tt(score: f32, ply: usize) -> f32 {
    if score > MATE_BOUND {
        score - ply as f32 * MATE_STEP
    } else if score < -MATE_BOUND {
        score + ply as f32 * MATE_STEP
    } else {
        score
    }
}

fn tt_cutoff(entry: &Entry, ply: usize, alpha: f32, beta: f32) -> Option<f32> {
    let score = score_from_tt(entry.score, ply);
    let usable = match entry.bound {
        Bound::Exact => true,
        Bound::Lower => score >= beta,
        Bound::Upper => score <= alpha,
    };
    usable.then_some(score)
}

pub fn hash(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

fn move_code(m: &Move) -> u16 {
    m.from().map_or(0, |from| from as u16) | (m.to() as u16) << 6 | m.promotion().map_or(0, |role| role as u16) << 12
}

fn correction_index(pos: &Chess) -> usize {
    let board = pos.board();
    let white = u64::from(board.pawns() & board.by_color(Color::White));
    let black = u64::from(board.pawns() & board.by_color(Color::Black));
    let mixed = white.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ black.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    ((pos.turn() as usize) << CORRECTION_BITS) + (mixed >> (64 - CORRECTION_BITS)) as usize
}

fn is_repetition(key: u64, ply: usize, halfmoves: usize, state: &SearchState) -> bool {
    (2..=halfmoves).step_by(2).any(|back| {
        let seen = if back <= ply { Some(state.path[ply - back]) } else { state.game.len().checked_sub(back - ply).map(|i| state.game[i]) };
        seen == Some(key)
    })
}

fn has_pieces(pos: &Chess) -> bool {
    let board = pos.board();
    (board.by_color(pos.turn()) & !(board.pawns() | board.kings())).any()
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

fn see_value(role: Role) -> i32 {
    if role == Role::King { 100 } else { piece_value(role) }
}

fn see(pos: &Chess, m: &Move) -> i32 {
    let board = pos.board();
    let Some(from) = m.from() else {
        return 0;
    };
    let to = m.to();
    let mut occupied = board.occupied() ^ Bitboard::from(from);
    if let Move::EnPassant { .. } = m {
        occupied ^= Bitboard::from(Square::from_coords(to.file(), from.rank()));
    }

    let mut gains = [0; 32];
    gains[0] = m.capture().map_or(0, see_value);
    let mut on_square = see_value(m.promotion().unwrap_or(m.role()));
    let mut color = pos.turn().other();
    let mut depth = 0;
    while depth < 31 {
        let attackers = board.attacks_to(to, color, occupied) & occupied;
        let Some(role) = Role::ALL.into_iter().find(|&role| (attackers & board.by_role(role)).any()) else {
            break;
        };
        depth += 1;
        gains[depth] = on_square - gains[depth - 1];
        on_square = see_value(role);
        occupied ^= Bitboard::from((attackers & board.by_role(role)).first().unwrap());
        color = color.other();
    }
    while depth > 0 {
        gains[depth - 1] = -(-gains[depth - 1]).max(gains[depth]);
        depth -= 1;
    }
    gains[0]
}

fn capture_priority(m: &Move) -> i32 {
    match m.capture() {
        Some(victim) => 10 * piece_value(victim) - piece_value(m.role()),
        None => -100,
    }
}

fn piece_square(turn: Color, m: &Move) -> usize {
    (turn as usize * 6 + m.role() as usize - 1) * 64 + m.to() as usize
}

fn capture_index(turn: Color, m: &Move) -> usize {
    piece_square(turn, m) * 6 + m.capture().map_or(0, |victim| victim as usize - 1)
}

fn previous_moves(ply: usize, state: &SearchState) -> [Option<usize>; 2] {
    [1, 2].map(|back| ply.checked_sub(back).and_then(|p| state.stack[p]))
}

fn quiet_history(turn: Color, m: &Move, ply: usize, state: &SearchState) -> i32 {
    let current = piece_square(turn, m);
    let butterfly = state.history[turn as usize][m.from().unwrap() as usize][m.to() as usize];
    let continuation: i32 = previous_moves(ply, state).into_iter().flatten().map(|previous| state.continuation[previous * PIECE_SQUARES + current]).sum();
    butterfly + continuation
}

fn history_bonus(depth: u32) -> i32 {
    (150 * depth as i32 - 100).min(1500)
}

fn apply_bonus(entry: &mut i32, bonus: i32) {
    *entry += bonus - *entry * bonus.abs() / MAX_HISTORY;
}

fn update_quiet(turn: Color, m: &Move, ply: usize, bonus: i32, state: &mut SearchState) {
    let current = piece_square(turn, m);
    apply_bonus(&mut state.history[turn as usize][m.from().unwrap() as usize][m.to() as usize], bonus);
    for previous in previous_moves(ply, state).into_iter().flatten() {
        apply_bonus(&mut state.continuation[previous * PIECE_SQUARES + current], bonus);
    }
}

fn move_priority(pos: &Chess, m: &Move, tt_move: u16, ply: usize, state: &SearchState) -> i32 {
    if move_code(m) == tt_move {
        return 1_000_000;
    }
    if let Some(victim) = m.capture() {
        let base = if see(pos, m) >= 0 { 300_000 } else { -300_000 };
        return base + 4096 * piece_value(victim) + state.capture_history[capture_index(pos.turn(), m)];
    }
    if let Some(promotion) = m.promotion() {
        return if promotion == Role::Queen { 200_000 } else { -400_000 };
    }
    if state.killers[ply][0] == Some(*m) {
        return 100_000;
    }
    if state.killers[ply][1] == Some(*m) {
        return 90_000;
    }
    quiet_history(pos.turn(), m, ply, state)
}

fn order_moves(pos: &Chess, moves: &mut MoveList, tt_move: u16, ply: usize, state: &SearchState) {
    moves.sort_by_cached_key(|m| -move_priority(pos, m, tt_move, ply, state));
}

fn out_of_time(shared: &Shared, state: &mut SearchState) -> bool {
    if state.timed && !state.stopped {
        let time_up = state.id == 0 && state.nodes % 1024 == 0 && (shared.hard_limit_reached() || state.nodes >= shared.node_limit.saturating_mul(HARD_NODE_FACTOR));
        if time_up || shared.stop.load(Ordering::Relaxed) {
            state.stopped = true;
        }
    }
    state.stopped
}

fn count_node(shared: &Shared, state: &mut SearchState) {
    state.nodes += 1;
    if state.nodes % 1024 == 0 {
        shared.nodes.fetch_add(1024, Ordering::Relaxed);
    }
}

fn quiescence(pos: &Chess, acc: &Accumulator, mut alpha: f32, beta: f32, ply: usize, shared: &Shared, state: &mut SearchState) -> f32 {
    if out_of_time(shared, state) {
        return 0.0;
    }
    count_node(shared, state);

    let mut moves = pos.legal_moves();
    if moves.is_empty() {
        return if pos.is_check() { mated_score(ply) } else { 0.0 };
    }
    if pos.is_insufficient_material() {
        return 0.0;
    }
    if ply >= MAX_PLY {
        return shared.nnue.evaluate(acc, pos.turn());
    }
    if let Some(score) = shared.tt.probe(hash(pos)).and_then(|entry| tt_cutoff(&entry, ply, alpha, beta)) {
        return score;
    }

    let mut best = f32::NEG_INFINITY;
    if !pos.is_check() {
        best = shared.nnue.evaluate(acc, pos.turn()) + state.correction[correction_index(pos)];
        if best >= beta {
            return best;
        }
        alpha = alpha.max(best);
        moves.retain(|m| m.is_capture() && see(pos, m) >= 0);
    }
    moves.sort_by_key(|m| -capture_priority(m));

    for m in moves {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let child_acc = shared.nnue.update(acc, pos.board(), child.board());
        let score = -quiescence(&child, &child_acc, -beta, -alpha, ply + 1, shared, state);
        if state.stopped {
            return 0.0;
        }
        best = best.max(score);
        alpha = alpha.max(score);
        if alpha >= beta {
            break;
        }
    }
    best
}

fn alphabeta(pos: &Chess, acc: &Accumulator, mut depth: u32, mut alpha: f32, mut beta: f32, ply: usize, shared: &Shared, state: &mut SearchState, null_allowed: bool, excluded: u16) -> f32 {
    let key = hash(pos);
    if pos.halfmoves() >= 100 || is_repetition(key, ply, pos.halfmoves() as usize, state) {
        return 0.0;
    }
    if let Some(score) = probe_tablebase(pos, ply, shared, state) {
        return score;
    }
    if depth == 0 || ply >= MAX_PLY {
        return quiescence(pos, acc, alpha, beta, ply, shared, state);
    }
    if out_of_time(shared, state) {
        return 0.0;
    }
    count_node(shared, state);
    state.path[ply] = key;

    alpha = alpha.max(mated_score(ply));
    beta = beta.min(-mated_score(ply + 1));
    if alpha >= beta {
        return alpha;
    }

    let mut tt_move = 0;
    let tt_entry = shared.tt.probe(key);
    if let Some(entry) = tt_entry {
        tt_move = entry.best;
        if excluded == 0 && entry.depth >= depth {
            if let Some(score) = tt_cutoff(&entry, ply, alpha, beta) {
                return score;
            }
        }
    }

    let mut moves = pos.legal_moves();
    if moves.is_empty() {
        return if pos.is_check() { mated_score(ply) } else { 0.0 };
    }
    if pos.is_insufficient_material() {
        return 0.0;
    }

    if excluded == 0 && depth >= IIR_DEPTH && tt_move == 0 {
        depth -= 1;
    }

    let in_check = pos.is_check();
    let pv_node = beta - alpha > 2.0 * WINDOW;
    let correction_slot = correction_index(pos);
    let raw_eval = if in_check { -MATE } else { shared.nnue.evaluate(acc, pos.turn()) };
    let static_eval = if in_check { -MATE } else { raw_eval + state.correction[correction_slot] };
    state.evals[ply] = static_eval;
    let improving = !in_check && (ply < 2 || static_eval > state.evals[ply - 2]);
    if excluded == 0 && !pv_node && !in_check && depth <= RFP_DEPTH && beta.abs() < TB_BOUND && static_eval - shared.params.rfp_margin as f32 * CP_SCALE * (depth - u32::from(improving)) as f32 >= beta {
        return static_eval;
    }

    let null_reduction = shared.params.nmp_reduction as u32;
    if excluded == 0 && null_allowed && depth >= 3 && depth >= null_reduction && beta.abs() < MATE_BOUND && !in_check && static_eval >= beta && has_pieces(pos) {
        state.stack[ply] = None;
        if let Ok(null_pos) = pos.clone().swap_turn() {
            let score = -alphabeta(&null_pos, acc, depth - null_reduction, -beta, -beta + WINDOW, ply + 1, shared, state, false, 0);
            if state.stopped {
                return 0.0;
            }
            if score >= beta {
                return beta;
            }
        }
    }

    let turn = pos.turn();
    let count = moves.len();
    let mut scores = [0; 256];
    let mut scored = false;
    if let Some(index) = moves.iter().position(|m| move_code(m) == tt_move) {
        moves.swap(0, index);
    } else {
        tt_move = 0;
    }

    let original_alpha = alpha;
    let futile = !pv_node && !in_check && depth <= FUTILITY_DEPTH && static_eval + shared.params.futility_margin as f32 * CP_SCALE * depth as f32 <= alpha;
    let late_moves = (shared.params.lmp_base as usize + (depth * depth) as usize) / (2 - usize::from(improving));
    let mut best = f32::NEG_INFINITY;
    let mut best_move = None;
    let mut quiets_tried = MoveList::new();
    let mut captures_tried = MoveList::new();
    for i in 0..count {
        if i > 0 || tt_move == 0 {
            if !scored {
                for j in i..count {
                    scores[j] = move_priority(pos, &moves[j], tt_move, ply, state);
                }
                scored = true;
            }
            let pick = (i..count).fold(i, |best, j| if scores[j] > scores[best] { j } else { best });
            moves.swap(i, pick);
            scores.swap(i, pick);
        }
        let m = moves[i];
        if excluded != 0 && move_code(&m) == excluded {
            continue;
        }
        let quiet = !m.is_capture() && !m.is_promotion();
        let history = if quiet { quiet_history(turn, &m, ply, state) } else { 0 };

        let mut extension = 0;
        if i == 0 && tt_move != 0 && excluded == 0 && shared.params.singular && depth >= SINGULAR_DEPTH && ply < 2 * state.root_depth as usize {
            if let Some(entry) = tt_entry {
                let tt_score = score_from_tt(entry.score, ply);
                if entry.depth + 3 >= depth && entry.bound != Bound::Upper && tt_score.abs() < TB_BOUND {
                    let singular_beta = tt_score - shared.params.singular_margin as f32 * CP_SCALE * depth as f32;
                    let score = alphabeta(pos, acc, (depth - 1) / 2, singular_beta - WINDOW, singular_beta, ply, shared, state, false, tt_move);
                    if state.stopped {
                        return 0.0;
                    }
                    if score < singular_beta {
                        extension = 1;
                    } else if singular_beta >= beta {
                        return singular_beta;
                    }
                }
            }
        }

        let mut child = pos.clone();
        child.play_unchecked(m);
        let gives_check = child.is_check();
        if i > 0 && !gives_check && best > -MATE_BOUND {
            if quiet && futile {
                continue;
            }
            if quiet && !pv_node && !in_check && depth <= LMP_DEPTH && i >= late_moves {
                continue;
            }
            if quiet && !pv_node && !in_check && depth <= HISTORY_DEPTH && history < -shared.params.history_margin * depth as i32 {
                continue;
            }
            if !in_check && depth <= SEE_DEPTH {
                let threshold = if quiet { -shared.params.quiet_see_margin * depth as i32 } else { -shared.params.capture_see_margin * (depth * depth) as i32 };
                if 100 * see(pos, &m) < threshold {
                    continue;
                }
            }
        }
        let child_acc = shared.nnue.update(acc, pos.board(), child.board());
        let new_depth = depth - 1 + u32::from(gives_check).max(extension);
        state.stack[ply] = Some(piece_square(turn, &m));

        let score = if i == 0 {
            -alphabeta(&child, &child_acc, new_depth, -beta, -alpha, ply + 1, shared, state, true, 0)
        } else {
            let mut reduction = 0;
            if depth >= 3 && i >= 3 && quiet && !in_check && !gives_check {
                let base = state.reductions[(depth as usize).min(63)][i.min(63)] as i32;
                let adjusted = base - i32::from(pv_node) + i32::from(!improving) - history / shared.params.history_lmr;
                reduction = adjusted.clamp(0, new_depth as i32 - 1) as u32;
            }
            let mut score = -alphabeta(&child, &child_acc, new_depth - reduction, -alpha - WINDOW, -alpha, ply + 1, shared, state, true, 0);
            if score > alpha && reduction > 0 {
                score = -alphabeta(&child, &child_acc, new_depth, -alpha - WINDOW, -alpha, ply + 1, shared, state, true, 0);
            }
            if score > alpha && score < beta {
                score = -alphabeta(&child, &child_acc, new_depth, -beta, -alpha, ply + 1, shared, state, true, 0);
            }
            score
        };
        if state.stopped {
            return 0.0;
        }
        if score > best {
            best = score;
            best_move = Some(m);
        }
        alpha = alpha.max(score);
        if alpha >= beta {
            let bonus = history_bonus(depth);
            if quiet {
                if state.killers[ply][0] != Some(m) {
                    state.killers[ply][1] = state.killers[ply][0];
                    state.killers[ply][0] = Some(m);
                }
                update_quiet(turn, &m, ply, bonus, state);
                for tried in &quiets_tried {
                    update_quiet(turn, tried, ply, -bonus, state);
                }
            } else if m.is_capture() {
                apply_bonus(&mut state.capture_history[capture_index(turn, &m)], bonus);
            }
            for tried in &captures_tried {
                apply_bonus(&mut state.capture_history[capture_index(turn, tried)], -bonus);
            }
            break;
        }
        if quiet {
            quiets_tried.push(m);
        } else if m.is_capture() {
            captures_tried.push(m);
        }
    }

    if excluded != 0 {
        return best;
    }
    let bound = if best <= original_alpha {
        Bound::Upper
    } else if best >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    let quiet_best = best_move.is_none_or(|m: Move| !m.is_capture() && !m.is_promotion());
    let trusted = !(bound == Bound::Lower && best <= static_eval) && !(bound == Bound::Upper && best >= static_eval);
    if !in_check && quiet_best && trusted && best.abs() < TB_BOUND {
        let weight = ((depth + 1) * (depth + 1)).min(128) as f32 / 256.0;
        let entry = &mut state.correction[correction_slot];
        *entry = (*entry * (1.0 - weight) + (best - raw_eval) * weight).clamp(-CORRECTION_LIMIT, CORRECTION_LIMIT);
    }
    shared.tt.store(key, Entry { depth, score: score_to_tt(best, ply), bound, best: best_move.map_or(0, |m| move_code(&m)) });
    best
}

fn search_root(pos: &Chess, depth: u32, mut alpha: f32, beta: f32, shared: &Shared, previous: u16, state: &mut SearchState) -> (Option<Move>, f32, f32) {
    let acc = shared.nnue.refresh(pos.board());
    state.path[0] = hash(pos);
    state.evals[0] = if pos.is_check() { -MATE } else { shared.nnue.evaluate(&acc, pos.turn()) + state.correction[correction_index(pos)] };
    let mut moves = pos.legal_moves();
    order_moves(pos, &mut moves, previous, 0, state);

    let start_nodes = state.nodes;
    let mut best = None;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_nodes = 0;
    for (i, m) in moves.into_iter().enumerate() {
        let move_start = state.nodes;
        let mut child = pos.clone();
        child.play_unchecked(m);
        let child_acc = shared.nnue.update(&acc, pos.board(), child.board());
        let new_depth = depth - 1 + u32::from(child.is_check());
        state.stack[0] = Some(piece_square(pos.turn(), &m));

        let score = if i == 0 {
            -alphabeta(&child, &child_acc, new_depth, -beta, -alpha, 1, shared, state, true, 0)
        } else {
            let score = -alphabeta(&child, &child_acc, new_depth, -alpha - WINDOW, -alpha, 1, shared, state, true, 0);
            if score > alpha && score < beta {
                -alphabeta(&child, &child_acc, new_depth, -beta, -alpha, 1, shared, state, true, 0)
            } else {
                score
            }
        };
        if state.stopped {
            break;
        }
        if score > best_score {
            best_score = score;
            best = Some(m);
            best_nodes = state.nodes - move_start;
        }
        alpha = alpha.max(score);
        if alpha >= beta {
            break;
        }
    }

    let fraction = best_nodes as f32 / (state.nodes - start_nodes).max(1) as f32;
    (best, best_score, fraction)
}

fn uci_score(score: f32) -> String {
    if score.abs() > MATE_BOUND {
        let plies = ((MATE - score.abs()) / MATE_STEP).round() as i32;
        let moves = (plies + 1) / 2;
        return if score > 0.0 { format!("mate {}", moves) } else { format!("mate -{}", moves) };
    }
    if score.abs() > TB_BOUND {
        return if score > 0.0 { "cp 20000".to_string() } else { "cp -20000".to_string() };
    }
    format!("cp {}", (score / CP_SCALE).round() as i32)
}

fn uci_move(m: Move) -> UciMove {
    UciMove::from_move(m, CastlingMode::Standard)
}


fn best_move(pos: &Chess, game: &[u64], max_depth: u32, shared: &Shared, state: &mut SearchState) -> (Option<Move>, f32) {
    let start = Instant::now();
    let main = state.id == 0;
    let report = main && !shared.silent;
    state.nodes = 0;
    state.stopped = false;
    state.timed = false;
    state.killers = [[None; 2]; MAX_PLY];
    state.tb_hits = 0;
    state.game = game.to_vec();
    state.reductions = reduction_table(&shared.params);

    if pos.legal_moves().is_empty() {
        return (None, 0.0);
    }

    if in_tablebase(pos, shared) {
        if let Ok(Some((m, dtz))) = shared.tablebase.best_move(pos) {
            let score = -dtz.signum() as f32 * TB_WIN;
            if report {
                println!("info depth 1 score {} tbhits 1 pv {}", uci_score(score), uci_move(m));
                println!("info string tablebase move {:?}", dtz);
            }
            return (Some(m), score);
        }
    }

    let mut best = None;
    let mut previous_score: f32 = 0.0;
    let mut stability = 0;
    let last_score = state.last_score.filter(|&(side, last)| side == pos.turn() && last.abs() < TB_BOUND).map(|(_, last)| last);

    for iteration in 1..=max_depth {
        let depth = if main { iteration } else { (iteration + (state.id % 2) as u32).min(max_depth) };
        state.root_depth = depth;
        state.timed = iteration > 1;
        let (mut alpha, mut beta) = if depth >= 4 && previous_score.abs() < TB_BOUND {
            let window = shared.params.aspiration as f32 * CP_SCALE;
            (previous_score - window, previous_score + window)
        } else {
            (f32::NEG_INFINITY, f32::INFINITY)
        };
        let mut hint = best.map_or(0, |m| move_code(&m));
        let (m, score, fraction) = loop {
            let (m, score, fraction) = search_root(pos, depth, alpha, beta, shared, hint, state);
            let full_window = alpha == f32::NEG_INFINITY && beta == f32::INFINITY;
            if state.stopped || full_window || (score > alpha && score < beta) {
                break (m, score, fraction);
            }
            if score <= alpha {
                alpha = f32::NEG_INFINITY;
            } else {
                beta = f32::INFINITY;
            }
            if let Some(m) = m {
                hint = move_code(&m);
            }
        };
        if state.stopped {
            break;
        }
        stability = if m == best { stability + 1 } else { 0 };
        best = m;
        previous_score = score;
        if !main {
            continue;
        }

        if report {
            let millis = start.elapsed().as_millis().max(1);
            let nodes = shared.nodes.load(Ordering::Relaxed) + state.nodes % 1024;
            let pv = m.map(|m| uci_move(m).to_string()).unwrap_or_default();
            println!(
                "info depth {} score {} nodes {} nps {} tbhits {} time {} pv {}",
                depth,
                uci_score(score),
                nodes,
                nodes as u128 * 1000 / millis,
                state.tb_hits,
                millis,
                pv
            );
        }

        if score > MATE_BOUND && ((MATE - score) / MATE_STEP) as u32 + 2 <= depth {
            break;
        }
        if state.nodes >= shared.node_limit {
            break;
        }
        let falling = last_score.filter(|_| shared.params.falling_eval && score.abs() < TB_BOUND).map_or(1.0, |last| 1.0 + ((last - score) / FALLING_EVAL).clamp(0.0, 1.0));
        let scale = STABILITY_SCALE[stability.min(STABILITY_SCALE.len() - 1)] * (1.5 - fraction) * 1.3 * falling;
        if shared.soft_limit_reached(scale) {
            break;
        }
    }

    if main && best.is_some() {
        state.last_score = Some((pos.turn(), previous_score));
    }
    (best.or_else(|| pos.legal_moves().first().copied()), previous_score)
}

pub fn analyse(pos: &Chess, game: &[u64], max_depth: u32, shared: &Shared, state: &mut SearchState) -> (Option<Move>, f32) {
    shared.prepare(NO_LIMIT, NO_LIMIT, false);
    state.id = 0;
    best_move(pos, game, max_depth, shared, state)
}

pub fn centipawns(score: f32) -> i32 {
    (score / CP_SCALE).clamp(-MAX_CENTIPAWNS, MAX_CENTIPAWNS).round() as i32
}

pub fn is_decisive(score: f32) -> bool {
    score.abs() >= TB_BOUND
}

pub fn tablebase_result(pos: &Chess, shared: &Shared) -> Option<i32> {
    if pos.halfmoves() != 0 || !in_tablebase(pos, shared) {
        return None;
    }
    Some(match shared.tablebase.probe_wdl_after_zeroing(pos).ok()? {
        Wdl::Win => 1,
        Wdl::Loss => -1,
        _ => 0,
    })
}

fn ponder_move(pos: &Chess, m: Move, shared: &Shared) -> Option<Move> {
    let mut child = pos.clone();
    child.play_unchecked(m);
    let entry = shared.tt.probe(hash(&child))?;
    child.legal_moves().into_iter().find(|reply| move_code(reply) == entry.best)
}

pub fn think(pos: &Chess, game: &[u64], max_depth: u32, shared: &Shared, workers: &mut [SearchState]) {
    let best = thread::scope(|scope| {
        let (main, helpers) = workers.split_first_mut().unwrap();
        for (i, helper) in helpers.iter_mut().enumerate() {
            helper.id = i + 1;
            thread::Builder::new().stack_size(STACK_SIZE).spawn_scoped(scope, move || best_move(pos, game, max_depth, shared, helper)).unwrap();
        }
        main.id = 0;
        let (best, _) = best_move(pos, game, max_depth, shared, main);
        while shared.pondering.load(Ordering::Relaxed) && !shared.stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(1));
        }
        shared.stop();
        best
    });

    match best {
        Some(m) => match ponder_move(pos, m, shared) {
            Some(reply) => println!("bestmove {} ponder {}", uci_move(m), uci_move(reply)),
            None => println!("bestmove {}", uci_move(m)),
        },
        None => println!("bestmove 0000"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;

    fn see_of(fen: &str, uci: &str) -> i32 {
        let pos: Chess = fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap();
        let m = uci.parse::<UciMove>().unwrap().to_move(&pos).unwrap();
        see(&pos, &m)
    }

    #[test]
    fn static_exchange() {
        assert_eq!(see_of("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"), 1);
        assert_eq!(see_of("4k3/8/2p5/3p4/8/5Q2/8/4K3 w - - 0 1", "f3d5"), -8);
        assert_eq!(see_of("4k3/8/2p5/3n4/4P3/8/8/4K3 w - - 0 1", "e4d5"), 2);
        assert_eq!(see_of("3rk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5"), 1);
        assert_eq!(see_of("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"), 1);
    }
}
