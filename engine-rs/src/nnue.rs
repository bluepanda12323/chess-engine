use shakmaty::{Bitboard, Board, Color, Piece, Role, Square};

const PIECE_FEATURES: usize = 768;
const PAWN_FEATURES: usize = 384;
const COUNT_FEATURES: usize = 100;
const MOBILITY_FEATURES: usize = 640;
const MAX_HIDDEN: usize = 1024;
const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = FILE_A << 7;
const RAYS: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];
const JUMPS: [(i32, i32); 8] = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)];
const RAY_ORIENT: [[usize; 8]; 4] = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 3, 2, 5, 4, 7, 6], [1, 0, 2, 3, 6, 7, 4, 5], [1, 0, 3, 2, 7, 6, 5, 4]];
const JUMP_ORIENT: [[usize; 8]; 4] = [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6], [2, 3, 0, 1, 6, 7, 4, 5], [3, 2, 1, 0, 7, 6, 5, 4]];
const COUNT_ROLES: [Role; 5] = [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen];
const MOBILITY_ROLES: [Role; 5] = [Role::Knight, Role::Bishop, Role::Rook, Role::Queen, Role::King];
const MOBILITY_DIRECTIONS: [(usize, usize); 5] = [(0, 8), (4, 8), (0, 4), (0, 8), (0, 8)];
const RAY_MASKS: [[u64; 8]; 64] = direction_masks(&RAYS, 7);
const JUMP_MASKS: [[u64; 8]; 64] = direction_masks(&JUMPS, 1);

pub struct Nnue {
    num_buckets: usize,
    buckets: [usize; 64],
    mirror: bool,
    pawn_features: bool,
    count_features: bool,
    mobility_features: bool,
    hidden: usize,
    layer2: usize,
    screlu: bool,
    feature_weights: Vec<f32>,
    feature_bias: Vec<f32>,
    psqt_weights: Vec<f32>,
    hidden_weights: Vec<f32>,
    hidden_bias: Vec<f32>,
    output_weights: Vec<f32>,
    output_bias: f32,
}

#[derive(Clone, Copy)]
pub struct Accumulator {
    values: [[f32; MAX_HIDDEN]; 2],
    psqt: [f32; 2],
    counts: [u8; 10],
    mobility: [u8; MOBILITY_FEATURES],
}

#[derive(Clone, Copy, PartialEq)]
struct View {
    flip: usize,
    mirror: usize,
    bucket: usize,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn u32(&mut self) -> u32 {
        let value = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        value
    }

    fn f32s(&mut self, count: usize) -> Vec<f32> {
        let values = self.bytes[self.pos..self.pos + 4 * count]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        self.pos += 4 * count;
        values
    }
}

const fn direction_masks(steps: &[(i32, i32); 8], max_length: i32) -> [[u64; 8]; 64] {
    let mut masks = [[0; 8]; 64];
    let mut square = 0;
    while square < 64 {
        let mut direction = 0;
        while direction < 8 {
            let (rank_step, file_step) = steps[direction];
            let mut length = 1;
            while length <= max_length {
                let (rank, file) = (square as i32 / 8 + rank_step * length, square as i32 % 8 + file_step * length);
                if rank < 0 || rank >= 8 || file < 0 || file >= 8 {
                    break;
                }
                masks[square][direction] |= 1 << (rank * 8 + file);
                length += 1;
            }
            direction += 1;
        }
        square += 1;
    }
    masks
}

fn ray_length(square: usize, direction: usize, occupied: u64) -> usize {
    let ray = RAY_MASKS[square][direction];
    let blockers = ray & occupied;
    let free = if blockers == 0 {
        ray
    } else if RAYS[direction].0 * 8 + RAYS[direction].1 > 0 {
        ray & ((1 << blockers.trailing_zeros()) - 1)
    } else {
        ray & !(u64::MAX >> blockers.leading_zeros())
    };
    free.count_ones() as usize
}

fn dot(inputs: &[f32], weights: &[f32]) -> f32 {
    inputs.iter().zip(weights).fold(0.0, |sum, (&input, &weight)| sum.algebraic_add(input.algebraic_mul(weight)))
}

fn activated_dot(values: &[f32], weights: &[f32], squared: bool) -> f32 {
    if squared {
        values.iter().zip(weights).fold(0.0, |sum, (&value, &weight)| {
            let clipped = value.clamp(0.0, 1.0);
            sum.algebraic_add(clipped.algebraic_mul(clipped).algebraic_mul(weight))
        })
    } else {
        values.iter().zip(weights).fold(0.0, |sum, (&value, &weight)| sum.algebraic_add(value.clamp(0.0, 1.0).algebraic_mul(weight)))
    }
}

fn activate_into(inputs: &mut [f32], values: &[f32], squared: bool) {
    for (input, &value) in inputs.iter_mut().zip(values) {
        let clipped = value.clamp(0.0, 1.0);
        *input = if squared { clipped * clipped } else { clipped };
    }
}

fn north_span(squares: u64) -> u64 {
    let mut span = squares << 8;
    span |= span << 8;
    span |= span << 16;
    span | span << 32
}

fn south_span(squares: u64) -> u64 {
    let mut span = squares >> 8;
    span |= span >> 8;
    span |= span >> 16;
    span | span >> 32
}

fn adjacent_files(squares: u64) -> u64 {
    ((squares << 1) & !FILE_A) | ((squares >> 1) & !FILE_H)
}

fn pawn_structure(board: &Board) -> [[u64; 3]; 2] {
    let mut structure = [[0; 3]; 2];
    for color in [Color::White, Color::Black] {
        let own = (board.pawns() & board.by_color(color)).0;
        let enemy = (board.pawns() & board.by_color(color.other())).0;
        let blockers = if color == Color::White { south_span(enemy) } else { north_span(enemy) };
        let files = own | north_span(own) | south_span(own);
        structure[color as usize] = [
            own & !(blockers | adjacent_files(blockers)),
            own & !adjacent_files(files),
            own & (north_span(own) | south_span(own)),
        ];
    }
    structure
}

fn side(color: Color) -> usize {
    if color == Color::White { 0 } else { 1 }
}

fn piece_counts(board: &Board) -> [u8; 10] {
    let mut counts = [0; 10];
    for color in [Color::White, Color::Black] {
        for (kind, &role) in COUNT_ROLES.iter().enumerate() {
            counts[side(color) * 5 + kind] = (board.by_color(color) & board.by_role(role)).count() as u8;
        }
    }
    counts
}

fn mobility(board: &Board) -> [u8; MOBILITY_FEATURES] {
    let mut histogram = [0; MOBILITY_FEATURES];
    let occupied = board.occupied().0;
    for (square, piece) in board {
        let Some(kind) = MOBILITY_ROLES.iter().position(|&role| role == piece.role) else {
            continue;
        };
        let square = square.to_usize();
        let (first, last) = MOBILITY_DIRECTIONS[kind];
        for direction in first..last {
            let length = if kind == 0 { usize::from(JUMP_MASKS[square][direction] & !occupied != 0) } else { ray_length(square, direction, occupied) };
            histogram[side(piece.color) * 320 + kind * 64 + direction * 8 + length] += 1;
        }
    }
    histogram
}

fn piece_feature(view: View, perspective: Color, piece: Piece, square: Square) -> usize {
    let own = if piece.color == perspective { 0 } else { 384 };
    view.bucket * PIECE_FEATURES + own + (piece.role as usize - 1) * 64 + (square.to_usize() ^ view.flip ^ view.mirror)
}

impl Nnue {
    pub fn load(path: &str) -> std::io::Result<Nnue> {
        let bytes = std::fs::read(path)?;
        let mut reader = Reader { bytes: &bytes, pos: 0 };

        let nnue = if bytes.starts_with(b"NNU4") {
            reader.pos = 4;
            let features = reader.u32() as usize;
            let hidden = reader.u32() as usize;
            let screlu = reader.u32() == 1;
            assert!(features == PIECE_FEATURES && hidden <= MAX_HIDDEN, "expected a {PIECE_FEATURES} -> at most {MAX_HIDDEN} NNUE, got {features} -> {hidden}");

            Nnue {
                num_buckets: 1,
                buckets: [0; 64],
                mirror: false,
                pawn_features: false,
                count_features: false,
                mobility_features: false,
                hidden,
                layer2: 0,
                screlu,
                feature_weights: reader.f32s(features * hidden),
                feature_bias: reader.f32s(hidden),
                psqt_weights: vec![0.0; features],
                hidden_weights: Vec::new(),
                hidden_bias: Vec::new(),
                output_weights: reader.f32s(2 * hidden),
                output_bias: reader.f32s(1)[0],
            }
        } else if bytes.starts_with(b"NNU2") || bytes.starts_with(b"NNU3") {
            reader.pos = 4;
            let num_buckets = reader.u32() as usize;
            let mirror = reader.u32() == 1;
            let pawn_features = reader.u32() == 1;
            let hidden = reader.u32() as usize;
            let layer2 = reader.u32() as usize;
            let (count_features, mobility_features) = if bytes.starts_with(b"NNU3") { (reader.u32() == 1, reader.u32() == 1) } else { (false, false) };
            let mut buckets = [0; 64];
            for bucket in buckets.iter_mut() {
                *bucket = reader.u32() as usize;
            }
            let features = num_buckets * PIECE_FEATURES
                + if pawn_features { PAWN_FEATURES } else { 0 }
                + if count_features { COUNT_FEATURES } else { 0 }
                + if mobility_features { MOBILITY_FEATURES } else { 0 };
            assert!(hidden <= MAX_HIDDEN, "at most {MAX_HIDDEN} hidden units, got {hidden}");

            Nnue {
                num_buckets,
                buckets,
                mirror,
                pawn_features,
                count_features,
                mobility_features,
                hidden,
                layer2,
                screlu: false,
                feature_weights: reader.f32s(features * hidden),
                feature_bias: reader.f32s(hidden),
                psqt_weights: reader.f32s(features),
                hidden_weights: reader.f32s(layer2 * 2 * hidden),
                hidden_bias: reader.f32s(layer2),
                output_weights: reader.f32s(if layer2 == 0 { 2 * hidden } else { layer2 }),
                output_bias: reader.f32s(1)[0],
            }
        } else {
            let features = reader.u32() as usize;
            let hidden = reader.u32() as usize;
            assert!(features == PIECE_FEATURES && hidden <= MAX_HIDDEN, "expected a {PIECE_FEATURES} -> at most {MAX_HIDDEN} NNUE, got {features} -> {hidden}");

            Nnue {
                num_buckets: 1,
                buckets: [0; 64],
                mirror: false,
                pawn_features: false,
                count_features: false,
                mobility_features: false,
                hidden,
                layer2: 0,
                screlu: false,
                feature_weights: reader.f32s(features * hidden),
                feature_bias: reader.f32s(hidden),
                psqt_weights: vec![0.0; features],
                hidden_weights: Vec::new(),
                hidden_bias: Vec::new(),
                output_weights: reader.f32s(2 * hidden),
                output_bias: reader.f32s(1)[0],
            }
        };

        assert!(reader.pos == bytes.len(), "the NNUE file is {} bytes, expected {}", bytes.len(), reader.pos);
        Ok(nnue)
    }

    fn view(&self, board: &Board, perspective: Color) -> View {
        let flip = if perspective == Color::White { 0 } else { 56 };
        let king = board.king_of(perspective).unwrap().to_usize() ^ flip;
        let mirror = if self.mirror && king % 8 >= 4 { 7 } else { 0 };
        View { flip, mirror, bucket: self.buckets[king] }
    }

    fn pawn_feature(&self, view: View, perspective: Color, color: Color, kind: usize, square: Square) -> usize {
        let own = if color == perspective { 0 } else { 64 };
        self.num_buckets * PIECE_FEATURES + kind * 128 + own + (square.to_usize() ^ view.flip ^ view.mirror)
    }

    fn count_base(&self) -> usize {
        self.num_buckets * PIECE_FEATURES + if self.pawn_features { PAWN_FEATURES } else { 0 }
    }

    fn mobility_base(&self) -> usize {
        self.count_base() + if self.count_features { COUNT_FEATURES } else { 0 }
    }

    fn count_feature(&self, perspective: Color, color: Color, kind: usize, count: u8) -> usize {
        let own = if color == perspective { 0 } else { 50 };
        self.count_base() + own + kind * 10 + (count as usize).min(9)
    }

    fn mobility_feature(&self, view: View, perspective: Color, cell: usize) -> usize {
        let (color, kind, direction, length) = (cell / 320, cell / 64 % 5, cell / 8 % 8, cell % 8);
        let own = if color == side(perspective) { 0 } else { 320 };
        let orientation = usize::from(view.flip != 0) * 2 + usize::from(view.mirror != 0);
        let table = if kind == 0 { &JUMP_ORIENT } else { &RAY_ORIENT };
        self.mobility_base() + own + kind * 64 + table[orientation][direction] * 8 + length
    }

    fn update_counts(&self, acc: &mut Accumulator, perspective: Color, old: &[u8; 10], new: &[u8; 10]) {
        for color in [Color::White, Color::Black] {
            for kind in 0..5 {
                let index = side(color) * 5 + kind;
                if old[index] != new[index] {
                    self.add(acc, perspective, self.count_feature(perspective, color, kind, old[index]), -1.0);
                    self.add(acc, perspective, self.count_feature(perspective, color, kind, new[index]), 1.0);
                }
            }
        }
    }

    fn update_mobility(&self, acc: &mut Accumulator, perspective: Color, view: View, old: &[u8; MOBILITY_FEATURES], new: &[u8; MOBILITY_FEATURES]) {
        for (block, (before, after)) in old.as_chunks::<8>().0.iter().zip(new.as_chunks::<8>().0).enumerate() {
            if before == after {
                continue;
            }
            for (i, (&b, &a)) in before.iter().zip(after).enumerate() {
                if b != a {
                    self.add(acc, perspective, self.mobility_feature(view, perspective, block * 8 + i), a as f32 - b as f32);
                }
            }
        }
    }

    fn column(&self, feature: usize) -> &[f32] {
        &self.feature_weights[feature * self.hidden..(feature + 1) * self.hidden]
    }

    fn add(&self, acc: &mut Accumulator, perspective: Color, feature: usize, sign: f32) {
        for (value, &weight) in acc.values[perspective as usize][..self.hidden].iter_mut().zip(self.column(feature)) {
            *value += sign * weight;
        }
        acc.psqt[perspective as usize] += sign * self.psqt_weights[feature];
    }

    fn apply(&self, acc: &mut Accumulator, perspective: Color, added: &[usize], removed: &[usize]) {
        let values = &mut acc.values[perspective as usize][..self.hidden];
        match (added, removed) {
            (&[plus], &[minus]) => {
                for ((value, &a), &r) in values.iter_mut().zip(self.column(plus)).zip(self.column(minus)) {
                    *value += a - r;
                }
            }
            (&[plus], &[minus1, minus2]) => {
                for (((value, &a), &r1), &r2) in values.iter_mut().zip(self.column(plus)).zip(self.column(minus1)).zip(self.column(minus2)) {
                    *value += a - r1 - r2;
                }
            }
            _ => {
                for &feature in added {
                    for (value, &weight) in values.iter_mut().zip(self.column(feature)) {
                        *value += weight;
                    }
                }
                for &feature in removed {
                    for (value, &weight) in values.iter_mut().zip(self.column(feature)) {
                        *value -= weight;
                    }
                }
            }
        }
        let psqt: f32 = added.iter().map(|&f| self.psqt_weights[f]).sum::<f32>() - removed.iter().map(|&f| self.psqt_weights[f]).sum::<f32>();
        acc.psqt[perspective as usize] += psqt;
    }

    fn update_pawns(&self, acc: &mut Accumulator, perspective: Color, view: View, old: &[[u64; 3]; 2], new: &[[u64; 3]; 2]) {
        for color in [Color::White, Color::Black] {
            for kind in 0..3 {
                let (before, after) = (old[color as usize][kind], new[color as usize][kind]);
                for square in Bitboard(before & !after) {
                    self.add(acc, perspective, self.pawn_feature(view, perspective, color, kind, square), -1.0);
                }
                for square in Bitboard(after & !before) {
                    self.add(acc, perspective, self.pawn_feature(view, perspective, color, kind, square), 1.0);
                }
            }
        }
    }

    fn refresh_side(&self, acc: &mut Accumulator, board: &Board, perspective: Color) {
        let view = self.view(board, perspective);
        acc.values[perspective as usize][..self.hidden].copy_from_slice(&self.feature_bias);
        acc.psqt[perspective as usize] = 0.0;
        for (square, piece) in board {
            self.add(acc, perspective, piece_feature(view, perspective, piece, square), 1.0);
        }
        if self.pawn_features {
            self.update_pawns(acc, perspective, view, &[[0; 3]; 2], &pawn_structure(board));
        }
        if self.count_features {
            for color in [Color::White, Color::Black] {
                for kind in 0..5 {
                    let count = acc.counts[side(color) * 5 + kind];
                    self.add(acc, perspective, self.count_feature(perspective, color, kind, count), 1.0);
                }
            }
        }
        if self.mobility_features {
            let mobility = acc.mobility;
            self.update_mobility(acc, perspective, view, &[0; MOBILITY_FEATURES], &mobility);
        }
    }

    pub fn refresh(&self, board: &Board) -> Accumulator {
        let mut acc = Accumulator { values: [[0.0; MAX_HIDDEN]; 2], psqt: [0.0; 2], counts: [0; 10], mobility: [0; MOBILITY_FEATURES] };
        if self.count_features {
            acc.counts = piece_counts(board);
        }
        if self.mobility_features {
            acc.mobility = mobility(board);
        }
        self.refresh_side(&mut acc, board, Color::White);
        self.refresh_side(&mut acc, board, Color::Black);
        acc
    }

    pub fn update(&self, acc: &Accumulator, before: &Board, after: &Board) -> Accumulator {
        let mut next = *acc;
        let mut changed = (before.white() ^ after.white()) | (before.black() ^ after.black());
        for role in Role::ALL {
            changed |= before.by_role(role) ^ after.by_role(role);
        }
        let pawns_changed = self.pawn_features && before.pawns() != after.pawns();
        let (old_pawns, new_pawns) = if pawns_changed { (pawn_structure(before), pawn_structure(after)) } else { ([[0; 3]; 2], [[0; 3]; 2]) };
        let new_counts = if self.count_features { piece_counts(after) } else { acc.counts };
        let new_mobility = if self.mobility_features { mobility(after) } else { acc.mobility };
        next.counts = new_counts;
        next.mobility = new_mobility;

        for perspective in [Color::White, Color::Black] {
            let view = self.view(after, perspective);
            if view != self.view(before, perspective) {
                self.refresh_side(&mut next, after, perspective);
                continue;
            }
            let (mut added, mut removed) = ([0; 4], [0; 4]);
            let (mut num_added, mut num_removed) = (0, 0);
            for square in changed {
                if let Some(piece) = before.piece_at(square) {
                    removed[num_removed] = piece_feature(view, perspective, piece, square);
                    num_removed += 1;
                }
                if let Some(piece) = after.piece_at(square) {
                    added[num_added] = piece_feature(view, perspective, piece, square);
                    num_added += 1;
                }
            }
            self.apply(&mut next, perspective, &added[..num_added], &removed[..num_removed]);
            if pawns_changed {
                self.update_pawns(&mut next, perspective, view, &old_pawns, &new_pawns);
            }
            if self.count_features {
                self.update_counts(&mut next, perspective, &acc.counts, &new_counts);
            }
            if self.mobility_features {
                self.update_mobility(&mut next, perspective, view, &acc.mobility, &new_mobility);
            }
        }

        next
    }

    pub fn evaluate(&self, acc: &Accumulator, turn: Color) -> f32 {
        let psqt = (acc.psqt[turn as usize] - acc.psqt[turn.other() as usize]) / 2.0;
        if self.layer2 == 0 {
            let (own, other) = self.output_weights.split_at(self.hidden);
            let us = activated_dot(&acc.values[turn as usize][..self.hidden], own, self.screlu);
            let them = activated_dot(&acc.values[turn.other() as usize][..self.hidden], other, self.screlu);
            return self.output_bias + us + them + psqt;
        }

        let mut inputs = [0.0; 2 * MAX_HIDDEN];
        activate_into(&mut inputs[..self.hidden], &acc.values[turn as usize][..self.hidden], self.screlu);
        activate_into(&mut inputs[self.hidden..2 * self.hidden], &acc.values[turn.other() as usize][..self.hidden], self.screlu);
        let inputs = &inputs[..2 * self.hidden];

        let mut output = self.output_bias;
        for (j, row) in self.hidden_weights.chunks_exact(2 * self.hidden).enumerate() {
            let value = self.hidden_bias[j] + dot(inputs, row);
            output += value.clamp(0.0, 1.0) * self.output_weights[j];
        }
        output + psqt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{Chess, Move, Position};

    fn random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn random_weights(seed: &mut u64, count: usize) -> Vec<f32> {
        (0..count).map(|_| (random(seed) % 2001) as f32 / 10000.0 - 0.1).collect()
    }

    fn random_screlu_nnue() -> Nnue {
        let mut seed = 192837465;
        let hidden = 16;
        Nnue {
            num_buckets: 1,
            buckets: [0; 64],
            mirror: false,
            pawn_features: false,
            count_features: false,
            mobility_features: false,
            hidden,
            layer2: 0,
            screlu: true,
            feature_weights: random_weights(&mut seed, PIECE_FEATURES * hidden),
            feature_bias: random_weights(&mut seed, hidden),
            psqt_weights: vec![0.0; PIECE_FEATURES],
            hidden_weights: Vec::new(),
            hidden_bias: Vec::new(),
            output_weights: random_weights(&mut seed, 2 * hidden),
            output_bias: 0.1,
        }
    }

    fn random_nnue() -> Nnue {
        let mut seed = 987654321;
        let (num_buckets, hidden, layer2) = (8, 16, 4);
        let features = num_buckets * PIECE_FEATURES + PAWN_FEATURES + COUNT_FEATURES + MOBILITY_FEATURES;
        let mut buckets = [0; 64];
        for (square, bucket) in buckets.iter_mut().enumerate() {
            *bucket = (square / 8).min(3) * 2 + usize::from(square % 8 % 4 >= 2);
        }

        Nnue {
            num_buckets,
            buckets,
            mirror: true,
            pawn_features: true,
            count_features: true,
            mobility_features: true,
            hidden,
            layer2,
            screlu: false,
            feature_weights: random_weights(&mut seed, features * hidden),
            feature_bias: random_weights(&mut seed, hidden),
            psqt_weights: random_weights(&mut seed, features),
            hidden_weights: random_weights(&mut seed, layer2 * 2 * hidden),
            hidden_bias: random_weights(&mut seed, layer2),
            output_weights: random_weights(&mut seed, layer2),
            output_bias: 0.1,
        }
    }

    fn check_updates(nnue: &Nnue) {
        let mut seed: u64 = 12345;
        let (mut castles, mut en_passants, mut promotions, mut king_moves) = (0, 0, 0, 0);

        for _ in 0..300 {
            let mut pos = Chess::default();
            let mut acc = nnue.refresh(pos.board());

            for _ in 0..200 {
                let moves = pos.legal_moves();
                if moves.is_empty() {
                    break;
                }
                let m = moves[(random(&mut seed) % moves.len() as u64) as usize];

                match m {
                    Move::Castle { .. } => castles += 1,
                    Move::EnPassant { .. } => en_passants += 1,
                    _ if m.is_promotion() => promotions += 1,
                    _ if m.role() == Role::King => king_moves += 1,
                    _ => {}
                }

                let mut child = pos.clone();
                child.play_unchecked(m);
                acc = nnue.update(&acc, pos.board(), child.board());
                pos = child;

                let fresh = nnue.refresh(pos.board());
                for side in 0..2 {
                    for i in 0..nnue.hidden {
                        assert!((acc.values[side][i] - fresh.values[side][i]).abs() < 1e-3);
                    }
                    assert!((acc.psqt[side] - fresh.psqt[side]).abs() < 1e-3);
                }
                assert!((nnue.evaluate(&acc, pos.turn()) - nnue.evaluate(&fresh, pos.turn())).abs() < 1e-3);
            }
        }

        println!("checked: {castles} castles, {en_passants} en passant, {promotions} promotions, {king_moves} king moves");
        assert!(castles > 0 && en_passants > 0 && promotions > 0 && king_moves > 0);
    }

    #[test]
    fn update_matches_refresh() {
        check_updates(&Nnue::load("../models/nnue.bin").unwrap());
    }

    #[test]
    fn update_matches_refresh_with_all_features() {
        check_updates(&random_nnue());
    }

    #[test]
    fn update_matches_refresh_with_screlu() {
        check_updates(&random_screlu_nnue());
    }
}
