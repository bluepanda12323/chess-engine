use shakmaty::{Bitboard, Board, Color, Piece, Role, Square};

const PIECE_FEATURES: usize = 768;
const PAWN_FEATURES: usize = 384;
const COUNT_FEATURES: usize = 100;
const MOBILITY_FEATURES: usize = 640;
const MAX_HIDDEN: usize = 1024;
const MAX_PIECES: usize = 32;
const MAX_PAWN_FEATURES: usize = 48;
const MAX_COUNT_FEATURES: usize = 10;
const MAX_MOBILITY_TOTAL: usize = 128;
const ACCUMULATOR_LIMIT: f32 = 32000.0;
const OUTPUT_LIMIT: f32 = 2.0e9;
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
    activation_scale: i16,
    feature_weights: Vec<i16>,
    feature_bias: Vec<i16>,
    psqt_weights: Vec<f32>,
    hidden_weights: Vec<f32>,
    quantized_hidden: Vec<i16>,
    hidden_scale: f32,
    hidden_bias: Vec<f32>,
    output_weights: Vec<f32>,
    quantized_output: Vec<i16>,
    output_scale: f32,
    output_bias: f32,
}

#[derive(Clone, Copy)]
pub struct Accumulator {
    values: [[i16; MAX_HIDDEN]; 2],
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

#[cfg(target_feature = "avx2")]
fn clipped_dot(values: &[i16], weights: &[i16], limit: i16) -> i32 {
    use std::arch::x86_64::*;
    let (value_blocks, value_rest) = values.as_chunks::<16>();
    let (weight_blocks, weight_rest) = weights.as_chunks::<16>();
    let mut sums = [0i32; 8];
    unsafe {
        let (low, high) = (_mm256_setzero_si256(), _mm256_set1_epi16(limit));
        let mut total = _mm256_setzero_si256();
        for (value, weight) in value_blocks.iter().zip(weight_blocks) {
            let value = _mm256_loadu_si256(value.as_ptr().cast());
            let weight = _mm256_loadu_si256(weight.as_ptr().cast());
            total = _mm256_add_epi32(total, _mm256_madd_epi16(_mm256_min_epi16(_mm256_max_epi16(value, low), high), weight));
        }
        _mm256_storeu_si256(sums.as_mut_ptr().cast(), total);
    }
    sums.iter().sum::<i32>() + scalar_clipped_dot(value_rest, weight_rest, limit)
}

#[cfg(not(target_feature = "avx2"))]
fn clipped_dot(values: &[i16], weights: &[i16], limit: i16) -> i32 {
    scalar_clipped_dot(values, weights, limit)
}

fn scalar_clipped_dot(values: &[i16], weights: &[i16], limit: i16) -> i32 {
    values.iter().zip(weights).map(|(&value, &weight)| i32::from(value.clamp(0, limit)) * i32::from(weight)).sum()
}

fn activate_into(inputs: &mut [f32], values: &[i16], limit: i16, squared: bool) {
    for (input, &value) in inputs.iter_mut().zip(values) {
        let clipped = value.clamp(0, limit) as f32 / limit as f32;
        *input = if squared { clipped * clipped } else { clipped };
    }
}

fn quantize(values: &[f32], scale: f32) -> Vec<i16> {
    values.iter().map(|&value| (value * scale).round() as i16).collect()
}

fn activation_scale(weights: &[f32], bias: &[f32], groups: &[(usize, usize)]) -> i16 {
    let hidden = bias.len();
    let mut largest = 0.0f32;
    for (unit, &b) in bias.iter().enumerate() {
        let mut bound = b.abs();
        let mut first_row = 0;
        for &(rows, active) in groups.iter().filter(|&&(rows, _)| rows > 0) {
            let mut column: Vec<f32> = (first_row..first_row + rows).map(|row| weights[row * hidden + unit].abs()).collect();
            let first = column.len() - active.min(column.len());
            column.select_nth_unstable_by(first, f32::total_cmp);
            bound += column[first..].iter().sum::<f32>();
            first_row += rows;
        }
        largest = largest.max(bound);
    }
    (ACCUMULATOR_LIMIT / largest).clamp(1.0, i16::MAX as f32) as i16
}

fn output_quantization(weights: &[f32], activation_scale: i16) -> f32 {
    let largest = weights.iter().fold(0.0f32, |largest, weight| largest.max(weight.abs()));
    let total: f32 = weights.iter().map(|weight| weight.abs()).sum();
    (i16::MAX as f32 / largest).min(OUTPUT_LIMIT / (activation_scale as f32 * total))
}

fn hidden_quantization(weights: &[f32], row_length: usize, activation_scale: i16) -> f32 {
    let largest = weights.iter().fold(0.0f32, |largest, weight| largest.max(weight.abs()));
    let total = weights.chunks_exact(row_length.max(1)).map(|row| row.iter().map(|weight| weight.abs()).sum::<f32>()).fold(0.0f32, f32::max);
    (i16::MAX as f32 / largest).min(OUTPUT_LIMIT / (activation_scale as f32 * total))
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
        Ok(Nnue::parse(&std::fs::read(path)?))
    }

    fn parse(bytes: &[u8]) -> Nnue {
        let mut reader = Reader { bytes, pos: 0 };
        let mut buckets = [0; 64];
        let has_psqt = bytes.starts_with(b"NNU2") || bytes.starts_with(b"NNU3");

        let (num_buckets, mirror, pawn_features, count_features, mobility_features, hidden, layer2, screlu) = if has_psqt {
            reader.pos = 4;
            let num_buckets = reader.u32() as usize;
            let mirror = reader.u32() == 1;
            let pawn_features = reader.u32() == 1;
            let hidden = reader.u32() as usize;
            let layer2 = reader.u32() as usize;
            let (count_features, mobility_features) = if bytes.starts_with(b"NNU3") { (reader.u32() == 1, reader.u32() == 1) } else { (false, false) };
            for bucket in buckets.iter_mut() {
                *bucket = reader.u32() as usize;
            }
            (num_buckets, mirror, pawn_features, count_features, mobility_features, hidden, layer2, false)
        } else {
            let has_screlu_flag = bytes.starts_with(b"NNU4");
            reader.pos = if has_screlu_flag { 4 } else { 0 };
            let features = reader.u32() as usize;
            let hidden = reader.u32() as usize;
            let screlu = has_screlu_flag && reader.u32() == 1;
            assert!(features == PIECE_FEATURES, "expected {PIECE_FEATURES} input features, got {features}");
            (1, false, false, false, false, hidden, 0, screlu)
        };
        assert!(hidden <= MAX_HIDDEN, "at most {MAX_HIDDEN} hidden units, got {hidden}");

        let features = num_buckets * PIECE_FEATURES
            + if pawn_features { PAWN_FEATURES } else { 0 }
            + if count_features { COUNT_FEATURES } else { 0 }
            + if mobility_features { MOBILITY_FEATURES } else { 0 };
        let feature_weights = reader.f32s(features * hidden);
        let feature_bias = reader.f32s(hidden);
        let psqt_weights = if has_psqt { reader.f32s(features) } else { vec![0.0; features] };
        let hidden_weights = reader.f32s(layer2 * 2 * hidden);
        let hidden_bias = reader.f32s(layer2);
        let output_weights = reader.f32s(if layer2 == 0 { 2 * hidden } else { layer2 });
        let output_bias = reader.f32s(1)[0];
        assert!(reader.pos == bytes.len(), "the NNUE file is {} bytes, expected {}", bytes.len(), reader.pos);

        let groups = [
            (num_buckets * PIECE_FEATURES, MAX_PIECES),
            (if pawn_features { PAWN_FEATURES } else { 0 }, MAX_PAWN_FEATURES),
            (if count_features { COUNT_FEATURES } else { 0 }, MAX_COUNT_FEATURES),
            (if mobility_features { MOBILITY_FEATURES } else { 0 }, MAX_MOBILITY_TOTAL),
        ];
        let activation_scale = activation_scale(&feature_weights, &feature_bias, &groups);
        let output_quantization = output_quantization(&output_weights, activation_scale);
        let hidden_quantization = hidden_quantization(&hidden_weights, 2 * hidden, activation_scale);

        Nnue {
            num_buckets,
            buckets,
            mirror,
            pawn_features,
            count_features,
            mobility_features,
            hidden,
            layer2,
            screlu,
            activation_scale,
            feature_weights: quantize(&feature_weights, activation_scale as f32),
            feature_bias: quantize(&feature_bias, activation_scale as f32),
            psqt_weights,
            quantized_hidden: quantize(&hidden_weights, hidden_quantization),
            hidden_scale: 1.0 / (activation_scale as f32 * hidden_quantization),
            hidden_weights,
            hidden_bias,
            quantized_output: quantize(&output_weights, output_quantization),
            output_scale: 1.0 / (activation_scale as f32 * output_quantization),
            output_weights,
            output_bias,
        }
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
                    self.add(acc, perspective, self.count_feature(perspective, color, kind, old[index]), -1);
                    self.add(acc, perspective, self.count_feature(perspective, color, kind, new[index]), 1);
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
                    self.add(acc, perspective, self.mobility_feature(view, perspective, block * 8 + i), a as i16 - b as i16);
                }
            }
        }
    }

    fn column(&self, feature: usize) -> &[i16] {
        &self.feature_weights[feature * self.hidden..(feature + 1) * self.hidden]
    }

    fn add(&self, acc: &mut Accumulator, perspective: Color, feature: usize, sign: i16) {
        for (value, &weight) in acc.values[perspective as usize][..self.hidden].iter_mut().zip(self.column(feature)) {
            *value = value.wrapping_add(sign.wrapping_mul(weight));
        }
        acc.psqt[perspective as usize] += sign as f32 * self.psqt_weights[feature];
    }

    fn apply(&self, acc: &mut Accumulator, source: &[i16], perspective: Color, added: &[usize], removed: &[usize]) {
        let values = &mut acc.values[perspective as usize][..self.hidden];
        let source = &source[..self.hidden];
        match (added, removed) {
            (&[plus], &[minus]) => {
                for (((value, &old), &a), &r) in values.iter_mut().zip(source).zip(self.column(plus)).zip(self.column(minus)) {
                    *value = old.wrapping_add(a.wrapping_sub(r));
                }
            }
            (&[plus], &[minus1, minus2]) => {
                for ((((value, &old), &a), &r1), &r2) in values.iter_mut().zip(source).zip(self.column(plus)).zip(self.column(minus1)).zip(self.column(minus2)) {
                    *value = old.wrapping_add(a.wrapping_sub(r1).wrapping_sub(r2));
                }
            }
            _ => {
                values.copy_from_slice(source);
                for &feature in added {
                    for (value, &weight) in values.iter_mut().zip(self.column(feature)) {
                        *value = value.wrapping_add(weight);
                    }
                }
                for &feature in removed {
                    for (value, &weight) in values.iter_mut().zip(self.column(feature)) {
                        *value = value.wrapping_sub(weight);
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
                    self.add(acc, perspective, self.pawn_feature(view, perspective, color, kind, square), -1);
                }
                for square in Bitboard(after & !before) {
                    self.add(acc, perspective, self.pawn_feature(view, perspective, color, kind, square), 1);
                }
            }
        }
    }

    fn refresh_side(&self, acc: &mut Accumulator, board: &Board, perspective: Color) {
        let view = self.view(board, perspective);
        acc.values[perspective as usize][..self.hidden].copy_from_slice(&self.feature_bias);
        acc.psqt[perspective as usize] = 0.0;
        for (square, piece) in board {
            self.add(acc, perspective, piece_feature(view, perspective, piece, square), 1);
        }
        if self.pawn_features {
            self.update_pawns(acc, perspective, view, &[[0; 3]; 2], &pawn_structure(board));
        }
        if self.count_features {
            for color in [Color::White, Color::Black] {
                for kind in 0..5 {
                    let count = acc.counts[side(color) * 5 + kind];
                    self.add(acc, perspective, self.count_feature(perspective, color, kind, count), 1);
                }
            }
        }
        if self.mobility_features {
            let mobility = acc.mobility;
            self.update_mobility(acc, perspective, view, &[0; MOBILITY_FEATURES], &mobility);
        }
    }

    pub fn refresh(&self, board: &Board) -> Accumulator {
        let mut acc = Accumulator { values: [[0; MAX_HIDDEN]; 2], psqt: [0.0; 2], counts: [0; 10], mobility: [0; MOBILITY_FEATURES] };
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
        let mut changed = (before.white() ^ after.white()) | (before.black() ^ after.black());
        for role in Role::ALL {
            changed |= before.by_role(role) ^ after.by_role(role);
        }
        let pawns_changed = self.pawn_features && before.pawns() != after.pawns();
        let (old_pawns, new_pawns) = if pawns_changed { (pawn_structure(before), pawn_structure(after)) } else { ([[0; 3]; 2], [[0; 3]; 2]) };
        let new_counts = if self.count_features { piece_counts(after) } else { acc.counts };
        let new_mobility = if self.mobility_features { mobility(after) } else { acc.mobility };
        let mut next = Accumulator { values: [[0; MAX_HIDDEN]; 2], psqt: acc.psqt, counts: new_counts, mobility: new_mobility };

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
            self.apply(&mut next, &acc.values[perspective as usize], perspective, &added[..num_added], &removed[..num_removed]);
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
        let us = &acc.values[turn as usize][..self.hidden];
        let them = &acc.values[turn.other() as usize][..self.hidden];
        if self.layer2 == 0 && !self.screlu {
            let (own, other) = self.quantized_output.split_at(self.hidden);
            let sum = clipped_dot(us, own, self.activation_scale) + clipped_dot(them, other, self.activation_scale);
            return self.output_bias + sum as f32 * self.output_scale + psqt;
        }
        if !self.screlu {
            let mut output = self.output_bias;
            for (j, row) in self.quantized_hidden.chunks_exact(2 * self.hidden).enumerate() {
                let (own, other) = row.split_at(self.hidden);
                let sum = clipped_dot(us, own, self.activation_scale) + clipped_dot(them, other, self.activation_scale);
                output += (self.hidden_bias[j] + sum as f32 * self.hidden_scale).clamp(0.0, 1.0) * self.output_weights[j];
            }
            return output + psqt;
        }

        let mut inputs = [0.0; 2 * MAX_HIDDEN];
        activate_into(&mut inputs[..self.hidden], us, self.activation_scale, self.screlu);
        activate_into(&mut inputs[self.hidden..2 * self.hidden], them, self.activation_scale, self.screlu);
        let inputs = &inputs[..2 * self.hidden];
        if self.layer2 == 0 {
            return self.output_bias + dot(inputs, &self.output_weights) + psqt;
        }

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

    fn encode(magic: &[u8], header: &[u32], blocks: &[Vec<f32>]) -> Vec<u8> {
        let mut bytes = magic.to_vec();
        for value in header {
            bytes.extend(value.to_le_bytes());
        }
        for value in blocks.iter().flatten() {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    fn random_plain_nnue(screlu: bool) -> Nnue {
        let mut seed = 192837465;
        let hidden = 16;
        let blocks = [
            random_weights(&mut seed, PIECE_FEATURES * hidden),
            random_weights(&mut seed, hidden),
            random_weights(&mut seed, 2 * hidden),
            vec![0.1],
        ];
        Nnue::parse(&encode(b"NNU4", &[PIECE_FEATURES as u32, hidden as u32, u32::from(screlu)], &blocks))
    }

    fn random_nnue() -> Nnue {
        let mut seed = 987654321;
        let (num_buckets, hidden, layer2) = (8, 16, 4);
        let features = num_buckets * PIECE_FEATURES + PAWN_FEATURES + COUNT_FEATURES + MOBILITY_FEATURES;
        let mut header = vec![num_buckets as u32, 1, 1, hidden as u32, layer2 as u32, 1, 1];
        header.extend((0..64).map(|square| (square / 8).min(3) * 2 + u32::from(square % 8 % 4 >= 2)));
        let blocks = [
            random_weights(&mut seed, features * hidden),
            random_weights(&mut seed, hidden),
            random_weights(&mut seed, features),
            random_weights(&mut seed, layer2 * 2 * hidden),
            random_weights(&mut seed, layer2),
            random_weights(&mut seed, layer2),
            vec![0.1],
        ];
        Nnue::parse(&encode(b"NNU3", &header, &blocks))
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
                    assert_eq!(acc.values[side][..nnue.hidden], fresh.values[side][..nnue.hidden]);
                    assert!((acc.psqt[side] - fresh.psqt[side]).abs() < 1e-3);
                }
                assert!((nnue.evaluate(&acc, pos.turn()) - nnue.evaluate(&fresh, pos.turn())).abs() < 1e-3);
            }
        }

        println!("checked: {castles} castles, {en_passants} en passant, {promotions} promotions, {king_moves} king moves");
        assert!(castles > 0 && en_passants > 0 && promotions > 0 && king_moves > 0);
    }

    fn float_evaluate(nnue: &Nnue, acc: &Accumulator, turn: Color) -> f32 {
        let psqt = (acc.psqt[turn as usize] - acc.psqt[turn.other() as usize]) / 2.0;
        let mut inputs = vec![0.0; 2 * nnue.hidden];
        activate_into(&mut inputs[..nnue.hidden], &acc.values[turn as usize][..nnue.hidden], nnue.activation_scale, false);
        activate_into(&mut inputs[nnue.hidden..], &acc.values[turn.other() as usize][..nnue.hidden], nnue.activation_scale, false);
        let mut output = nnue.output_bias;
        for (j, row) in nnue.hidden_weights.chunks_exact(2 * nnue.hidden).enumerate() {
            output += (nnue.hidden_bias[j] + dot(&inputs, row)).clamp(0.0, 1.0) * nnue.output_weights[j];
        }
        output + psqt
    }

    #[test]
    fn quantized_layer2_matches_float() {
        let nnue = random_nnue();
        let mut seed = 2468;
        let mut pos = Chess::default();
        let mut largest = 0.0f32;
        for _ in 0..200 {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            pos.play_unchecked(moves[(random(&mut seed) % moves.len() as u64) as usize]);
            let acc = nnue.refresh(pos.board());
            largest = largest.max((nnue.evaluate(&acc, pos.turn()) - float_evaluate(&nnue, &acc, pos.turn())).abs());
        }
        println!("largest difference between int16 and float layer2: {largest}");
        assert!(largest < 1e-3);
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
        check_updates(&random_plain_nnue(true));
    }

    #[test]
    fn update_matches_refresh_with_clipped_relu() {
        check_updates(&random_plain_nnue(false));
    }
}
