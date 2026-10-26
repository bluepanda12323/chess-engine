use shakmaty::{Chess, Color, Position};

pub const INPUT_SIZE: usize = 448;

pub fn encode(pos: &Chess) -> [f32; INPUT_SIZE] {
    let mut input = [0.0; INPUT_SIZE];

    for (square, piece) in pos.board() {
        let layer = piece.role as usize - 1;
        let row = 7 - square.rank().to_usize();
        let col = square.file().to_usize();
        input[layer * 64 + row * 8 + col] = if piece.color == Color::White { 1.0 } else { -1.0 };
    }

    if pos.turn() == Color::White {
        for value in &mut input[6 * 64..] {
            *value = 1.0;
        }
    }

    input
}
