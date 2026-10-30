const MOVE_OVERHEAD: u64 = 100;
const MOVES_LEFT: u64 = 30;

pub fn limits(time_left: u64, increment: u64) -> (u64, u64) {
    let available = time_left.saturating_sub(MOVE_OVERHEAD);
    let budget = available / MOVES_LEFT + increment * 3 / 4;
    (budget / 2, (2 * budget).min(available / 4))
}
