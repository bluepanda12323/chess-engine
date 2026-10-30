use std::sync::atomic::{AtomicU64, Ordering};

const BUCKET_SIZE: usize = 4;
const AGE_MASK: u64 = 63;

#[derive(Clone, Copy, PartialEq)]
pub enum Bound {
    Exact = 1,
    Lower = 2,
    Upper = 3,
}

#[derive(Clone, Copy)]
pub struct Entry {
    pub depth: u32,
    pub score: f32,
    pub bound: Bound,
    pub best: u16,
}

pub struct Table {
    slots: Vec<[AtomicU64; 2]>,
    generation: AtomicU64,
}

fn unpack(data: u64) -> Entry {
    let bound = match (data >> 40) & 3 {
        1 => Bound::Exact,
        2 => Bound::Lower,
        _ => Bound::Upper,
    };
    Entry { depth: ((data >> 32) & 0xff) as u32, score: f32::from_bits(data as u32), bound, best: (data >> 48) as u16 }
}

impl Table {
    pub fn new(megabytes: usize) -> Table {
        let count = ((megabytes.max(1) << 20) / size_of::<[AtomicU64; 2]>()).max(BUCKET_SIZE) / BUCKET_SIZE * BUCKET_SIZE;
        Table { slots: (0..count).map(|_| [AtomicU64::new(0), AtomicU64::new(0)]).collect(), generation: AtomicU64::new(0) }
    }

    pub fn clear(&self) {
        for [check, data] in &self.slots {
            check.store(0, Ordering::Relaxed);
            data.store(0, Ordering::Relaxed);
        }
    }

    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    fn bucket(&self, key: u64) -> &[[AtomicU64; 2]] {
        let buckets = self.slots.len() / BUCKET_SIZE;
        let start = ((key as u128 * buckets as u128) >> 64) as usize * BUCKET_SIZE;
        &self.slots[start..start + BUCKET_SIZE]
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        self.bucket(key).iter().find_map(|[check, data]| {
            let data = data.load(Ordering::Relaxed);
            (data != 0 && check.load(Ordering::Relaxed) ^ data == key).then(|| unpack(data))
        })
    }

    pub fn store(&self, key: u64, mut entry: Entry) {
        let generation = self.generation.load(Ordering::Relaxed) & AGE_MASK;
        let bucket = self.bucket(key);
        let mut target = 0;
        let mut target_worth = i64::MAX;
        for (i, [check, data]) in bucket.iter().enumerate() {
            let data = data.load(Ordering::Relaxed);
            if data == 0 {
                target = i;
                break;
            }
            if check.load(Ordering::Relaxed) ^ data == key {
                let old = unpack(data);
                let old_age = (data >> 42) & AGE_MASK;
                if entry.bound != Bound::Exact && entry.depth + 4 < old.depth && old_age == generation {
                    return;
                }
                if entry.best == 0 {
                    entry.best = old.best;
                }
                target = i;
                break;
            }
            let age = (generation.wrapping_sub((data >> 42) & AGE_MASK)) & AGE_MASK;
            let worth = ((data >> 32) & 0xff) as i64 - 8 * age as i64;
            if worth < target_worth {
                target_worth = worth;
                target = i;
            }
        }
        let data = u64::from(entry.score.to_bits()) | u64::from(entry.depth.min(255)) << 32 | (entry.bound as u64) << 40 | generation << 42 | u64::from(entry.best) << 48;
        let [check, stored] = &bucket[target];
        check.store(key ^ data, Ordering::Relaxed);
        stored.store(data, Ordering::Relaxed);
    }
}
