//! Port of `oqmc/lookup.h` — randomised table lookups. A single pre-computed
//! table (PMJ samples, blue-noise ranks) can be re-used across domains by
//! shuffling the index and XOR-scrambling the value (random digit scramble,
//! Kollig & Keller).

use crate::permute::shuffle;
use crate::rotate::rotate_bytes;

/// Random digit scramble: XOR the value with a random number (fast, structure-
/// preserving). The seed must be constant for a given sequence.
#[inline]
pub const fn random_digit_scramble(value: u32, hash: u32) -> u32 {
    value ^ hash
}

/// Compute a randomised value from a pre-computed table.
///
/// `TABLE` is the row width of the table, `DEPTH` (1..=4, `<= TABLE`) the output
/// count. The index is shuffled progressively; an index beyond the table length
/// wraps (mask to 16 bits) and reuses samples.
#[inline]
pub fn shuffled_scrambled_lookup<const TABLE: usize, const DEPTH: usize>(
    index: u32,
    hash: u32,
    table: &[[u32; TABLE]],
) -> [u32; DEPTH] {
    const { assert!(TABLE >= DEPTH, "Table width must be >= depth") };
    const {
        assert!(
            DEPTH >= 1 && DEPTH <= 4,
            "Pattern depth must be within [1, 4]"
        )
    };

    let index = shuffle(index, hash);

    let mut sample = [0u32; DEPTH];
    let mut i = 0;
    while i < DEPTH {
        let row = &table[(index & 0xffff) as usize];
        sample[i] = random_digit_scramble(row[i], rotate_bytes(hash, i as i32));
        i += 1;
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scramble_is_xor() {
        assert_eq!(random_digit_scramble(0xf0f0_f0f0, 0x0f0f_0f0f), u32::MAX);
        assert_eq!(random_digit_scramble(7, 0), 7);
        assert_eq!(random_digit_scramble(random_digit_scramble(9, 5), 5), 9);
    }

    #[test]
    fn lookup_indexes_the_shuffled_row_and_scrambles_columns() {
        let table: Vec<[u32; 4]> = (0..65536u32).map(|i| [i, i + 1, i + 2, i + 3]).collect();
        for hash in [0u32, 1, 0xdead_beef] {
            for index in [0u32, 5, 65535, 70000] {
                let row = (shuffle(index, hash) & 0xffff) as usize;
                let out = shuffled_scrambled_lookup::<4, 4>(index, hash, &table);
                for (k, &o) in out.iter().enumerate() {
                    assert_eq!(o, table[row][k] ^ rotate_bytes(hash, k as i32));
                }
            }
        }
    }

    #[test]
    fn lookup_depth_can_be_smaller_than_the_table_width() {
        let table: Vec<[u32; 4]> = (0..65536u32).map(|i| [i; 4]).collect();
        let full = shuffled_scrambled_lookup::<4, 4>(42, 3, &table);
        assert_eq!(
            &full[..1],
            &shuffled_scrambled_lookup::<4, 1>(42, 3, &table)[..]
        );
        assert_eq!(
            &full[..2],
            &shuffled_scrambled_lookup::<4, 2>(42, 3, &table)[..]
        );
        assert_eq!(
            &full[..3],
            &shuffled_scrambled_lookup::<4, 3>(42, 3, &table)[..]
        );
    }

    #[test]
    fn lookup_covers_every_row_once_over_a_full_index_range() {
        let table: Vec<[u32; 1]> = (0..65536u32).map(|i| [i]).collect();
        let hash = 0x1234_5678;
        let mut seen = vec![false; 65536];
        for index in 0..65536u32 {
            let row = (shuffled_scrambled_lookup::<1, 1>(index, hash, &table)[0] ^ hash) as usize;
            assert!(!seen[row]);
            seen[row] = true;
        }
    }
}
