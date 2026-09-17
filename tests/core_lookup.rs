//! Property tests for `openqmc::lookup` (oqmc/lookup.h) — randomised table
//! lookups.

mod common;

use openqmc::lookup::{random_digit_scramble, shuffled_scrambled_lookup};
use openqmc::permute::shuffle;
use openqmc::rotate::rotate_bytes;

const HASHES: [u32; 5] = [0, 1, 0x07bb_2fe2, 0xdead_beef, u32::MAX];

/// A 2^16-row table whose row `i` is `[i, i << 16, !i, i * 3]`, so every value
/// identifies its row.
fn marker_table() -> Vec<[u32; 4]> {
    (0..65536u32)
        .map(|i| [i, i << 16, !i, i.wrapping_mul(3)])
        .collect()
}

#[test]
fn random_digit_scramble_is_xor() {
    for v in common::random_u32s(1, 4096) {
        for h in HASHES {
            assert_eq!(random_digit_scramble(v, h), v ^ h);
        }
    }
}

#[test]
fn random_digit_scramble_is_an_involution() {
    for v in common::random_u32s(2, 4096) {
        for h in HASHES {
            assert_eq!(random_digit_scramble(random_digit_scramble(v, h), h), v);
        }
    }
}

#[test]
fn random_digit_scramble_with_zero_hash_is_identity() {
    for v in common::EDGE_U32 {
        assert_eq!(random_digit_scramble(v, 0), v);
    }
}

#[test]
fn random_digit_scramble_is_a_bijection() {
    for h in HASHES {
        common::assert_low_bits_bijection(12, &format!("scramble {h:#x}"), |v| {
            random_digit_scramble(v, h)
        });
    }
}

#[test]
fn random_digit_scramble_preserves_strata_of_a_stratified_set() {
    // XOR with a constant permutes elementary intervals, so a 1D-stratified
    // set stays stratified.
    for h in HASHES {
        for m in [1u32, 4, 8] {
            let vals: Vec<u32> = (0..(1u32 << m))
                .map(|i| random_digit_scramble(i << (32 - m), h))
                .collect();
            common::assert_1d_stratified(&vals, m, &format!("hash {h:#x} m={m}"));
        }
    }
}

#[test]
fn lookup_reads_the_shuffled_row_and_scrambles_per_column() {
    let table = marker_table();
    for h in HASHES {
        for i in [0u32, 1, 2, 1000, 65535, 65536, 70000, u32::MAX] {
            let row = (shuffle(i, h) & 0xffff) as usize;
            let out = shuffled_scrambled_lookup::<4, 4>(i, h, &table);
            for (k, &o) in out.iter().enumerate() {
                assert_eq!(
                    o,
                    table[row][k] ^ rotate_bytes(h, k as i32),
                    "hash {h:#x} i {i} col {k}"
                );
            }
        }
    }
}

#[test]
fn lookup_depth_prefixes_are_consistent() {
    let table = marker_table();
    for h in HASHES {
        for i in [0u32, 7, 65535, 1 << 20] {
            let four = shuffled_scrambled_lookup::<4, 4>(i, h, &table);
            assert_eq!(
                &four[..3],
                &shuffled_scrambled_lookup::<4, 3>(i, h, &table)[..]
            );
            assert_eq!(
                &four[..2],
                &shuffled_scrambled_lookup::<4, 2>(i, h, &table)[..]
            );
            assert_eq!(
                &four[..1],
                &shuffled_scrambled_lookup::<4, 1>(i, h, &table)[..]
            );
        }
    }
}

#[test]
fn lookup_visits_every_row_exactly_once_over_16_bits() {
    let table = marker_table();
    for h in [0u32, 0x07bb_2fe2] {
        let mut seen = vec![false; 65536];
        for i in 0..65536u32 {
            let out = shuffled_scrambled_lookup::<4, 1>(i, h, &table);
            let row = (out[0] ^ h) as usize;
            assert!(!seen[row], "hash {h:#x}: row {row} twice");
            seen[row] = true;
        }
    }
}

#[test]
fn lookup_wraps_indices_beyond_the_table() {
    // The shuffled index is masked to 16 bits, so each aligned 2^16 block of
    // indices reuses every row exactly once.
    let table = marker_table();
    let h = 0xdead_beef;
    let mut rows: Vec<u32> = (65536..131072u32)
        .map(|i| shuffled_scrambled_lookup::<4, 1>(i, h, &table)[0] ^ h)
        .collect();
    rows.sort_unstable();
    assert!(rows.iter().enumerate().all(|(k, &r)| r == k as u32));
}

#[test]
fn lookup_beyond_the_table_is_not_a_plain_repeat() {
    let table = marker_table();
    let h = 0xdead_beef;
    let differs = (0..64u32).any(|i| {
        shuffled_scrambled_lookup::<4, 4>(i, h, &table)
            != shuffled_scrambled_lookup::<4, 4>(i + 65536, h, &table)
    });
    assert!(differs);
}

#[test]
fn lookup_with_zero_hash_is_shuffle_only() {
    let table = marker_table();
    for i in 0..1024u32 {
        let out = shuffled_scrambled_lookup::<4, 4>(i, 0, &table);
        let row = (shuffle(i, 0) & 0xffff) as usize;
        assert_eq!(out, table[row]);
    }
}

#[test]
fn lookup_columns_use_distinct_scramble_keys() {
    let table = vec![[0u32; 4]; 65536];
    for h in [0x07bb_2fe2u32, 0xdead_beef, 0x0102_0304] {
        let out = shuffled_scrambled_lookup::<4, 4>(0, h, &table);
        assert_eq!(
            out,
            [
                h,
                rotate_bytes(h, 1),
                rotate_bytes(h, 2),
                rotate_bytes(h, 3)
            ]
        );
        common::assert_all_distinct(&out, &format!("keys for {h:#x}"));
    }
}

#[test]
fn lookup_works_on_two_column_tables() {
    let table: Vec<[u32; 2]> = (0..65536u32).map(|i| [i, !i]).collect();
    for h in HASHES {
        for i in [0u32, 5, 65535, 99999] {
            let row = (shuffle(i, h) & 0xffff) as usize;
            let out = shuffled_scrambled_lookup::<2, 2>(i, h, &table);
            assert_eq!(out, [row as u32 ^ h, !(row as u32) ^ rotate_bytes(h, 1)]);
            let one = shuffled_scrambled_lookup::<2, 1>(i, h, &table);
            assert_eq!(one[0], out[0]);
        }
    }
}

#[test]
fn lookup_works_on_wide_tables() {
    // TABLE may exceed DEPTH; extra columns are ignored.
    let table: Vec<[u32; 6]> = (0..65536u32)
        .map(|i| [i, i + 1, i + 2, i + 3, 0xaaaa, 0xbbbb])
        .collect();
    let h = 0x1234_5678;
    for i in 0..256u32 {
        let row = shuffle(i, h) & 0xffff;
        let out = shuffled_scrambled_lookup::<6, 4>(i, h, &table);
        assert_eq!(out[0], row ^ h);
        assert_eq!(out[3], (row + 3) ^ rotate_bytes(h, 3));
    }
}

#[test]
fn lookup_is_deterministic() {
    let table = marker_table();
    for h in HASHES {
        for i in common::random_u32s(3, 256) {
            assert_eq!(
                shuffled_scrambled_lookup::<4, 4>(i, h, &table),
                shuffled_scrambled_lookup::<4, 4>(i, h, &table)
            );
        }
    }
}

#[test]
fn lookup_preserves_stratification_of_a_stratified_table() {
    // A table whose column 0 is the Van der Corput sequence stays stratified
    // over aligned index blocks after shuffling and scrambling.
    let table: Vec<[u32; 1]> = (0..65536u32).map(|i| [i.reverse_bits()]).collect();
    for h in HASHES {
        for m in [1u32, 4, 8, 12] {
            for block in [0u32, 1, 3] {
                let start = block << m;
                let vals: Vec<u32> = (start..start + (1 << m))
                    .map(|i| shuffled_scrambled_lookup::<1, 1>(i, h, &table)[0])
                    .collect();
                common::assert_1d_stratified(&vals, m, &format!("hash {h:#x} m={m} block={block}"));
            }
        }
    }
}

#[test]
fn lookup_distinct_hashes_give_distinct_outputs() {
    let table = marker_table();
    let mut outs: Vec<[u32; 4]> = Vec::new();
    for p in common::PRIMES {
        outs.push(shuffled_scrambled_lookup::<4, 4>(
            0,
            openqmc::pcg::hash(p),
            &table,
        ));
    }
    common::assert_all_distinct(&outs, "lookup outputs across hashes");
}

#[test]
fn function_is_usable_in_const_context() {
    const V: u32 = random_digit_scramble(0xf0f0, 0x0ff0);
    assert_eq!(V, 0xff00);
}
