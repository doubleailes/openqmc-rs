//! Property tests for `openqmc::owen` (oqmc/owen.h) — Owen-scrambled Sobol.

mod common;

use openqmc::owen::{scramble_and_reverse, shuffled_scrambled_sobol, sobol_reversed_index};
use openqmc::pcg;
use openqmc::permute::laine_karras_permutation;
use openqmc::reverse::{reverse_bits16, reverse_bits32};

const SEEDS: [u32; 6] = [0, 1, 0x07bb_2fe2, 0xdead_beef, 0x8000_0000, 0xffff_ffff];

/// The plain (unscrambled) 16-bit Sobol value at `index` in `dim`, MSB first,
/// widened to 32 bits.
fn plain_sobol(index: u16, dim: usize) -> u32 {
    (reverse_bits16(sobol_reversed_index(reverse_bits16(index), dim)) as u32) << 16
}

#[test]
fn reversed_index_of_zero_is_zero() {
    for dim in 0..4 {
        assert_eq!(sobol_reversed_index(0, dim), 0);
    }
}

#[test]
fn reversed_index_dim0_is_bit_reversal_exhaustively() {
    for i in 0..=u16::MAX {
        assert_eq!(sobol_reversed_index(i, 0), reverse_bits16(i));
    }
}

#[test]
fn reversed_index_is_xor_linear_in_every_dimension() {
    let values = common::random_u32s(1, 4096);
    for dim in 0..4 {
        for w in values.windows(2) {
            let a = w[0] as u16;
            let b = w[1] as u16;
            assert_eq!(
                sobol_reversed_index(a ^ b, dim),
                sobol_reversed_index(a, dim) ^ sobol_reversed_index(b, dim),
                "dim {dim}"
            );
        }
    }
}

#[test]
fn reversed_index_is_the_xor_of_its_basis_columns() {
    for dim in 0..4 {
        let columns: Vec<u16> = (0..16).map(|k| sobol_reversed_index(1 << k, dim)).collect();
        for i in common::random_u32s(2, 2048).into_iter().map(|v| v as u16) {
            let mut expected = 0u16;
            for (k, &col) in columns.iter().enumerate() {
                if i & (1 << k) != 0 {
                    expected ^= col;
                }
            }
            assert_eq!(
                sobol_reversed_index(i, dim),
                expected,
                "dim {dim} index {i}"
            );
        }
    }
}

#[test]
fn direction_columns_have_descending_leading_bits() {
    // Column k has its leading one at bit 15-k, making the generator matrix
    // triangular (hence invertible) in every dimension.
    for dim in 0..4 {
        for k in 0..16u32 {
            let col = sobol_reversed_index(1 << k, dim);
            assert_ne!(col, 0, "dim {dim} col {k}");
            assert_eq!(
                15 - col.leading_zeros(),
                15 - k,
                "dim {dim} col {k}: {col:#018b}"
            );
        }
    }
}

#[test]
fn reversed_index_is_a_bijection_in_every_dimension() {
    for dim in 0..4 {
        let mut seen = vec![false; 65536];
        for i in 0..=u16::MAX {
            let v = sobol_reversed_index(i, dim) as usize;
            assert!(!seen[v], "dim {dim}: value {v} hit twice");
            seen[v] = true;
        }
    }
}

#[test]
fn dimensions_are_pairwise_distinct_maps() {
    for a in 0..4 {
        for b in (a + 1)..4 {
            let differs =
                (1..1024u16).any(|i| sobol_reversed_index(i, a) != sobol_reversed_index(i, b));
            assert!(differs, "dims {a} and {b} coincide");
        }
    }
}

#[test]
fn plain_sobol_every_dimension_is_a_01_sequence() {
    for dim in 0..4 {
        for m in 0..=16u32 {
            let vals: Vec<u32> = (0..(1u32 << m))
                .map(|i| plain_sobol(i as u16, dim))
                .collect();
            common::assert_1d_stratified(&vals, m, &format!("plain sobol dim {dim} m={m}"));
        }
    }
}

#[test]
fn plain_sobol_aligned_blocks_are_stratified() {
    for dim in 0..4 {
        for m in 1..=8u32 {
            for block in [1u32, 2, 3, 100, 255] {
                let start = block << m;
                if start + (1 << m) > 65536 {
                    continue;
                }
                let vals: Vec<u32> = (start..start + (1 << m))
                    .map(|i| plain_sobol(i as u16, dim))
                    .collect();
                common::assert_1d_stratified(
                    &vals,
                    m,
                    &format!("plain sobol dim {dim} block {block}"),
                );
            }
        }
    }
}

#[test]
fn plain_sobol_first_pair_is_a_02_sequence() {
    for m in 0..=16u32 {
        let pts: Vec<[u32; 2]> = (0..(1u32 << m))
            .map(|i| [plain_sobol(i as u16, 0), plain_sobol(i as u16, 1)])
            .collect();
        common::assert_02_net(&pts, m, &format!("plain sobol (0,1) m={m}"));
    }
}

#[test]
fn plain_sobol_other_pairs_are_not_02_nets() {
    // Regression guard on the direction tables: only the (0,1) projection has
    // t = 0. Each other pair fails for some small m.
    let is_net = |a: usize, b: usize, m: u32| -> bool {
        let n = 1usize << m;
        for i in 0..=m {
            let xb = i;
            let yb = m - i;
            let mut seen = vec![false; n];
            for idx in 0..n as u32 {
                let px = plain_sobol(idx as u16, a);
                let py = plain_sobol(idx as u16, b);
                let x = if xb == 0 {
                    0
                } else {
                    (px >> (32 - xb)) as usize
                };
                let y = if yb == 0 {
                    0
                } else {
                    (py >> (32 - yb)) as usize
                };
                let c = (y << xb) | x;
                if seen[c] {
                    return false;
                }
                seen[c] = true;
            }
        }
        true
    };
    assert!((0..=10).all(|m| is_net(0, 1, m)));
    for (a, b) in [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        assert!(
            (1..=10).any(|m| !is_net(a, b, m)),
            "pair ({a},{b}) unexpectedly a (0,2)-net"
        );
    }
}

#[test]
fn scramble_and_reverse_is_reverse_of_lk() {
    for v in common::random_u32s(3, 4096) {
        for seed in SEEDS {
            assert_eq!(
                scramble_and_reverse(v, seed),
                reverse_bits32(laine_karras_permutation(v, seed))
            );
        }
    }
}

#[test]
fn scramble_and_reverse_is_a_bijection_on_low_bits() {
    for seed in SEEDS {
        // Output is reversed, so injectivity on the low k input bits shows up
        // in the high k output bits.
        let bits = 12u32;
        let mut seen = vec![false; 1 << bits];
        for i in 0..(1u32 << bits) {
            let out = (scramble_and_reverse(i, seed) >> (32 - bits)) as usize;
            assert!(!seen[out], "seed {seed:#x}");
            seen[out] = true;
        }
    }
}

#[test]
fn scramble_and_reverse_preserves_16_bit_strata_of_reversed_values() {
    // Feeding the 16-bit reversed Sobol values through the scramble keeps each
    // top-16-bit stratum occupied exactly once.
    for seed in SEEDS {
        let vals: Vec<u32> = (0..=u16::MAX)
            .map(|v| scramble_and_reverse(v as u32, seed))
            .collect();
        common::assert_1d_stratified(&vals, 16, &format!("scramble seed {seed:#x}"));
    }
}

#[test]
fn shuffled_depth_prefixes_are_consistent() {
    for seed in SEEDS {
        for i in [0u32, 1, 2, 3, 255, 4096, 65535, 65536, 1 << 20, u32::MAX] {
            let four = shuffled_scrambled_sobol::<4>(i, seed);
            let three = shuffled_scrambled_sobol::<3>(i, seed);
            let two = shuffled_scrambled_sobol::<2>(i, seed);
            let one = shuffled_scrambled_sobol::<1>(i, seed);
            assert_eq!(&four[..3], &three[..]);
            assert_eq!(&four[..2], &two[..]);
            assert_eq!(&four[..1], &one[..]);
        }
    }
}

#[test]
fn shuffled_is_deterministic() {
    for seed in SEEDS {
        for i in common::random_u32s(4, 256) {
            assert_eq!(
                shuffled_scrambled_sobol::<4>(i, seed),
                shuffled_scrambled_sobol::<4>(i, seed)
            );
        }
    }
}

#[test]
fn shuffled_every_dimension_is_stratified_over_prefixes() {
    for seed in SEEDS {
        for m in 0..=12u32 {
            let block: Vec<[u32; 4]> = (0..(1u32 << m))
                .map(|i| shuffled_scrambled_sobol::<4>(i, seed))
                .collect();
            for dim in 0..4 {
                let vals: Vec<u32> = block.iter().map(|s| s[dim]).collect();
                common::assert_1d_stratified(&vals, m, &format!("seed {seed:#x} dim {dim} m={m}"));
            }
        }
    }
}

#[test]
fn shuffled_every_dimension_is_stratified_over_the_full_16_bit_range() {
    for seed in [0u32, 0x07bb_2fe2] {
        let block: Vec<[u32; 4]> = (0..65536u32)
            .map(|i| shuffled_scrambled_sobol::<4>(i, seed))
            .collect();
        for dim in 0..4 {
            let vals: Vec<u32> = block.iter().map(|s| s[dim]).collect();
            common::assert_1d_stratified(&vals, 16, &format!("seed {seed:#x} dim {dim} full"));
        }
    }
}

#[test]
fn shuffled_every_dimension_is_stratified_over_aligned_blocks() {
    for seed in SEEDS {
        for m in 1..=8u32 {
            for block in [1u32, 7, 100, 255, 1000] {
                let start = block << m;
                let rows: Vec<[u32; 4]> = (start..start + (1 << m))
                    .map(|i| shuffled_scrambled_sobol::<4>(i, seed))
                    .collect();
                for dim in 0..4 {
                    let vals: Vec<u32> = rows.iter().map(|s| s[dim]).collect();
                    common::assert_1d_stratified(
                        &vals,
                        m,
                        &format!("seed {seed:#x} dim {dim} block {block} m={m}"),
                    );
                }
            }
        }
    }
}

#[test]
fn shuffled_first_pair_is_a_02_sequence_over_prefixes() {
    for seed in SEEDS {
        for m in 0..=12u32 {
            let pts: Vec<[u32; 2]> = (0..(1u32 << m))
                .map(|i| shuffled_scrambled_sobol::<2>(i, seed))
                .collect();
            common::assert_02_net(&pts, m, &format!("seed {seed:#x} m={m}"));
        }
    }
}

#[test]
fn shuffled_first_pair_is_a_02_net_at_full_16_bits() {
    let pts: Vec<[u32; 2]> = (0..65536u32)
        .map(|i| shuffled_scrambled_sobol::<2>(i, 0x07bb_2fe2))
        .collect();
    common::assert_02_net(&pts, 16, "full 16-bit (0,1) net");
}

#[test]
fn shuffled_first_pair_is_a_02_net_over_aligned_blocks() {
    for seed in SEEDS {
        for m in 1..=8u32 {
            for block in [1u32, 3, 64, 511] {
                let start = block << m;
                let pts: Vec<[u32; 2]> = (start..start + (1 << m))
                    .map(|i| shuffled_scrambled_sobol::<2>(i, seed))
                    .collect();
                common::assert_02_net(&pts, m, &format!("seed {seed:#x} block {block} m={m}"));
            }
        }
    }
}

#[test]
fn shuffled_values_repeat_as_a_set_beyond_16_bits() {
    // Indices >= 2^16 reuse the same 2^16 sample values, in a different order.
    let seed = 0xdead_beef;
    let mut base: Vec<[u32; 4]> = (0..65536u32)
        .map(|i| shuffled_scrambled_sobol::<4>(i, seed))
        .collect();
    base.sort_unstable();
    for block in [1u32, 2, 0xffff] {
        let start = block << 16;
        let mut next: Vec<[u32; 4]> = (0..65536u32)
            .map(|j| shuffled_scrambled_sobol::<4>(start.wrapping_add(j), seed))
            .collect();
        next.sort_unstable();
        assert_eq!(base, next, "block {block}");
    }
}

#[test]
fn shuffled_values_are_not_in_the_same_order_beyond_16_bits() {
    let seed = 0xdead_beef;
    let differs = (0..256u32).any(|i| {
        shuffled_scrambled_sobol::<4>(i, seed) != shuffled_scrambled_sobol::<4>(i + 65536, seed)
    });
    assert!(differs);
}

#[test]
fn shuffled_distinct_seeds_give_distinct_sequences() {
    let mut firsts: Vec<[u32; 4]> = Vec::new();
    for p in common::PRIMES {
        firsts.push(shuffled_scrambled_sobol::<4>(0, pcg::hash(p)));
    }
    common::assert_all_distinct(&firsts, "first samples across seeds");
}

#[test]
fn shuffled_dimensions_are_pairwise_distinct_within_a_sample() {
    let mut equal_pairs = 0;
    for i in 0..4096u32 {
        let s = shuffled_scrambled_sobol::<4>(i, 0x07bb_2fe2);
        for a in 0..4 {
            for b in (a + 1)..4 {
                if s[a] == s[b] {
                    equal_pairs += 1;
                }
            }
        }
    }
    assert!(equal_pairs <= 1, "{equal_pairs} coincident dimension pairs");
}

#[test]
fn shuffled_output_covers_low_bits_too() {
    // The Owen scramble fills all 32 output bits, not just the 16 Sobol bits.
    let mut low_or = 0u32;
    let mut low_and = u32::MAX;
    for i in 0..4096u32 {
        let s = shuffled_scrambled_sobol::<1>(i, 0x07bb_2fe2)[0];
        low_or |= s & 0xffff;
        low_and &= s & 0xffff;
    }
    assert_eq!(low_or, 0xffff);
    assert_eq!(low_and, 0);
}

#[test]
fn shuffled_means_are_close_to_half() {
    let n = 1u32 << 12;
    for seed in SEEDS {
        let mut sums = [0.0f64; 4];
        for i in 0..n {
            let s = shuffled_scrambled_sobol::<4>(i, seed);
            for d in 0..4 {
                sums[d] += s[d] as f64 / 4294967296.0;
            }
        }
        for (d, sum) in sums.iter().enumerate() {
            let mean = sum / n as f64;
            assert!(
                (mean - 0.5).abs() < 1e-3,
                "seed {seed:#x} dim {d}: mean {mean}"
            );
        }
    }
}

#[test]
fn shuffled_integrates_a_smooth_function_far_better_than_random() {
    // QMC error on a smooth 2D integrand should be well below the MC error at
    // the same sample count.
    let f = |x: f64, y: f64| (x * x + y * y) - 2.0 / 3.0;
    let n = 1u32 << 12;
    let mut qmc = 0.0f64;
    for i in 0..n {
        let s = shuffled_scrambled_sobol::<2>(i, 0x07bb_2fe2);
        qmc += f(s[0] as f64 / 4294967296.0, s[1] as f64 / 4294967296.0);
    }
    let mut mc = 0.0f64;
    let mut r = common::rng(11);
    for _ in 0..n {
        mc += f(r.next_f32() as f64, r.next_f32() as f64);
    }
    let qmc_err = (qmc / n as f64).abs();
    let mc_err = (mc / n as f64).abs();
    assert!(qmc_err < 1e-3, "qmc err {qmc_err}");
    assert!(
        qmc_err < mc_err,
        "qmc {qmc_err} not better than mc {mc_err}"
    );
}
