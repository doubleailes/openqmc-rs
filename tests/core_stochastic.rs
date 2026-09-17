//! Property tests for `openqmc::stochastic` (oqmc/stochastic.h) — the
//! progressive multi-jittered (0,2) table construction.

mod common;

use openqmc::stochastic::stochastic_pmj_init;

const FULL: usize = 1 << 16;

fn build() -> Vec<[u32; 4]> {
    let mut t = vec![[0u32; 4]; FULL];
    stochastic_pmj_init(FULL, &mut t);
    t
}

#[test]
fn init_is_deterministic_across_calls() {
    assert_eq!(build(), build());
}

#[test]
#[should_panic(expected = "PMJ init requires a full 2^16 table")]
fn init_rejects_smaller_sample_counts() {
    let mut t = vec![[0u32; 4]; 1024];
    stochastic_pmj_init(1024, &mut t);
}

#[test]
#[should_panic(expected = "PMJ init requires a full 2^16 table")]
fn init_rejects_larger_sample_counts() {
    let mut t = vec![[0u32; 4]; FULL * 2];
    stochastic_pmj_init(FULL * 2, &mut t);
}

#[test]
#[should_panic]
fn init_rejects_a_too_short_table() {
    let mut t = vec![[0u32; 4]; FULL - 1];
    stochastic_pmj_init(FULL, &mut t);
}

#[test]
fn init_only_touches_the_first_2_16_rows_of_a_longer_table() {
    let sentinel = [0xdead_beef; 4];
    let mut t = vec![sentinel; FULL + 100];
    stochastic_pmj_init(FULL, &mut t);
    assert_eq!(&t[..FULL], &build()[..]);
    assert!(t[FULL..].iter().all(|r| *r == sentinel));
}

#[test]
fn first_pair_is_a_02_sequence_over_every_prefix() {
    let t = build();
    for m in 0..=16u32 {
        let pts: Vec<[u32; 2]> = t[..1 << m].iter().map(|r| [r[0], r[1]]).collect();
        common::assert_02_net(&pts, m, &format!("pmj (0,1) m={m}"));
    }
}

#[test]
fn second_pair_is_a_02_sequence_over_every_prefix() {
    let t = build();
    for m in 0..=16u32 {
        let pts: Vec<[u32; 2]> = t[..1 << m].iter().map(|r| [r[2], r[3]]).collect();
        common::assert_02_net(&pts, m, &format!("pmj (2,3) m={m}"));
    }
}

#[test]
fn first_pair_is_a_02_net_over_aligned_blocks() {
    let t = build();
    for m in 1..=12u32 {
        for block in [1usize, 3, 17, 255] {
            let start = block << m;
            if start + (1 << m) > FULL {
                continue;
            }
            let pts: Vec<[u32; 2]> = t[start..start + (1 << m)]
                .iter()
                .map(|r| [r[0], r[1]])
                .collect();
            common::assert_02_net(&pts, m, &format!("pmj (0,1) block {block} m={m}"));
        }
    }
}

#[test]
fn second_pair_is_a_02_net_over_aligned_blocks() {
    let t = build();
    for m in 1..=12u32 {
        for block in [1usize, 2, 9, 100] {
            let start = block << m;
            if start + (1 << m) > FULL {
                continue;
            }
            let pts: Vec<[u32; 2]> = t[start..start + (1 << m)]
                .iter()
                .map(|r| [r[2], r[3]])
                .collect();
            common::assert_02_net(&pts, m, &format!("pmj (2,3) block {block} m={m}"));
        }
    }
}

#[test]
fn every_dimension_is_a_permutation_of_the_16_bit_strata() {
    let t = build();
    for d in 0..4 {
        let vals: Vec<u32> = t.iter().map(|r| r[d]).collect();
        common::assert_1d_stratified(&vals, 16, &format!("pmj dim {d}"));
    }
}

#[test]
fn every_dimension_is_stratified_over_every_prefix() {
    let t = build();
    for d in 0..4 {
        for m in 0..=16u32 {
            let vals: Vec<u32> = t[..1 << m].iter().map(|r| r[d]).collect();
            common::assert_1d_stratified(&vals, m, &format!("pmj dim {d} m={m}"));
        }
    }
}

#[test]
fn cross_pairs_are_not_02_nets() {
    // The second pair is an independent randomisation of the first, so mixed
    // projections have no net structure.
    let t = build();
    let is_net = |a: usize, b: usize, m: u32| -> bool {
        let n = 1usize << m;
        for i in 0..=m {
            let xb = i;
            let yb = m - i;
            let mut seen = vec![false; n];
            for r in &t[..n] {
                let x = if xb == 0 {
                    0
                } else {
                    (r[a] >> (32 - xb)) as usize
                };
                let y = if yb == 0 {
                    0
                } else {
                    (r[b] >> (32 - yb)) as usize
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
    for (a, b) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
        assert!(
            (1..=12).any(|m| !is_net(a, b, m)),
            "pair ({a},{b}) unexpectedly a net"
        );
    }
}

#[test]
fn second_pair_is_not_a_copy_of_the_first() {
    let t = build();
    let same = t.iter().filter(|r| r[0] == r[2] && r[1] == r[3]).count();
    assert!(same <= 1, "{same} rows have identical pairs");
}

#[test]
fn rows_are_all_distinct() {
    let t = build();
    common::assert_all_distinct(&t, "pmj rows");
}

#[test]
fn all_32_bits_are_exercised_in_every_dimension() {
    let t = build();
    for d in 0..4 {
        let or = t.iter().fold(0u32, |acc, r| acc | r[d]);
        let and = t.iter().fold(u32::MAX, |acc, r| acc & r[d]);
        assert_eq!(or, u32::MAX, "dim {d}");
        assert_eq!(and, 0, "dim {d}");
    }
}

#[test]
fn low_bits_are_balanced() {
    // The random jitter within each stratum should be unbiased.
    let t = build();
    for d in 0..4 {
        for bit in 0..16 {
            let ones = t.iter().filter(|r| (r[d] >> bit) & 1 == 1).count();
            let frac = ones as f64 / FULL as f64;
            assert!((frac - 0.5).abs() < 0.02, "dim {d} bit {bit}: {frac}");
        }
    }
}

#[test]
fn means_are_close_to_half_over_prefixes() {
    let t = build();
    for m in [4u32, 8, 12, 16] {
        let n = 1usize << m;
        for d in 0..4 {
            let mean = t[..n]
                .iter()
                .map(|r| r[d] as f64 / 4294967296.0)
                .sum::<f64>()
                / n as f64;
            assert!(
                (mean - 0.5).abs() < 2.0 / n as f64 + 1e-3,
                "m={m} dim {d}: {mean}"
            );
        }
    }
}

#[test]
fn first_pair_integrates_a_smooth_function_far_better_than_random() {
    let f = |x: f64, y: f64| (x * x + y * y) - 2.0 / 3.0;
    let t = build();
    let n = 1usize << 12;
    let qmc: f64 = t[..n]
        .iter()
        .map(|r| f(r[0] as f64 / 4294967296.0, r[1] as f64 / 4294967296.0))
        .sum::<f64>()
        / n as f64;
    let mut r = common::rng(21);
    let mc: f64 = (0..n)
        .map(|_| f(r.next_f32() as f64, r.next_f32() as f64))
        .sum::<f64>()
        / n as f64;
    assert!(qmc.abs() < 1e-3, "pmj err {qmc}");
    assert!(qmc.abs() < mc.abs(), "pmj {qmc} not better than mc {mc}");
}

#[test]
fn second_pair_integrates_a_smooth_function_accurately() {
    let f = |x: f64, y: f64| (x * x + y * y) - 2.0 / 3.0;
    let t = build();
    let n = 1usize << 12;
    let qmc: f64 = t[..n]
        .iter()
        .map(|r| f(r[2] as f64 / 4294967296.0, r[3] as f64 / 4294967296.0))
        .sum::<f64>()
        / n as f64;
    assert!(qmc.abs() < 1e-3, "pmj (2,3) err {qmc}");
}
