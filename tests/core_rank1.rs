//! Property tests for `openqmc::rank1` (oqmc/rank1.h) — the shifted rank-1
//! lattice.

mod common;

use openqmc::pcg;
use openqmc::permute::reverse_and_shuffle;
use openqmc::rank1::{lattice_reversed_index, shuffled_rotated_lattice};

/// Generator vector from Hickernell et al., as documented in rank1.rs.
const GENERATOR: [u32; 4] = [1, 364981, 245389, 97823];

const PATTERNS: [u32; 6] = [0, 1, 12345, 0x07bb_2fe2, 0xdead_beef, u32::MAX];

#[test]
fn reversed_index_of_one_is_the_generator() {
    for (dim, &g) in GENERATOR.iter().enumerate() {
        assert_eq!(lattice_reversed_index(1, dim), g);
    }
}

#[test]
fn reversed_index_of_zero_is_zero() {
    for dim in 0..4 {
        assert_eq!(lattice_reversed_index(0, dim), 0);
    }
}

#[test]
fn generator_components_are_odd() {
    for dim in 0..4 {
        assert_eq!(lattice_reversed_index(1, dim) & 1, 1, "dim {dim}");
    }
}

#[test]
fn reversed_index_is_additive() {
    let values = common::random_u32s(1, 4096);
    for dim in 0..4 {
        for w in values.windows(2) {
            assert_eq!(
                lattice_reversed_index(w[0].wrapping_add(w[1]), dim),
                lattice_reversed_index(w[0], dim).wrapping_add(lattice_reversed_index(w[1], dim))
            );
        }
    }
}

#[test]
fn reversed_index_is_wrapping_multiplication() {
    for (dim, &g) in GENERATOR.iter().enumerate() {
        for i in common::EDGE_U32
            .iter()
            .copied()
            .chain(common::random_u32s(2, 2048))
        {
            assert_eq!(lattice_reversed_index(i, dim), g.wrapping_mul(i));
        }
    }
}

#[test]
fn reversed_index_is_a_bijection_on_low_bits() {
    for dim in 0..4 {
        for bits in [1u32, 4, 8, 12, 16] {
            common::assert_low_bits_bijection(
                bits,
                &format!("lattice dim {dim} bits {bits}"),
                |i| lattice_reversed_index(i, dim),
            );
        }
    }
}

#[test]
fn reversed_index_dim0_is_identity() {
    for i in common::EDGE_U32
        .iter()
        .copied()
        .chain(common::random_u32s(3, 4096))
    {
        assert_eq!(lattice_reversed_index(i, 0), i);
    }
}

#[test]
fn shuffled_depth_prefixes_are_consistent() {
    for p in PATTERNS {
        for i in [0u32, 1, 2, 255, 65535, 65536, u32::MAX] {
            let four = shuffled_rotated_lattice::<4>(i, p);
            assert_eq!(&four[..3], &shuffled_rotated_lattice::<3>(i, p)[..]);
            assert_eq!(&four[..2], &shuffled_rotated_lattice::<2>(i, p)[..]);
            assert_eq!(&four[..1], &shuffled_rotated_lattice::<1>(i, p)[..]);
        }
    }
}

#[test]
fn shuffled_is_deterministic() {
    for p in PATTERNS {
        for i in common::random_u32s(4, 256) {
            assert_eq!(
                shuffled_rotated_lattice::<4>(i, p),
                shuffled_rotated_lattice::<4>(i, p)
            );
        }
    }
}

#[test]
fn shuffled_matches_the_documented_construction() {
    // sample[d] = GEN[d] * reverse_and_shuffle(i, output(p)) + rng_d(p)
    for p in PATTERNS {
        let mut shifts_state = p;
        let shifts: [u32; 4] = std::array::from_fn(|_| pcg::rng(&mut shifts_state));
        for i in common::random_u32s(5, 512) {
            let r = reverse_and_shuffle(i, pcg::output(p));
            let s = shuffled_rotated_lattice::<4>(i, p);
            for d in 0..4 {
                assert_eq!(
                    s[d],
                    GENERATOR[d].wrapping_mul(r).wrapping_add(shifts[d]),
                    "p {p:#x} i {i} d {d}"
                );
            }
        }
    }
}

#[test]
fn shuffled_dimensions_are_affine_in_dimension_zero() {
    // GEN[d] * sample[0] - sample[d] is constant across indices for a pattern.
    for p in PATTERNS {
        let base = shuffled_rotated_lattice::<4>(0, p);
        let consts: [u32; 4] =
            std::array::from_fn(|d| GENERATOR[d].wrapping_mul(base[0]).wrapping_sub(base[d]));
        for i in 1..4096u32 {
            let s = shuffled_rotated_lattice::<4>(i, p);
            for d in 0..4 {
                assert_eq!(
                    GENERATOR[d].wrapping_mul(s[0]).wrapping_sub(s[d]),
                    consts[d],
                    "p {p:#x} i {i} d {d}"
                );
            }
        }
    }
}

#[test]
fn shuffled_every_dimension_is_stratified_over_prefixes() {
    for p in PATTERNS {
        for m in 0..=12u32 {
            let rows: Vec<[u32; 4]> = (0..(1u32 << m))
                .map(|i| shuffled_rotated_lattice::<4>(i, p))
                .collect();
            for dim in 0..4 {
                let vals: Vec<u32> = rows.iter().map(|s| s[dim]).collect();
                common::assert_1d_stratified(&vals, m, &format!("pattern {p:#x} dim {dim} m={m}"));
            }
        }
    }
}

#[test]
fn shuffled_every_dimension_is_stratified_over_the_full_16_bit_range() {
    for p in [0u32, 0x07bb_2fe2] {
        let rows: Vec<[u32; 4]> = (0..65536u32)
            .map(|i| shuffled_rotated_lattice::<4>(i, p))
            .collect();
        for dim in 0..4 {
            let vals: Vec<u32> = rows.iter().map(|s| s[dim]).collect();
            common::assert_1d_stratified(&vals, 16, &format!("pattern {p:#x} dim {dim} full"));
        }
    }
}

#[test]
fn shuffled_every_dimension_is_stratified_over_aligned_blocks() {
    for p in PATTERNS {
        for m in 1..=8u32 {
            for block in [1u32, 5, 77, 1000, 65535] {
                let start = block << m;
                let rows: Vec<[u32; 4]> = (start..start + (1 << m))
                    .map(|i| shuffled_rotated_lattice::<4>(i, p))
                    .collect();
                for dim in 0..4 {
                    let vals: Vec<u32> = rows.iter().map(|s| s[dim]).collect();
                    common::assert_1d_stratified(
                        &vals,
                        m,
                        &format!("pattern {p:#x} dim {dim} block {block} m={m}"),
                    );
                }
            }
        }
    }
}

#[test]
fn shuffled_block_differences_lie_on_the_lattice() {
    // Within an aligned 2^m block, all pairwise differences are multiples of
    // GEN * 2^(32-m): the block is a shifted rank-1 lattice.
    for p in PATTERNS {
        for m in [1u32, 3, 6, 8] {
            let rows: Vec<[u32; 4]> = (0..(1u32 << m))
                .map(|i| shuffled_rotated_lattice::<4>(i, p))
                .collect();
            let step = 1u32 << (32 - m);
            for a in &rows {
                for b in &rows {
                    let k = a[0].wrapping_sub(b[0]);
                    assert_eq!(k % step, 0, "dim 0 difference not on the lattice");
                    let k_units = k / step;
                    for d in 1..4 {
                        let expected = GENERATOR[d].wrapping_mul(k_units.wrapping_mul(step));
                        assert_eq!(a[d].wrapping_sub(b[d]), expected, "p {p:#x} m {m} d {d}");
                    }
                }
            }
        }
    }
}

#[test]
fn shuffled_block_is_the_full_shifted_lattice() {
    for p in PATTERNS {
        for m in [2u32, 5, 10] {
            let rows: Vec<[u32; 4]> = (0..(1u32 << m))
                .map(|i| shuffled_rotated_lattice::<4>(i, p))
                .collect();
            let step = 1u32 << (32 - m);
            let origin = rows[0];
            let mut got: Vec<[u32; 4]> = rows
                .iter()
                .map(|r| std::array::from_fn(|d| r[d].wrapping_sub(origin[d])))
                .collect();
            got.sort_unstable();
            let mut want: Vec<[u32; 4]> = (0..(1u32 << m))
                .map(|k| std::array::from_fn(|d| GENERATOR[d].wrapping_mul(k.wrapping_mul(step))))
                .collect();
            want.sort_unstable();
            assert_eq!(got, want, "p {p:#x} m {m}");
        }
    }
}

#[test]
fn shuffled_values_repeat_as_a_set_beyond_16_bits_in_dim0() {
    // The shuffled index keeps its low 16 bits per aligned 2^16 block, so each
    // block visits the same 2^16 top-16-bit strata.
    let p = 0xdead_beef;
    let mut base: Vec<u32> = (0..65536u32)
        .map(|i| shuffled_rotated_lattice::<1>(i, p)[0] >> 16)
        .collect();
    base.sort_unstable();
    assert!(base.iter().enumerate().all(|(k, &v)| v == k as u32));
    let mut next: Vec<u32> = (65536..131072u32)
        .map(|i| shuffled_rotated_lattice::<1>(i, p)[0] >> 16)
        .collect();
    next.sort_unstable();
    assert_eq!(base, next);
}

#[test]
fn shuffled_distinct_patterns_give_distinct_sequences() {
    let mut firsts: Vec<[u32; 4]> = Vec::new();
    for prime in common::PRIMES {
        firsts.push(shuffled_rotated_lattice::<4>(0, pcg::hash(prime)));
    }
    common::assert_all_distinct(&firsts, "first lattice samples");
}

#[test]
fn shuffled_means_are_close_to_half() {
    let n = 1u32 << 12;
    for p in PATTERNS {
        let mut sums = [0.0f64; 4];
        for i in 0..n {
            let s = shuffled_rotated_lattice::<4>(i, p);
            for d in 0..4 {
                sums[d] += s[d] as f64 / 4294967296.0;
            }
        }
        for (d, sum) in sums.iter().enumerate() {
            let mean = sum / n as f64;
            assert!(
                (mean - 0.5).abs() < 1e-3,
                "pattern {p:#x} dim {d}: mean {mean}"
            );
        }
    }
}

#[test]
fn shuffled_integrates_a_smooth_periodic_function_accurately() {
    // Lattices excel on periodic integrands: integral of sin(2πx)sin(2πy) is 0.
    let n = 1u32 << 12;
    let mut acc = 0.0f64;
    for i in 0..n {
        let s = shuffled_rotated_lattice::<2>(i, 0x07bb_2fe2);
        let x = s[0] as f64 / 4294967296.0;
        let y = s[1] as f64 / 4294967296.0;
        acc += (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).sin();
    }
    assert!(
        (acc / n as f64).abs() < 1e-3,
        "lattice err {}",
        acc / n as f64
    );
}

#[test]
fn function_is_usable_in_const_context() {
    const V: u32 = lattice_reversed_index(3, 1);
    assert_eq!(V, 364981 * 3);
}
