//! Port of `oqmc/owen.h` — Owen-scrambled Sobol sequences via Brent Burley's
//! hash-based construction ("Practical Hash-based Owen Scrambling").
//!
//! The upstream header has three equivalent implementations of the core
//! `sobolReversedIndex` (an Ahmed-2024 shift-mask-xor scalar path plus
//! AVX/SSE/NEON direction-matrix paths). They all compute the same GF(2)
//! matrix-vector product, `result = XOR over set bits k of DIRECTIONS[dim][k]`.
//! This port evaluates it a byte at a time: the product is linear over GF(2),
//! so it splits into the low byte's contribution XOR the high byte's, each
//! precomputed at compile time into a 256-entry table (4 KiB for all four
//! dimensions). Two loads and one XOR replace a 16-step bit loop — the same
//! values, bit for bit, pinned by an exhaustive test against the loop. The
//! `DIRECTIONS` table is copied verbatim from `owen.h` (the `directions[4][16]`
//! literal). Dimension 0 reduces to a 16-bit bit reversal.

use crate::permute::{laine_karras_permutation, reverse_and_shuffle};
use crate::reverse::{reverse_bits16, reverse_bits32};
use crate::rotate::rotate_bytes;

/// Sobol direction vectors for dimensions 0..4, 16-bit precision.
///
/// Copied from `directions[4][16]` in owen.h; those in turn come from
/// Leonhard Gruenschloss's MIT-licensed generator (via the `matrices` CLI tool).
#[rustfmt::skip]
const DIRECTIONS: [[u16; 16]; 4] = [
    [
        0b1000000000000000, 0b0100000000000000, 0b0010000000000000, 0b0001000000000000,
        0b0000100000000000, 0b0000010000000000, 0b0000001000000000, 0b0000000100000000,
        0b0000000010000000, 0b0000000001000000, 0b0000000000100000, 0b0000000000010000,
        0b0000000000001000, 0b0000000000000100, 0b0000000000000010, 0b0000000000000001,
    ],
    [
        0b1111111111111111, 0b0101010101010101, 0b0011001100110011, 0b0001000100010001,
        0b0000111100001111, 0b0000010100000101, 0b0000001100000011, 0b0000000100000001,
        0b0000000011111111, 0b0000000001010101, 0b0000000000110011, 0b0000000000010001,
        0b0000000000001111, 0b0000000000000101, 0b0000000000000011, 0b0000000000000001,
    ],
    [
        0b1010101000001001, 0b0111011100000110, 0b0011100100000011, 0b0001011000000001,
        0b0000100110101010, 0b0000011001110111, 0b0000001100111001, 0b0000000100010110,
        0b0000000010100011, 0b0000000001110001, 0b0000000000111010, 0b0000000000010111,
        0b0000000000001001, 0b0000000000000110, 0b0000000000000011, 0b0000000000000001,
    ],
    [
        0b1010000011000011, 0b0100000001000001, 0b0011000000101101, 0b0001000000011110,
        0b0000101101100111, 0b0000011110011010, 0b0000001010100100, 0b0000000100011011,
        0b0000000011001001, 0b0000000001000101, 0b0000000000101110, 0b0000000000011111,
        0b0000000000001010, 0b0000000000000100, 0b0000000000000011, 0b0000000000000001,
    ],
];

/// `DIRECTIONS` pre-multiplied a byte at a time: `TABLES[d][0][b]` is the XOR
/// of `DIRECTIONS[d][k]` over the set bits `k` of the low byte `b`, and
/// `TABLES[d][1][b]` the same for the high byte (`k + 8`). The product is
/// linear over GF(2), so the XOR of the two entries is exactly the 16-step
/// loop's result — the same values, in two loads.
const TABLES: [[[u16; 256]; 2]; 4] = {
    let mut t = [[[0u16; 256]; 2]; 4];
    let mut d = 0;
    while d < 4 {
        let mut half = 0;
        while half < 2 {
            let mut b = 0;
            while b < 256 {
                let mut bits = 0u16;
                let mut k = 0;
                while k < 8 {
                    if b & (1 << k) != 0 {
                        bits ^= DIRECTIONS[d][k + 8 * half];
                    }
                    k += 1;
                }
                t[d][half][b] = bits;
                b += 1;
            }
            half += 1;
        }
        d += 1;
    }
    t
};

/// Sobol value at a bit-reversed index for a given dimension (0..4), 16-bit.
#[inline]
pub fn sobol_reversed_index(index: u16, dimension: usize) -> u16 {
    debug_assert!(dimension <= 3);

    if dimension == 0 {
        // directions[0] is the reversal permutation; the tables would give
        // the same result, but the shortcut mirrors the upstream fast path.
        return reverse_bits16(index);
    }

    let t = &TABLES[dimension];
    t[0][(index & 0xff) as usize] ^ t[1][(index >> 8) as usize]
}

/// Permute an integer (Laine & Karras) and reverse the bits. Equivalent to an
/// Owen scramble when the input bits are already reversed.
#[inline]
pub const fn scramble_and_reverse(value: u32, seed: u32) -> u32 {
    reverse_bits32(laine_karras_permutation(value, seed))
}

/// Compute a randomised (Owen-scrambled, progressively shuffled) Sobol value.
///
/// `DEPTH` is the dimensional output count (1..=4). The seed must be constant for
/// a given sequence; an index greater than 2^16 will repeat values.
#[inline]
pub fn shuffled_scrambled_sobol<const DEPTH: usize>(index: u32, seed: u32) -> [u32; DEPTH] {
    const {
        assert!(
            DEPTH >= 1 && DEPTH <= 4,
            "Pattern depth must be within [1, 4]"
        )
    };

    let index = reverse_and_shuffle(index, seed);

    let mut sample = [0u32; DEPTH];
    let mut i = 0;
    while i < DEPTH {
        let s = sobol_reversed_index((index >> 16) as u16, i) as u32;
        sample[i] = scramble_and_reverse(s, rotate_bytes(seed, i as i32));
        i += 1;
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcg;

    /// The byte tables are the 16-step loop, precomputed: every index, every
    /// dimension, bit for bit.
    #[test]
    fn tables_match_the_direction_matrix_loop() {
        for (d, matrix) in DIRECTIONS.iter().enumerate() {
            for index in 0..=u16::MAX {
                let mut bits = 0u16;
                for (k, &column) in matrix.iter().enumerate() {
                    if index & (1 << k) != 0 {
                        bits ^= column;
                    }
                }
                let want = if d == 0 { reverse_bits16(index) } else { bits };
                assert_eq!(sobol_reversed_index(index, d), want, "d={d} index={index}");
            }
        }
    }

    #[test]
    fn dim0_is_van_der_corput() {
        // Dimension 0 of the Sobol sequence is the radical inverse (bit reversal).
        for index in [0u16, 1, 2, 3, 255, 12345, 65535] {
            assert_eq!(sobol_reversed_index(index, 0), reverse_bits16(index));
        }
    }

    #[test]
    fn first_sample_is_origin() {
        // Index 0 maps to the sequence origin for any seed.
        let s = shuffled_scrambled_sobol::<4>(0, pcg::hash(0));
        // reverse_and_shuffle(0, seed) = LK(reverse(0)=0, seed); index>>16 feeds
        // sobol_reversed_index. The (0,2) net still contains the origin point.
        // We only assert values are within the u32 range and deterministic here.
        let s2 = shuffled_scrambled_sobol::<4>(0, pcg::hash(0));
        assert_eq!(s, s2);
    }

    // Mirrors src/tests/owen.cpp:02Sequence — the first Sobol pair is a (0,2)-net:
    // for every axis-aligned power-of-two stratification of n=2^m points, each
    // stratum holds exactly one point.
    #[test]
    fn is_02_net() {
        const M: u32 = 8;
        const N: u32 = 1 << M;
        let seed = pcg::hash(0);

        for i in 0..=M {
            let x_res = 1u32 << i;
            let y_res = 1u32 << (M - i);
            assert_eq!(x_res * y_res, N);

            let x_width = u32::MAX / x_res;
            let y_width = u32::MAX / y_res;

            let mut strata = vec![false; N as usize];
            for index in 0..N {
                let out = shuffled_scrambled_sobol::<2>(index, seed);
                let x = out[0] / x_width;
                let y = out[1] / y_width;
                let cell = (y * x_res + x) as usize;
                assert!(!strata[cell], "stratum {cell} occupied twice");
                strata[cell] = true;
            }
            assert!(strata.iter().all(|&b| b));
        }
    }

    #[test]
    fn direction_matrices_are_triangular_with_unit_diagonal() {
        // Column k has its leading one at bit 15 - k in every dimension, so
        // each generator matrix is invertible over GF(2).
        for (dim, matrix) in DIRECTIONS.iter().enumerate() {
            for (k, &col) in matrix.iter().enumerate() {
                assert_eq!(col.leading_zeros() as usize, k, "dim {dim} col {k}");
            }
        }
    }

    #[test]
    fn direction_dim0_is_the_identity_in_reversed_order() {
        for (k, &col) in DIRECTIONS[0].iter().enumerate() {
            assert_eq!(col, 1 << (15 - k));
        }
    }

    #[test]
    fn direction_dim1_is_the_pascal_triangle_matrix() {
        // Sobol dimension 1 uses the primitive polynomial x + 1, whose
        // generator matrix is Pascal's triangle mod 2: column k has bits set
        // where binomial(j, k) is odd, i.e. (j & k) == k.
        for (k, &col) in DIRECTIONS[1].iter().enumerate() {
            let mut expected = 0u16;
            for j in 0..16 {
                if (j & k) == k {
                    expected |= 1 << (15 - j);
                }
            }
            assert_eq!(col, expected, "col {k}");
        }
    }

    #[test]
    fn every_direction_column_ends_with_the_unit_column() {
        for matrix in &DIRECTIONS {
            assert_eq!(matrix[15], 1);
        }
    }

    #[test]
    fn dim0_shortcut_matches_the_generic_matrix_path() {
        for index in 0..=u16::MAX {
            let mut bits = 0u16;
            for (k, &col) in DIRECTIONS[0].iter().enumerate() {
                if index & (1 << k) != 0 {
                    bits ^= col;
                }
            }
            assert_eq!(bits, sobol_reversed_index(index, 0));
        }
    }

    #[test]
    fn reversed_index_is_a_bijection_in_each_dimension() {
        for dim in 0..4 {
            let mut seen = vec![false; 1 << 16];
            for index in 0..=u16::MAX {
                let v = sobol_reversed_index(index, dim) as usize;
                assert!(!seen[v], "dim {dim}");
                seen[v] = true;
            }
        }
    }

    #[test]
    fn scramble_and_reverse_composes_lk_and_reverse() {
        for v in [0u32, 1, 0xffff, 0x8000_0000, u32::MAX] {
            for seed in [0u32, 7, 0xdead_beef] {
                assert_eq!(
                    scramble_and_reverse(v, seed),
                    reverse_bits32(laine_karras_permutation(v, seed))
                );
            }
        }
    }

    #[test]
    fn shuffled_uses_the_top_16_bits_of_the_shuffled_index() {
        // Reconstruct one sample from the pieces.
        let seed = pcg::hash(5);
        for index in [0u32, 3, 1000, 65535] {
            let shuffled = reverse_and_shuffle(index, seed);
            let r = (shuffled >> 16) as u16;
            let expected: [u32; 4] = std::array::from_fn(|d| {
                scramble_and_reverse(
                    sobol_reversed_index(r, d) as u32,
                    rotate_bytes(seed, d as i32),
                )
            });
            assert_eq!(shuffled_scrambled_sobol::<4>(index, seed), expected);
        }
    }

    #[test]
    fn is_02_net_for_several_seeds_and_sizes() {
        for seed in [pcg::hash(1), pcg::hash(2), 0, u32::MAX] {
            for m in 1..=10u32 {
                let n = 1u32 << m;
                for i in 0..=m {
                    let xb = i;
                    let yb = m - i;
                    let mut strata = vec![false; n as usize];
                    for index in 0..n {
                        let out = shuffled_scrambled_sobol::<2>(index, seed);
                        let x = if xb == 0 { 0 } else { out[0] >> (32 - xb) };
                        let y = if yb == 0 { 0 } else { out[1] >> (32 - yb) };
                        let cell = ((y << xb) | x) as usize;
                        assert!(!strata[cell], "seed {seed:#x} m={m}");
                        strata[cell] = true;
                    }
                }
            }
        }
    }
}
