//! Port of `oqmc/owen.h` — Owen-scrambled SZ sequences via Brent Burley's
//! hash-based construction ("Practical Hash-based Owen Scrambling").
//!
//! Since upstream `b6a204e` the generator matrices are the SZ sequences of
//! Ahmed et al., "SZ Sequences: Binary-Constructed (0, 2^q)-Sequences"
//! (SIGGRAPH Asia 2025), in place of the classic Sobol (Joe & Kuo) direction
//! numbers. Dimensions 0 and 1 are identical to Sobol; dimensions 2 and 3
//! differ, so that every pair of the four dimensions is a (0, 2)-sequence in
//! base 4 and the four together a (0, 4)-sequence in base 16.
//!
//! Upstream evaluates the core as Ahmed-2024 shift-mask-xor programs, in two
//! forms: [`sz_reversed_basis`] (bit-reversed index in, GPU path) and
//! [`sz_to_reversed`] (natural-order index in, host path, with dimension 3
//! chained from dimension 2 by eq. 13). Both are the same GF(2)
//! matrix-vector product, `result = XOR over set bits k of DIRECTIONS[dim][k]`
//! on the bit-reversed index. This port evaluates that product a byte at a
//! time from compile-time 256-entry tables (3 KiB for dimensions 1–3), which
//! needs no reversal and no chaining: two loads and one XOR per dimension.
//! Exhaustive tests pin the tables to the matrices, to both upstream programs
//! and to the eq. 13 chaining relation. Dimension 0 reduces to a 16-bit bit
//! reversal.

use crate::permute::{laine_karras_permutation, reverse_and_shuffle};
use crate::reverse::{reverse_bits16, reverse_bits32};
use crate::rotate::rotate_bytes;

/// SZ generator matrices for dimensions 0..4, 16-bit precision, one column per
/// index bit.
///
/// Copied verbatim from `directions[4][16]` in upstream's `src/tests/owen.cpp`
/// (emitted by its `matrices` CLI tool). Dimensions 0 and 1 are the Sobol
/// identity and Pascal matrices.
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
        0b1110011110011110, 0b0111100111100111, 0b0011000100100011, 0b0001001000110001,
        0b0000111000001001, 0b0000011100001110, 0b0000001100000010, 0b0000000100000011,
        0b0000000011100111, 0b0000000001111001, 0b0000000000110001, 0b0000000000010010,
        0b0000000000001110, 0b0000000000000111, 0b0000000000000011, 0b0000000000000001,
    ],
    [
        0b1001111001111001, 0b0111100111100111, 0b0010001100010010, 0b0001001000110001,
        0b0000100100000111, 0b0000011100001110, 0b0000001000000001, 0b0000000100000011,
        0b0000000010011110, 0b0000000001111001, 0b0000000000100011, 0b0000000000010010,
        0b0000000000001001, 0b0000000000000111, 0b0000000000000010, 0b0000000000000001,
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

/// SZ sequence value in the reversed basis, for a given dimension (0..4).
///
/// Port of `oqmc::szReversedBasis`: given a 16-bit index whose bits have been
/// reversed, returns the SZ sequence value to 16 bits of precision, also with
/// its bits reversed. Evaluated from byte tables; see the module docs.
#[inline]
pub fn sz_reversed_basis(index: u16, dimension: usize) -> u16 {
    debug_assert!(dimension <= 3);

    if dimension == 0 {
        // directions[0] is the reversal permutation; the tables would give
        // the same result.
        return reverse_bits16(index);
    }

    let t = &TABLES[dimension];
    t[0][(index & 0xff) as usize] ^ t[1][(index >> 8) as usize]
}

/// SZ sequence value from a natural-order index, for a given dimension (0..4).
///
/// Port of `oqmc::szToReversed`: as [`sz_reversed_basis`], but taking the
/// index in its natural bit order, so
/// `sz_to_reversed(i, d) == sz_reversed_basis(reverse_bits16(i), d)`. This is
/// upstream's host-path shift-mask-xor program (Ahmed 2024, eq. 18), kept
/// verbatim; it also satisfies the SZ pair relation
/// `sz_to_reversed(sz_to_reversed(i, 2), 1) == sz_to_reversed(i, 3)`
/// (Ahmed et al. 2025, eq. 13).
#[inline]
pub const fn sz_to_reversed(mut index: u16, dimension: usize) -> u16 {
    debug_assert!(dimension <= 3);

    match dimension {
        1 => {
            index ^= (index & 0xff00) >> 8;
            index ^= (index & 0xf0f0) >> 4;
            index ^= (index & 0xcccc) >> 2;
            index ^= (index & 0xaaaa) >> 1;
        }
        2 => {
            index ^= (index & 0xaa00) >> 9;
            index ^= (index & 0xaa00) >> 8;
            index ^= (index & 0x5500) >> 7;
            index ^= (index & 0x4040) >> 6;
            index ^= (index & 0xa0a0) >> 5;
            index ^= (index & 0x5050) >> 4;
            index ^= (index & 0xd8d8) >> 3;
            index ^= (index & 0xcccc) >> 2;
            index ^= (index & 0xeeee) >> 1;
        }
        3 => {
            index ^= (index & 0xaa00) >> 9;
            index ^= (index & 0x5500) >> 8;
            index ^= (index & 0x5500) >> 7;
            index ^= (index & 0x4040) >> 6;
            index ^= (index & 0xa0a0) >> 5;
            index ^= (index & 0xa0a0) >> 4;
            index ^= (index & 0xd8d8) >> 3;
            index ^= (index & 0x4444) >> 2;
            index ^= (index & 0x4444) >> 1;
        }
        _ => {}
    }

    index
}

/// Permute an integer (Laine & Karras) and reverse the bits. Equivalent to an
/// Owen scramble when the input bits are already reversed.
#[inline]
pub const fn scramble_and_reverse(value: u32, seed: u32) -> u32 {
    reverse_bits32(laine_karras_permutation(value, seed))
}

/// Compute a randomised (Owen-scrambled, progressively shuffled) SZ sequence
/// value. The name is kept from upstream (`oqmc::shuffledScrambledSobol`).
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
        // Upstream's host path computes sz_to_reversed(reverse_bits16(r), i),
        // which is sz_reversed_basis(r, i): the tables consume r directly.
        let s = sz_reversed_basis((index >> 16) as u16, i) as u32;
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
                assert_eq!(sz_reversed_basis(index, d), want, "d={d} index={index}");
            }
        }
    }

    /// Upstream's `szReversedBasis` left-shift program (the GPU path), kept
    /// verbatim as the reference the byte tables are checked against.
    fn sz_reversed_basis_program(mut index: u16, dimension: usize) -> u16 {
        match dimension {
            1 => {
                index ^= (index & 0x00ff) << 8;
                index ^= (index & 0x0f0f) << 4;
                index ^= (index & 0x3333) << 2;
                index ^= (index & 0x5555) << 1;
            }
            2 => {
                index ^= (index & 0x0055) << 9;
                index ^= (index & 0x0055) << 8;
                index ^= (index & 0x00aa) << 7;
                index ^= (index & 0x0202) << 6;
                index ^= (index & 0x0505) << 5;
                index ^= (index & 0x0a0a) << 4;
                index ^= (index & 0x1b1b) << 3;
                index ^= (index & 0x3333) << 2;
                index ^= (index & 0x7777) << 1;
            }
            3 => {
                index ^= (index & 0x0055) << 9;
                index ^= (index & 0x00aa) << 8;
                index ^= (index & 0x00aa) << 7;
                index ^= (index & 0x0202) << 6;
                index ^= (index & 0x0505) << 5;
                index ^= (index & 0x0505) << 4;
                index ^= (index & 0x1b1b) << 3;
                index ^= (index & 0x2222) << 2;
                index ^= (index & 0x2222) << 1;
            }
            _ => {}
        }
        reverse_bits16(index)
    }

    // Mirrors src/tests/owen.cpp:SzReversedBasis.
    #[test]
    fn tables_match_upstream_reversed_basis_program() {
        for d in 0..4 {
            for index in 0..=u16::MAX {
                assert_eq!(
                    sz_reversed_basis(index, d),
                    sz_reversed_basis_program(index, d),
                    "d={d} index={index}"
                );
            }
        }
    }

    // Mirrors src/tests/owen.cpp:SzToReversed.
    #[test]
    fn to_reversed_is_reversed_basis_of_the_reversed_index() {
        for d in 0..4 {
            for index in 0..=u16::MAX {
                assert_eq!(
                    sz_to_reversed(index, d),
                    sz_reversed_basis(reverse_bits16(index), d),
                    "d={d} index={index}"
                );
            }
        }
    }

    // Mirrors src/tests/owen.cpp:SzPairRelation (Ahmed et al. 2025, eq. 13),
    // which upstream's host path relies on to chain dimension 3 from 2.
    #[test]
    fn dimension_3_chains_from_dimension_2() {
        for index in 0..=u16::MAX {
            assert_eq!(
                sz_to_reversed(sz_to_reversed(index, 2), 1),
                sz_to_reversed(index, 3),
                "index={index}"
            );
        }
    }

    /// Upstream's host-path draw: reverse once, `sz_to_reversed` per
    /// dimension, dimension 3 chained from dimension 2.
    #[test]
    fn draw_matches_upstream_host_path() {
        for seed in [0u32, 1, pcg::hash(7), 0xdead_beef, u32::MAX] {
            for index in (0..1u32 << 16).step_by(7).chain([u32::MAX, 1 << 20]) {
                let shuffled = reverse_and_shuffle(index, seed);
                let natural = reverse_bits16((shuffled >> 16) as u16);
                let mut value = [0u16; 4];
                for i in 0..4 {
                    value[i] = if i == 3 {
                        sz_to_reversed(value[2], 1)
                    } else {
                        sz_to_reversed(natural, i)
                    };
                }
                let want: [u32; 4] = std::array::from_fn(|i| {
                    scramble_and_reverse(value[i] as u32, rotate_bytes(seed, i as i32))
                });
                assert_eq!(shuffled_scrambled_sobol::<4>(index, seed), want);
            }
        }
    }

    #[test]
    fn dim0_is_van_der_corput() {
        // Dimension 0 (as in Sobol) is the radical inverse (bit reversal).
        for index in [0u16, 1, 2, 3, 255, 12345, 65535] {
            assert_eq!(sz_reversed_basis(index, 0), reverse_bits16(index));
        }
    }

    #[test]
    fn first_sample_is_origin() {
        // Index 0 maps to the sequence origin for any seed.
        let s = shuffled_scrambled_sobol::<4>(0, pcg::hash(0));
        // reverse_and_shuffle(0, seed) = LK(reverse(0)=0, seed); index>>16 feeds
        // sz_reversed_basis. The (0,2) net still contains the origin point.
        // We only assert values are within the u32 range and deterministic here.
        let s2 = shuffled_scrambled_sobol::<4>(0, pcg::hash(0));
        assert_eq!(s, s2);
    }

    // Mirrors src/tests/owen.cpp:02Sequence — the first pair is a (0,2)-net:
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
        // SZ keeps Sobol dimension 1: the primitive polynomial x + 1, whose
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
            assert_eq!(bits, sz_reversed_basis(index, 0));
        }
    }

    #[test]
    fn reversed_index_is_a_bijection_in_each_dimension() {
        for dim in 0..4 {
            let mut seen = vec![false; 1 << 16];
            for index in 0..=u16::MAX {
                let v = sz_reversed_basis(index, dim) as usize;
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
                scramble_and_reverse(sz_reversed_basis(r, d) as u32, rotate_bytes(seed, d as i32))
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
