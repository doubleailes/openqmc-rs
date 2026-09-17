//! Port of `oqmc/rank1.h` — a rank-1 lattice (Hickernell et al., "Weighted
//! Compound Integration Rules with Higher Order Convergence for all N") made
//! progressive with a radical inversion of the sample index, randomised with
//! toroidal shifts.

use crate::pcg;
use crate::permute::reverse_and_shuffle;

/// Generator vector from the Hickernell et al. publication.
const LATTICE: [u32; 4] = [1, 364981, 245389, 97823];

/// Toroidal shift: offset a value, relying on integer wraparound.
#[inline]
const fn rotate(value: u32, distance: u32) -> u32 {
    value.wrapping_add(distance)
}

/// Rank-1 lattice value at a bit-reversed index for a given dimension (0..4).
#[inline]
pub const fn lattice_reversed_index(index: u32, dimension: usize) -> u32 {
    debug_assert!(dimension <= 3);
    LATTICE[dimension].wrapping_mul(index)
}

/// Compute a randomised rank-1 lattice value.
///
/// `DEPTH` is the dimensional output count (1..=4). `pattern_id` seeds the
/// randomisation and must be constant for a given lattice.
#[inline]
pub fn shuffled_rotated_lattice<const DEPTH: usize>(
    index: u32,
    mut pattern_id: u32,
) -> [u32; DEPTH] {
    const {
        assert!(
            DEPTH >= 1 && DEPTH <= 4,
            "Pattern depth must be within [1, 4]"
        )
    };

    let index = reverse_and_shuffle(index, pcg::output(pattern_id));

    let mut sample = [0u32; DEPTH];
    let mut i = 0;
    while i < DEPTH {
        // pcg::rng advances `pattern_id` and returns a fresh shift each iteration.
        sample[i] = rotate(lattice_reversed_index(index, i), pcg::rng(&mut pattern_id));
        i += 1;
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let a = shuffled_rotated_lattice::<4>(7, pcg::hash(3));
        let b = shuffled_rotated_lattice::<4>(7, pcg::hash(3));
        assert_eq!(a, b);
    }

    #[test]
    fn dimension0_generator_is_one() {
        // LATTICE[0] == 1, so dim0 is a pure radical-inverse + shift.
        assert_eq!(lattice_reversed_index(12345, 0), 12345);
    }

    #[test]
    fn generator_vector_matches_the_publication() {
        assert_eq!(LATTICE, [1, 364981, 245389, 97823]);
    }

    #[test]
    fn generator_components_are_odd_and_distinct() {
        for (d, &g) in LATTICE.iter().enumerate() {
            assert_eq!(g & 1, 1, "dim {d}");
        }
        let mut sorted = LATTICE;
        sorted.sort_unstable();
        for w in sorted.windows(2) {
            assert_ne!(w[0], w[1]);
        }
    }

    #[test]
    fn toroidal_rotate_is_wrapping_addition() {
        assert_eq!(rotate(u32::MAX, 1), 0);
        assert_eq!(rotate(0, 0), 0);
        assert_eq!(rotate(5, 7), 12);
        assert_eq!(rotate(0x8000_0000, 0x8000_0000), 0);
    }

    #[test]
    fn reversed_index_is_generator_times_index() {
        for (d, &g) in LATTICE.iter().enumerate() {
            for i in [0u32, 1, 2, 12345, u32::MAX] {
                assert_eq!(lattice_reversed_index(i, d), g.wrapping_mul(i));
            }
        }
    }

    #[test]
    fn shuffled_seeds_the_shuffle_with_output_and_the_shifts_with_rng() {
        let pattern = pcg::hash(9);
        for index in [0u32, 1, 77, 65535] {
            let r = reverse_and_shuffle(index, pcg::output(pattern));
            let mut state = pattern;
            let expected: [u32; 4] =
                std::array::from_fn(|d| rotate(lattice_reversed_index(r, d), pcg::rng(&mut state)));
            assert_eq!(shuffled_rotated_lattice::<4>(index, pattern), expected);
        }
    }

    #[test]
    fn depth_prefixes_agree() {
        let p = pcg::hash(4);
        let four = shuffled_rotated_lattice::<4>(11, p);
        assert_eq!(&four[..2], &shuffled_rotated_lattice::<2>(11, p)[..]);
        assert_eq!(&four[..1], &shuffled_rotated_lattice::<1>(11, p)[..]);
    }

    #[test]
    fn each_dimension_is_stratified_over_a_prefix() {
        let p = pcg::hash(8);
        for m in 1..=10u32 {
            for d in 0..4 {
                let mut seen = vec![false; 1 << m];
                for i in 0..(1u32 << m) {
                    let cell = (shuffled_rotated_lattice::<4>(i, p)[d] >> (32 - m)) as usize;
                    assert!(!seen[cell], "m={m} d={d}");
                    seen[cell] = true;
                }
            }
        }
    }
}
