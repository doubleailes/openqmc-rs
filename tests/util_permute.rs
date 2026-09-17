//! Property tests for `openqmc::permute` (oqmc/permute.h) — the Laine-Karras
//! permutation and the hash-based Owen scramble built from it.

mod common;

use openqmc::permute::{laine_karras_permutation, reverse_and_shuffle, shuffle};
use openqmc::reverse::reverse_bits32;

const SEEDS: [u32; 8] = [
    0,
    1,
    2,
    0x07bb_2fe2,
    0xdead_beef,
    0x8000_0000,
    0xffff_ffff,
    0x1234_5678,
];

#[test]
fn lk_is_deterministic() {
    for v in common::random_u32s(1, 256) {
        for seed in SEEDS {
            assert_eq!(
                laine_karras_permutation(v, seed),
                laine_karras_permutation(v, seed)
            );
        }
    }
}

#[test]
fn lk_lower_bits_never_depend_on_higher_bits() {
    // Output bit i depends only on input bits <= i: flipping bit j leaves all
    // output bits below j untouched.
    for v in common::random_u32s(2, 2048) {
        for seed in SEEDS {
            let base = laine_karras_permutation(v, seed);
            for j in 0..32 {
                let flipped = laine_karras_permutation(v ^ (1 << j), seed);
                let low_mask = (1u32 << j) - 1;
                assert_eq!(
                    base & low_mask,
                    flipped & low_mask,
                    "{v:#x} seed {seed:#x} bit {j}"
                );
            }
        }
    }
}

#[test]
fn lk_is_a_bijection_on_every_low_bit_prefix() {
    for bits in 1..=12u32 {
        for seed in SEEDS {
            common::assert_low_bits_bijection(
                bits,
                &format!("lk bits={bits} seed={seed:#x}"),
                |i| laine_karras_permutation(i, seed),
            );
        }
    }
}

#[test]
fn lk_is_a_bijection_on_the_low_16_bits() {
    for seed in [0u32, 7, 0xdead_beef] {
        common::assert_low_bits_bijection(16, &format!("lk 16 bits seed={seed:#x}"), |i| {
            laine_karras_permutation(i, seed)
        });
    }
}

#[test]
fn lk_with_different_seeds_gives_different_permutations() {
    let probe: Vec<u32> = (0..64).collect();
    let mut results: Vec<Vec<u32>> = Vec::new();
    for seed in common::PRIMES {
        let perm: Vec<u32> = probe
            .iter()
            .map(|&v| laine_karras_permutation(v, seed))
            .collect();
        assert!(
            !results.contains(&perm),
            "seed {seed} repeats a permutation"
        );
        results.push(perm);
    }
}

#[test]
fn lk_flipping_a_bit_always_flips_that_bit() {
    // The diagonal of the (lower-triangular) permutation is always one: since
    // the multiplier (seed >> 16 | 1) is odd, bit j of the output toggles when
    // bit j of the input toggles (and lower bits are held fixed).
    for v in common::random_u32s(3, 2048) {
        for seed in SEEDS {
            let base = laine_karras_permutation(v, seed);
            for j in 0..32 {
                let flipped = laine_karras_permutation(v ^ (1 << j), seed);
                assert_ne!(
                    base & (1 << j),
                    flipped & (1 << j),
                    "{v:#x} seed {seed:#x} bit {j}"
                );
            }
        }
    }
}

#[test]
fn reverse_and_shuffle_is_lk_of_reversed() {
    for v in common::random_u32s(4, 4096) {
        for seed in SEEDS {
            assert_eq!(
                reverse_and_shuffle(v, seed),
                laine_karras_permutation(reverse_bits32(v), seed)
            );
        }
    }
}

#[test]
fn shuffle_is_reverse_of_reverse_and_shuffle() {
    for v in common::random_u32s(5, 4096) {
        for seed in SEEDS {
            assert_eq!(
                shuffle(v, seed),
                reverse_bits32(reverse_and_shuffle(v, seed))
            );
        }
    }
}

#[test]
fn shuffle_higher_bits_never_depend_on_lower_bits() {
    // The Owen property: flipping input bit j leaves output bits above j alone.
    for v in common::random_u32s(6, 2048) {
        for seed in SEEDS {
            let base = shuffle(v, seed);
            for j in 0..32 {
                let flipped = shuffle(v ^ (1 << j), seed);
                let high_mask = !((2u64.pow(j + 1) - 1) as u32);
                assert_eq!(
                    base & high_mask,
                    flipped & high_mask,
                    "{v:#x} seed {seed:#x} bit {j}"
                );
                assert_ne!(base & (1 << j), flipped & (1 << j));
            }
        }
    }
}

#[test]
fn shuffle_preserves_top_bit_prefix_classes() {
    // Two values sharing their top k bits shuffle to values sharing top k bits.
    let values = common::random_u32s(7, 1024);
    for seed in SEEDS {
        for &a in &values {
            for k in [1u32, 4, 8, 12, 16, 24] {
                let b = a ^ (common::rng(a ^ k).next_u32() >> k);
                assert_eq!(a >> (32 - k), b >> (32 - k));
                assert_eq!(shuffle(a, seed) >> (32 - k), shuffle(b, seed) >> (32 - k));
            }
        }
    }
}

#[test]
fn shuffle_is_a_bijection_on_every_low_bit_prefix() {
    for bits in 1..=12u32 {
        for seed in SEEDS {
            common::assert_low_bits_bijection(
                bits,
                &format!("shuffle bits={bits} seed={seed:#x}"),
                |i| shuffle(i, seed),
            );
        }
    }
}

#[test]
fn shuffle_is_a_bijection_on_the_low_16_bits() {
    for seed in [0u32, 0x07bb_2fe2, 0xdead_beef] {
        common::assert_low_bits_bijection(16, &format!("shuffle 16 bits seed={seed:#x}"), |i| {
            shuffle(i, seed)
        });
    }
}

#[test]
fn shuffle_maps_aligned_blocks_onto_aligned_blocks() {
    // Indices [b*2^m, (b+1)*2^m) shuffle to some other block of the same size.
    for seed in SEEDS {
        for m in 1..=10u32 {
            for block in [0u32, 1, 5, 100, 4000] {
                let start = block << m;
                let outs: Vec<u32> = (start..start + (1 << m))
                    .map(|i| shuffle(i, seed))
                    .collect();
                let prefix = outs[0] >> m;
                assert!(
                    outs.iter().all(|&o| o >> m == prefix),
                    "seed {seed:#x} m={m} block={block}"
                );
                let mut low: Vec<u32> = outs.iter().map(|&o| o & ((1 << m) - 1)).collect();
                low.sort_unstable();
                assert!(low.iter().enumerate().all(|(i, &l)| l == i as u32));
            }
        }
    }
}

#[test]
fn reverse_and_shuffle_low_16_bits_are_constant_over_a_16_bit_index_range() {
    // Indices < 2^16 reverse into the top half, so LK leaves the low 16 output
    // bits depending only on the seed. The samplers rely on this when they take
    // `>> 16`.
    for seed in SEEDS {
        let low = reverse_and_shuffle(0, seed) & 0xffff;
        for i in (0..65536u32).step_by(7) {
            assert_eq!(
                reverse_and_shuffle(i, seed) & 0xffff,
                low,
                "seed {seed:#x} i={i}"
            );
        }
    }
}

#[test]
fn reverse_and_shuffle_top_16_bits_permute_a_16_bit_index_range() {
    for seed in [0u32, 3, 0xdead_beef] {
        let mut seen = vec![false; 65536];
        for i in 0..65536u32 {
            let top = (reverse_and_shuffle(i, seed) >> 16) as usize;
            assert!(!seen[top], "seed {seed:#x} collision at {i}");
            seen[top] = true;
        }
    }
}

#[test]
fn shuffle_with_distinct_seeds_disagrees_somewhere() {
    let probe: Vec<u32> = (0..256).collect();
    for (i, &a) in common::PRIMES.iter().enumerate() {
        for &b in &common::PRIMES[i + 1..] {
            assert!(
                probe.iter().any(|&v| shuffle(v, a) != shuffle(v, b)),
                "{a} vs {b}"
            );
        }
    }
}

#[test]
fn shuffle_is_not_the_identity() {
    for seed in SEEDS {
        let moved = (0..1024u32).filter(|&v| shuffle(v, seed) != v).count();
        assert!(moved > 900, "seed {seed:#x} moved only {moved}");
    }
}

#[test]
fn functions_are_usable_in_const_context() {
    const A: u32 = laine_karras_permutation(5, 7);
    const B: u32 = reverse_and_shuffle(5, 7);
    const C: u32 = shuffle(5, 7);
    assert_eq!(A, laine_karras_permutation(5, 7));
    assert_eq!(B, reverse_and_shuffle(5, 7));
    assert_eq!(C, shuffle(5, 7));
}
