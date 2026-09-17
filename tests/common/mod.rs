//! Shared helpers for the integration test suites. Dependency-free: the crate's
//! own PCG stream serves as the deterministic source of "random" test inputs.

#![allow(dead_code)]

use openqmc::pcg::Rng;

/// The first twenty primes — the same seed set the upstream C++ tests use.
pub const PRIMES: [u32; 20] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
];

/// Interesting 32-bit edge values.
pub const EDGE_U32: [u32; 12] = [
    0,
    1,
    2,
    0x7f,
    0x80,
    0xff,
    0x7fff_ffff,
    0x8000_0000,
    0x8000_0001,
    0xffff_fffe,
    0xffff_ffff,
    0xdead_beef,
];

/// Interesting signed 32-bit edge values.
pub const EDGE_I32: [i32; 9] = [0, 1, -1, 2, -2, 255, 256, i32::MAX, i32::MIN];

/// A deterministic stream of test inputs seeded from `seed`.
pub fn rng(seed: u32) -> Rng {
    Rng::new(openqmc::pcg::hash(seed))
}

/// `n` deterministic pseudo-random `u32`s.
pub fn random_u32s(seed: u32, n: usize) -> Vec<u32> {
    let mut r = rng(seed);
    (0..n).map(|_| r.next_u32()).collect()
}

/// Assert that `values`, each in `[0, 2^32)`, is stratified into `1 << m`
/// equal-width strata with exactly one point per stratum.
pub fn assert_1d_stratified(values: &[u32], m: u32, what: &str) {
    let n = 1usize << m;
    assert_eq!(values.len(), n, "{what}: need exactly 2^{m} points");
    let mut seen = vec![false; n];
    for &v in values {
        let cell = if m == 0 { 0 } else { (v >> (32 - m)) as usize };
        assert!(!seen[cell], "{what}: stratum {cell} of {n} occupied twice");
        seen[cell] = true;
    }
    assert!(seen.iter().all(|&b| b), "{what}: some stratum left empty");
}

/// Assert that the 2D point set is a (0, m, 2)-net in base 2: every
/// axis-aligned elementary interval of area 2^-m holds exactly one point.
pub fn assert_02_net(points: &[[u32; 2]], m: u32, what: &str) {
    let n = 1usize << m;
    assert_eq!(points.len(), n, "{what}: need exactly 2^{m} points");
    for i in 0..=m {
        let xbits = i;
        let ybits = m - i;
        let mut seen = vec![false; n];
        for p in points {
            let x = if xbits == 0 {
                0
            } else {
                (p[0] >> (32 - xbits)) as usize
            };
            let y = if ybits == 0 {
                0
            } else {
                (p[1] >> (32 - ybits)) as usize
            };
            let cell = (y << xbits) | x;
            assert!(
                !seen[cell],
                "{what}: elementary interval {cell} ({xbits}x{ybits} bits) occupied twice"
            );
            seen[cell] = true;
        }
        assert!(seen.iter().all(|&b| b), "{what}: some interval left empty");
    }
}

/// Assert every element of `items` is distinct.
pub fn assert_all_distinct<T: Ord + Clone + std::fmt::Debug>(items: &[T], what: &str) {
    let mut sorted = items.to_vec();
    sorted.sort();
    for w in sorted.windows(2) {
        assert_ne!(w[0], w[1], "{what}: duplicate value {:?}", w[0]);
    }
}

/// Assert `f` is a bijection on `[0, 2^bits)` when restricted to its low `bits`
/// output bits.
pub fn assert_low_bits_bijection(bits: u32, what: &str, f: impl Fn(u32) -> u32) {
    let n = 1usize << bits;
    let mask = (n as u32).wrapping_sub(1);
    let mut seen = vec![false; n];
    for i in 0..n as u32 {
        let out = (f(i) & mask) as usize;
        assert!(!seen[out], "{what}: output {out} produced twice");
        seen[out] = true;
    }
}

/// Reference conversion: truncate `v` to 24 significant bits and scale by
/// 2^-32, computed in f64 so every step is exact.
pub fn reference_uint_to_float(v: u32) -> f32 {
    if v == 0 {
        return 0.0;
    }
    let width = 32 - v.leading_zeros();
    let drop = width.saturating_sub(24);
    let truncated = (v >> drop) << drop;
    (truncated as f64 / 4294967296.0) as f32
}
