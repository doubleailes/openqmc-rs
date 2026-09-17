//! Property tests for `openqmc::range` (oqmc/range.h) — Lemire's multiply-shift.

mod common;

use openqmc::range::{uint_to_range, uint_to_range_between};

#[test]
fn range_one_always_yields_zero() {
    for v in common::EDGE_U32
        .iter()
        .copied()
        .chain(common::random_u32s(1, 4096))
    {
        assert_eq!(uint_to_range(v, 1), 0);
    }
}

#[test]
fn zero_input_always_yields_zero() {
    for range in [1u32, 2, 3, 7, 1000, 65536, 1 << 31, u32::MAX] {
        assert_eq!(uint_to_range(0, range), 0);
    }
}

#[test]
fn max_input_always_yields_range_minus_one() {
    for range in [1u32, 2, 3, 7, 1000, 65536, 1 << 31, u32::MAX] {
        assert_eq!(uint_to_range(u32::MAX, range), range - 1);
    }
}

#[test]
fn output_is_always_below_range_for_edge_inputs() {
    for v in common::EDGE_U32 {
        for range in [
            1u32,
            2,
            3,
            5,
            6,
            7,
            100,
            255,
            256,
            257,
            65535,
            65536,
            1 << 31,
            u32::MAX,
        ] {
            assert!(uint_to_range(v, range) < range, "{v:#x} into {range}");
        }
    }
}

#[test]
fn output_is_always_below_range_for_random_inputs() {
    let values = common::random_u32s(2, 1 << 16);
    for range in [2u32, 3, 10, 12345, 1 << 20, u32::MAX] {
        for &v in &values {
            assert!(uint_to_range(v, range) < range);
        }
    }
}

#[test]
fn matches_the_multiply_shift_formula() {
    for v in common::random_u32s(3, 1 << 16) {
        for range in [1u32, 2, 3, 97, 65536, 0xffff_ffff] {
            let expected = ((v as u64 * range as u64) >> 32) as u32;
            assert_eq!(uint_to_range(v, range), expected);
        }
    }
}

#[test]
fn is_monotonic_non_decreasing_in_the_value() {
    let mut values = common::random_u32s(4, 1 << 16);
    values.sort_unstable();
    for range in [2u32, 3, 1000, 1 << 24] {
        for w in values.windows(2) {
            assert!(uint_to_range(w[0], range) <= uint_to_range(w[1], range));
        }
    }
}

#[test]
fn is_monotonic_non_decreasing_in_the_range() {
    for v in common::random_u32s(5, 4096) {
        let mut last = 0;
        for range in 1..200u32 {
            let r = uint_to_range(v, range);
            assert!(r >= last, "{v:#x} range {range}");
            last = r;
        }
    }
}

#[test]
fn power_of_two_ranges_take_the_top_bits() {
    for v in common::random_u32s(6, 1 << 14) {
        for k in 0..32u32 {
            let range = 1u32 << k;
            let expected = if k == 0 { 0 } else { v >> (32 - k) };
            assert_eq!(uint_to_range(v, range), expected, "{v:#x} into 2^{k}");
        }
    }
}

#[test]
fn power_of_two_buckets_are_exactly_equal_sized() {
    // Bucket k for range 2^m is exactly [k << (32-m), (k+1) << (32-m)).
    for m in 1..=16u32 {
        let range = 1u32 << m;
        let width = 1u64 << (32 - m);
        for k in [0u64, 1, (range as u64) / 2, range as u64 - 1] {
            let first = (k * width) as u32;
            let last = (k * width + width - 1) as u32;
            assert_eq!(uint_to_range(first, range), k as u32);
            assert_eq!(uint_to_range(last, range), k as u32);
            if k > 0 {
                assert_eq!(uint_to_range(first - 1, range), k as u32 - 1);
            }
        }
    }
}

#[test]
fn bucket_boundaries_match_the_analytic_ceiling() {
    // The smallest v that maps to bucket k is ceil(k * 2^32 / range).
    for range in [2u32, 3, 5, 6, 7, 9, 10, 100, 255, 257, 1000, 65537] {
        for k in 1..range.min(300) {
            let first = ((k as u64 * (1u64 << 32)).div_ceil(range as u64)) as u32;
            assert_eq!(uint_to_range(first, range), k, "range {range} bucket {k}");
            assert_eq!(
                uint_to_range(first - 1, range),
                k - 1,
                "range {range} bucket {k}"
            );
        }
    }
}

#[test]
fn bucket_sizes_differ_by_at_most_one() {
    for range in [3u32, 5, 6, 7, 9, 10, 11, 13, 100, 999] {
        let mut sizes = Vec::with_capacity(range as usize);
        for k in 0..range as u64 {
            let lo = (k * (1u64 << 32)).div_ceil(range as u64);
            let hi = ((k + 1) * (1u64 << 32)).div_ceil(range as u64);
            sizes.push(hi - lo);
        }
        let min = *sizes.iter().min().unwrap();
        let max = *sizes.iter().max().unwrap();
        assert!(max - min <= 1, "range {range}: sizes {min}..{max}");
        assert_eq!(sizes.iter().sum::<u64>(), 1u64 << 32);
    }
}

#[test]
fn every_bucket_is_reachable() {
    for range in [2u32, 3, 7, 16, 100, 1000] {
        let mut seen = vec![false; range as usize];
        for k in 0..range as u64 {
            let first = (k * (1u64 << 32)).div_ceil(range as u64) as u32;
            seen[uint_to_range(first, range) as usize] = true;
        }
        assert!(seen.iter().all(|&b| b), "range {range}");
    }
}

#[test]
fn histogram_is_flat_for_a_stratified_input() {
    // A perfectly uniform grid maps to a perfectly flat histogram whenever the
    // range divides the grid size.
    for range in [2u32, 4, 8, 16, 64, 256, 1024] {
        let grid = 1u64 << 16;
        let mut counts = vec![0u32; range as usize];
        for i in 0..grid {
            let v = (i << 16) as u32;
            counts[uint_to_range(v, range) as usize] += 1;
        }
        let expected = (grid / range as u64) as u32;
        assert!(
            counts.iter().all(|&c| c == expected),
            "range {range}: {counts:?}"
        );
    }
}

#[test]
fn histogram_is_nearly_flat_for_random_input() {
    let values = common::random_u32s(7, 1 << 18);
    for range in [3u32, 7, 10, 33] {
        let mut counts = vec![0u64; range as usize];
        for &v in &values {
            counts[uint_to_range(v, range) as usize] += 1;
        }
        let expected = values.len() as f64 / range as f64;
        for (k, &c) in counts.iter().enumerate() {
            let dev = (c as f64 - expected).abs() / expected;
            assert!(dev < 0.05, "range {range} bucket {k}: {c} vs {expected}");
        }
    }
}

#[test]
fn between_offsets_by_begin() {
    for v in common::random_u32s(8, 1 << 14) {
        for &(begin, end) in &[(0u32, 10u32), (5, 15), (100, 101), (1000, 1_000_000)] {
            let r = uint_to_range_between(v, begin, end);
            assert!(r >= begin && r < end);
            assert_eq!(r, uint_to_range(v, end - begin) + begin);
        }
    }
}

#[test]
fn between_with_unit_width_returns_begin() {
    for v in common::EDGE_U32 {
        for begin in [0u32, 1, 42, 65535, u32::MAX - 1] {
            assert_eq!(uint_to_range_between(v, begin, begin + 1), begin);
        }
    }
}

#[test]
fn between_extremes() {
    assert_eq!(uint_to_range_between(0, 7, 20), 7);
    assert_eq!(uint_to_range_between(u32::MAX, 7, 20), 19);
    assert_eq!(uint_to_range_between(0x8000_0000, 0, 2), 1);
    assert_eq!(uint_to_range_between(0x7fff_ffff, 0, 2), 0);
}

#[test]
fn between_covers_the_full_u32_span() {
    assert_eq!(uint_to_range_between(0, 0, u32::MAX), 0);
    assert_eq!(uint_to_range_between(u32::MAX, 0, u32::MAX), u32::MAX - 1);
    assert_eq!(uint_to_range_between(u32::MAX, 1, u32::MAX), u32::MAX - 1);
}

#[test]
fn functions_are_usable_in_const_context() {
    const A: u32 = uint_to_range(u32::MAX, 10);
    const B: u32 = uint_to_range_between(0, 3, 9);
    assert_eq!(A, 9);
    assert_eq!(B, 3);
}

#[test]
fn preserves_ordering_of_stratified_qmc_samples() {
    // High bits are preserved: strata of the input map to contiguous buckets.
    let range = 10u32;
    let mut last = 0;
    for i in 0..1000u32 {
        let v = i.wrapping_mul(0x0041_8937); // ~ i * 2^32 / 1000
        let r = uint_to_range(v, range);
        assert!(r >= last);
        last = r;
    }
}
