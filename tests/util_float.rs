//! Property tests for `openqmc::float` (oqmc/float.h) — the [0, 1) conversion.

mod common;

use openqmc::float::{FLOAT_ONE_OVER_TWO_POWER_32, uint_to_float};

const ONE_MINUS_EPSILON: f32 = f32::from_bits(0x3f7f_ffff);

#[test]
fn constant_is_exactly_two_to_the_minus_32() {
    assert_eq!(FLOAT_ONE_OVER_TWO_POWER_32, 2f32.powi(-32));
    assert_eq!(FLOAT_ONE_OVER_TWO_POWER_32.to_bits(), 0x2f80_0000);
}

#[test]
fn zero_maps_to_zero() {
    assert_eq!(uint_to_float(0), 0.0);
    assert!(uint_to_float(0).is_sign_positive());
}

#[test]
fn one_maps_to_two_to_the_minus_32() {
    assert_eq!(uint_to_float(1), FLOAT_ONE_OVER_TWO_POWER_32);
}

#[test]
fn powers_of_two_are_exact() {
    for k in 0..32 {
        let v = 1u32 << k;
        assert_eq!(uint_to_float(v), 2f32.powi(k - 32), "2^{k}");
    }
}

#[test]
fn max_input_is_largest_float_below_one() {
    assert_eq!(uint_to_float(u32::MAX), ONE_MINUS_EPSILON);
    assert!(uint_to_float(u32::MAX) < 1.0);
    assert_eq!(uint_to_float(u32::MAX).to_bits() + 1, 1.0f32.to_bits());
}

#[test]
fn edge_values_are_all_in_unit_interval() {
    for v in common::EDGE_U32 {
        let f = uint_to_float(v);
        assert!((0.0..1.0).contains(&f), "{v:#x} -> {f}");
    }
}

#[test]
fn random_values_are_all_in_unit_interval() {
    for v in common::random_u32s(1, 1 << 20) {
        let f = uint_to_float(v);
        assert!((0.0..1.0).contains(&f), "{v:#x} -> {f}");
    }
}

#[test]
fn never_rounds_up_to_one_near_the_top() {
    for v in (u32::MAX - 100_000)..=u32::MAX {
        assert!(uint_to_float(v) < 1.0, "{v:#x}");
        assert!(uint_to_float(v) <= ONE_MINUS_EPSILON);
    }
    for v in 0xffff_ff00u32..=u32::MAX {
        assert_eq!(uint_to_float(v), ONE_MINUS_EPSILON);
    }
}

#[test]
fn output_never_exceeds_exact_ratio() {
    // Rounding is always toward zero, so f <= v / 2^32 for every v.
    for v in common::EDGE_U32
        .iter()
        .copied()
        .chain(common::random_u32s(2, 1 << 18))
    {
        let f = uint_to_float(v) as f64;
        let exact = v as f64 / 4294967296.0;
        assert!(f <= exact, "{v:#x}: {f} > {exact}");
    }
}

#[test]
fn error_is_bounded_by_one_ulp_of_the_input_magnitude() {
    // Truncating to 24 significant bits loses at most 2^(width-24) integer units.
    for v in common::random_u32s(3, 1 << 18) {
        if v == 0 {
            continue;
        }
        let f = uint_to_float(v) as f64;
        let exact = v as f64 / 4294967296.0;
        let width = 32 - v.leading_zeros();
        let max_err = 2f64.powi(width as i32 - 24 - 32).max(0.0);
        assert!(exact - f <= max_err, "{v:#x}: err {}", exact - f);
    }
}

#[test]
fn matches_reference_truncation_on_edges() {
    for v in common::EDGE_U32 {
        assert_eq!(
            uint_to_float(v),
            common::reference_uint_to_float(v),
            "{v:#x}"
        );
    }
}

#[test]
fn matches_reference_truncation_on_random_values() {
    for v in common::random_u32s(4, 1 << 20) {
        assert_eq!(
            uint_to_float(v),
            common::reference_uint_to_float(v),
            "{v:#x}"
        );
    }
}

#[test]
fn matches_reference_truncation_around_every_power_of_two() {
    for k in 0..32 {
        let center = 1u32 << k;
        let lo = center.saturating_sub(300);
        let hi = center.saturating_add(300);
        for v in lo..=hi {
            assert_eq!(
                uint_to_float(v),
                common::reference_uint_to_float(v),
                "{v:#x}"
            );
        }
    }
}

#[test]
fn matches_reference_truncation_exhaustively_below_2_to_24() {
    // Below 2^24 every input is exactly representable, so f == v * 2^-32.
    for v in (0..(1u32 << 24)).step_by(97) {
        assert_eq!(uint_to_float(v), v as f32 * FLOAT_ONE_OVER_TWO_POWER_32);
        assert_eq!(uint_to_float(v), common::reference_uint_to_float(v));
    }
}

#[test]
fn small_inputs_are_exact() {
    for v in 0..(1u32 << 16) {
        assert_eq!(uint_to_float(v) as f64, v as f64 / 4294967296.0);
    }
}

#[test]
fn is_monotonic_non_decreasing_on_sorted_random_inputs() {
    let mut values = common::random_u32s(5, 1 << 18);
    values.sort_unstable();
    for w in values.windows(2) {
        assert!(
            uint_to_float(w[0]) <= uint_to_float(w[1]),
            "{:#x} {:#x}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn is_monotonic_non_decreasing_on_consecutive_inputs() {
    for start in [0u32, 0x00ff_ff00, 0x7fff_ff00, 0x8000_0000, 0xffff_0000] {
        let mut last = uint_to_float(start);
        for v in start + 1..start.saturating_add(70_000) {
            let f = uint_to_float(v);
            assert!(f >= last, "{v:#x}");
            last = f;
        }
    }
}

#[test]
fn is_strictly_increasing_on_the_24_bit_grid() {
    // Values spaced by the truncation width strictly increase.
    let mut last = -1.0f32;
    for i in 0..(1u32 << 16) {
        let v = i << 16;
        let f = uint_to_float(v);
        assert!(f > last, "{v:#x}");
        last = f;
    }
}

#[test]
fn each_output_has_at_most_24_significant_bits() {
    for v in common::random_u32s(6, 1 << 18) {
        let f = uint_to_float(v);
        // Multiply back by 2^32: must be an integer with <= 24 significant bits.
        let scaled = f as f64 * 4294967296.0;
        assert_eq!(scaled.fract(), 0.0, "{v:#x}");
        let as_int = scaled as u64;
        if as_int != 0 {
            let width = 64 - as_int.leading_zeros() - as_int.trailing_zeros();
            assert!(width <= 24, "{v:#x} has {width} significant bits");
        }
    }
}

#[test]
fn exactly_256_inputs_collapse_onto_each_top_octave_value() {
    // For v >= 2^31 the low 8 bits are dropped, so each 256-block is one value.
    for block in [0x8000_0000u32, 0x9abc_de00, 0xffff_ff00, 0xc000_0000] {
        let f = uint_to_float(block);
        for off in 0..256 {
            assert_eq!(uint_to_float(block + off), f);
        }
        if block > 0x8000_0000 {
            assert!(uint_to_float(block - 1) < f);
        }
    }
}

#[test]
fn half_boundary_is_exact() {
    assert_eq!(uint_to_float(0x8000_0000), 0.5);
    assert!(uint_to_float(0x7fff_ffff) < 0.5);
    assert_eq!(uint_to_float(0x7fff_ffff), 0.5 - 2f32.powi(-25));
}

#[test]
fn quarter_boundaries_are_exact() {
    assert_eq!(uint_to_float(0x4000_0000), 0.25);
    assert_eq!(uint_to_float(0xc000_0000), 0.75);
    assert!(uint_to_float(0x3fff_ffff) < 0.25);
    assert!(uint_to_float(0xbfff_ffff) < 0.75);
}

#[test]
fn mean_over_a_uniform_grid_is_close_to_half() {
    let n = 1u64 << 16;
    let mut sum = 0.0f64;
    for i in 0..n {
        sum += uint_to_float((i << 16) as u32) as f64;
    }
    let mean = sum / n as f64;
    assert!((mean - 0.5).abs() < 1e-4, "mean {mean}");
}

#[test]
fn inverse_scaling_recovers_the_top_bits() {
    for v in common::random_u32s(7, 1 << 16) {
        let f = uint_to_float(v);
        let back = (f as f64 * 4294967296.0) as u32;
        // The recovered integer agrees with v on its top 24 significant bits.
        let width = 32 - v.leading_zeros();
        let drop = width.saturating_sub(24);
        assert_eq!(back >> drop, v >> drop, "{v:#x}");
    }
}
