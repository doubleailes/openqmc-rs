//! Exhaustive and property tests for `openqmc::encode` (oqmc/encode.h).

mod common;

use openqmc::encode::{EncodeKey, decode_bits16, encode_bits16};

fn key(x: i32, y: i32, z: i32) -> EncodeKey {
    EncodeKey { x, y, z }
}

#[test]
fn encode_key_is_copy_eq_debug() {
    let k = key(1, 2, 3);
    let copy = k;
    assert_eq!(k, copy);
    assert_ne!(k, key(3, 2, 1));
    assert!(format!("{k:?}").contains("EncodeKey"));
}

#[test]
fn zero_key_encodes_to_zero_for_every_layout() {
    assert_eq!(encode_bits16::<8, 8, 0>(key(0, 0, 0)), 0);
    assert_eq!(encode_bits16::<16, 0, 0>(key(0, 0, 0)), 0);
    assert_eq!(encode_bits16::<0, 16, 0>(key(0, 0, 0)), 0);
    assert_eq!(encode_bits16::<0, 0, 16>(key(0, 0, 0)), 0);
    assert_eq!(encode_bits16::<5, 5, 6>(key(0, 0, 0)), 0);
    assert_eq!(encode_bits16::<4, 4, 4>(key(0, 0, 0)), 0);
    assert_eq!(encode_bits16::<0, 0, 0>(key(0, 0, 0)), 0);
}

#[test]
fn zero_precision_layout_ignores_everything() {
    for x in common::EDGE_I32 {
        for y in common::EDGE_I32 {
            assert_eq!(encode_bits16::<0, 0, 0>(key(x, y, x ^ y)), 0);
        }
    }
    assert_eq!(decode_bits16::<0, 0, 0>(0xffff), key(0, 0, 0));
}

#[test]
fn x_is_stored_in_the_low_bits() {
    for x in 0..256 {
        assert_eq!(encode_bits16::<8, 8, 0>(key(x, 0, 0)), x as u16);
    }
}

#[test]
fn y_is_stored_above_x() {
    for y in 0..256 {
        assert_eq!(encode_bits16::<8, 8, 0>(key(0, y, 0)), (y as u16) << 8);
    }
}

#[test]
fn z_is_stored_above_y() {
    for z in 0..64 {
        assert_eq!(encode_bits16::<5, 5, 6>(key(0, 0, z)), (z as u16) << 10);
    }
}

#[test]
fn axes_are_independent_bit_fields() {
    for x in [0, 1, 17, 31] {
        for y in [0, 2, 19, 31] {
            for z in [0, 3, 33, 63] {
                let e = encode_bits16::<5, 5, 6>(key(x, y, z));
                let ex = encode_bits16::<5, 5, 6>(key(x, 0, 0));
                let ey = encode_bits16::<5, 5, 6>(key(0, y, 0));
                let ez = encode_bits16::<5, 5, 6>(key(0, 0, z));
                assert_eq!(e, ex | ey | ez);
                assert_eq!(ex & ey, 0);
                assert_eq!(ey & ez, 0);
                assert_eq!(ex & ez, 0);
            }
        }
    }
}

#[test]
fn exhaustive_round_trip_8_8_0() {
    for x in 0..256 {
        for y in 0..256 {
            let k = key(x, y, 0);
            let e = encode_bits16::<8, 8, 0>(k);
            assert_eq!(decode_bits16::<8, 8, 0>(e), k);
        }
    }
}

#[test]
fn exhaustive_round_trip_5_5_6() {
    for x in 0..32 {
        for y in 0..32 {
            for z in 0..64 {
                let k = key(x, y, z);
                let e = encode_bits16::<5, 5, 6>(k);
                assert_eq!(decode_bits16::<5, 5, 6>(e), k);
            }
        }
    }
}

#[test]
fn exhaustive_round_trip_4_4_4() {
    for x in 0..16 {
        for y in 0..16 {
            for z in 0..16 {
                let k = key(x, y, z);
                let e = encode_bits16::<4, 4, 4>(k);
                assert_eq!(decode_bits16::<4, 4, 4>(e), k);
                assert!(e < 1 << 12);
            }
        }
    }
}

#[test]
fn exhaustive_round_trip_16_0_0() {
    for x in 0..65536 {
        let e = encode_bits16::<16, 0, 0>(key(x, 0, 0));
        assert_eq!(e, x as u16);
        assert_eq!(decode_bits16::<16, 0, 0>(e), key(x, 0, 0));
    }
}

#[test]
fn exhaustive_round_trip_0_16_0() {
    for y in 0..65536 {
        let e = encode_bits16::<0, 16, 0>(key(0, y, 0));
        assert_eq!(e, y as u16);
        assert_eq!(decode_bits16::<0, 16, 0>(e), key(0, y, 0));
    }
}

#[test]
fn exhaustive_round_trip_0_0_16() {
    for z in 0..65536 {
        let e = encode_bits16::<0, 0, 16>(key(0, 0, z));
        assert_eq!(e, z as u16);
        assert_eq!(decode_bits16::<0, 0, 16>(e), key(0, 0, z));
    }
}

#[test]
fn decode_then_encode_is_identity_on_every_u16_for_full_layouts() {
    for v in 0..=u16::MAX {
        assert_eq!(encode_bits16::<8, 8, 0>(decode_bits16::<8, 8, 0>(v)), v);
        assert_eq!(encode_bits16::<5, 5, 6>(decode_bits16::<5, 5, 6>(v)), v);
        assert_eq!(encode_bits16::<16, 0, 0>(decode_bits16::<16, 0, 0>(v)), v);
        assert_eq!(encode_bits16::<1, 7, 8>(decode_bits16::<1, 7, 8>(v)), v);
    }
}

#[test]
fn decode_then_encode_clears_unused_high_bits_for_partial_layouts() {
    for v in 0..=u16::MAX {
        assert_eq!(
            encode_bits16::<4, 4, 4>(decode_bits16::<4, 4, 4>(v)),
            v & 0x0fff
        );
        assert_eq!(
            encode_bits16::<8, 0, 0>(decode_bits16::<8, 0, 0>(v)),
            v & 0x00ff
        );
    }
}

#[test]
fn encode_is_a_bijection_on_the_in_range_domain() {
    let mut seen = vec![false; 65536];
    for x in 0..256 {
        for y in 0..256 {
            let e = encode_bits16::<8, 8, 0>(key(x, y, 0)) as usize;
            assert!(!seen[e], "collision at ({x}, {y})");
            seen[e] = true;
        }
    }
    assert!(seen.iter().all(|&b| b));
}

#[test]
fn out_of_range_coordinates_are_masked_not_saturated() {
    assert_eq!(encode_bits16::<8, 8, 0>(key(256, 0, 0)), 0);
    assert_eq!(encode_bits16::<8, 8, 0>(key(257, 0, 0)), 1);
    assert_eq!(encode_bits16::<8, 8, 0>(key(511, 0, 0)), 255);
    assert_eq!(encode_bits16::<8, 8, 0>(key(0, 256, 0)), 0);
    assert_eq!(encode_bits16::<8, 8, 0>(key(0, 300, 0)), (300 - 256) << 8);
    assert_eq!(encode_bits16::<5, 5, 6>(key(0, 0, 64)), 0);
    assert_eq!(encode_bits16::<5, 5, 6>(key(0, 0, 65)), 1 << 10);
}

#[test]
fn coordinates_wrap_with_period_two_to_the_precision() {
    for x in -1000..1000 {
        for period_multiple in [-3, -1, 1, 2, 7] {
            let a = encode_bits16::<8, 8, 0>(key(x, 0, 0));
            let b = encode_bits16::<8, 8, 0>(key(x + 256 * period_multiple, 0, 0));
            assert_eq!(a, b, "x={x} k={period_multiple}");
        }
    }
}

#[test]
fn negative_coordinates_use_twos_complement_low_bits() {
    assert_eq!(encode_bits16::<8, 8, 0>(key(-1, 0, 0)), 0xff);
    assert_eq!(encode_bits16::<8, 8, 0>(key(-2, 0, 0)), 0xfe);
    assert_eq!(encode_bits16::<8, 8, 0>(key(-256, 0, 0)), 0x00);
    assert_eq!(encode_bits16::<8, 8, 0>(key(0, -1, 0)), 0xff00);
    assert_eq!(encode_bits16::<8, 8, 0>(key(-1, -1, 0)), 0xffff);
    assert_eq!(encode_bits16::<8, 8, 0>(key(i32::MIN, 0, 0)), 0);
    assert_eq!(encode_bits16::<8, 8, 0>(key(i32::MAX, 0, 0)), 0xff);
}

#[test]
fn negative_coordinate_round_trips_to_its_wrapped_value() {
    for x in -600..600 {
        for y in -600..600i32 {
            if (x + y) % 7 != 0 {
                continue;
            }
            let e = encode_bits16::<8, 8, 0>(key(x, y, 0));
            let d = decode_bits16::<8, 8, 0>(e);
            assert_eq!(d.x, x.rem_euclid(256));
            assert_eq!(d.y, y.rem_euclid(256));
            assert_eq!(d.z, 0);
        }
    }
}

#[test]
fn decoded_values_never_exceed_their_precision() {
    for v in 0..=u16::MAX {
        let d = decode_bits16::<5, 5, 6>(v);
        assert!((0..32).contains(&d.x));
        assert!((0..32).contains(&d.y));
        assert!((0..64).contains(&d.z));
        let d = decode_bits16::<8, 8, 0>(v);
        assert!((0..256).contains(&d.x));
        assert!((0..256).contains(&d.y));
        assert_eq!(d.z, 0);
    }
}

#[test]
fn toroidal_addition_before_encode_matches_wrapped_lookup() {
    // The blue-noise tables shift pixels toroidally by adding decoded offsets
    // before re-encoding; check that this equals per-axis modular addition.
    for &(px, py) in &[(0, 0), (255, 255), (17, 200), (128, 1)] {
        for &(sx, sy) in &[(0, 0), (1, 0), (0, 1), (255, 255), (100, 77)] {
            let e = encode_bits16::<8, 8, 0>(key(px + sx, py + sy, 0));
            let expected = encode_bits16::<8, 8, 0>(key((px + sx) % 256, (py + sy) % 256, 0));
            assert_eq!(e, expected);
        }
    }
}

#[test]
fn asymmetric_layouts_round_trip() {
    for x in 0..2 {
        for y in 0..128 {
            for z in 0..256 {
                let k = key(x, y, z);
                let e = encode_bits16::<1, 7, 8>(k);
                assert_eq!(decode_bits16::<1, 7, 8>(e), k);
            }
        }
    }
    for x in 0..1024 {
        for y in 0..64 {
            let k = key(x, y, 0);
            let e = encode_bits16::<10, 6, 0>(k);
            assert_eq!(decode_bits16::<10, 6, 0>(e), k);
        }
    }
}

#[test]
fn layouts_summing_to_less_than_16_leave_high_bits_zero() {
    for v in 0..=u16::MAX {
        let k = decode_bits16::<4, 4, 4>(v);
        assert!(encode_bits16::<4, 4, 4>(k) < 1 << 12);
        let k = decode_bits16::<3, 3, 3>(v);
        assert!(encode_bits16::<3, 3, 3>(k) < 1 << 9);
    }
}
