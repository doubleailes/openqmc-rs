//! Property tests for `openqmc::rotate` (oqmc/rotate.h).

mod common;

use openqmc::rotate::{rotate_bits, rotate_bytes};

/// The upstream C++ formula, reproduced with wrapping shifts.
fn upstream_rotate(value: u32, distance: u32) -> u32 {
    (value >> (distance & 31)) | value.wrapping_shl(distance.wrapping_neg() & 31)
}

#[test]
fn rotating_by_zero_is_identity() {
    for v in common::EDGE_U32 {
        assert_eq!(rotate_bits(v, 0), v);
        assert_eq!(rotate_bytes(v, 0), v);
    }
}

#[test]
fn rotating_by_32_bits_is_identity() {
    for v in common::EDGE_U32
        .iter()
        .copied()
        .chain(common::random_u32s(1, 1024))
    {
        assert_eq!(rotate_bits(v, 32), v);
        assert_eq!(rotate_bits(v, 64), v);
        assert_eq!(rotate_bits(v, 96), v);
    }
}

#[test]
fn rotating_by_four_bytes_is_identity() {
    for v in common::EDGE_U32
        .iter()
        .copied()
        .chain(common::random_u32s(2, 1024))
    {
        assert_eq!(rotate_bytes(v, 4), v);
        assert_eq!(rotate_bytes(v, 8), v);
        assert_eq!(rotate_bytes(v, -4), v);
    }
}

#[test]
fn distance_is_reduced_modulo_32() {
    for v in common::random_u32s(3, 512) {
        for d in 0..256u32 {
            assert_eq!(rotate_bits(v, d), rotate_bits(v, d % 32));
        }
        for d in [u32::MAX, u32::MAX - 31, 1 << 31, (1 << 31) + 5] {
            assert_eq!(rotate_bits(v, d), rotate_bits(v, d % 32));
        }
    }
}

#[test]
fn matches_the_upstream_shift_or_formula() {
    for v in common::random_u32s(4, 4096) {
        for d in 0..64u32 {
            assert_eq!(rotate_bits(v, d), upstream_rotate(v, d), "{v:#x} by {d}");
        }
    }
    for v in common::EDGE_U32 {
        for d in common::EDGE_U32 {
            assert_eq!(rotate_bits(v, d), upstream_rotate(v, d), "{v:#x} by {d}");
        }
    }
}

#[test]
fn is_a_right_rotation() {
    assert_eq!(rotate_bits(0b1, 1), 0x8000_0000);
    assert_eq!(rotate_bits(0b10, 1), 0b1);
    assert_eq!(rotate_bits(0x8000_0000, 31), 1);
    assert_eq!(rotate_bits(0x0000_00ff, 4), 0xf000_000f);
    assert_eq!(rotate_bits(0x1234_5678, 8), 0x7812_3456);
    assert_eq!(rotate_bits(0x1234_5678, 16), 0x5678_1234);
}

#[test]
fn rotations_compose_additively() {
    for v in common::random_u32s(5, 1024) {
        for a in 0..32u32 {
            for b in [0u32, 1, 7, 8, 15, 16, 31] {
                assert_eq!(rotate_bits(rotate_bits(v, a), b), rotate_bits(v, a + b));
            }
        }
    }
}

#[test]
fn rotation_is_invertible() {
    for v in common::random_u32s(6, 1024) {
        for d in 0..32u32 {
            assert_eq!(rotate_bits(rotate_bits(v, d), 32 - d), v);
            assert_eq!(rotate_bits(v, d).rotate_left(d), v);
        }
    }
}

#[test]
fn rotation_preserves_popcount() {
    for v in common::random_u32s(7, 4096) {
        for d in [1u32, 3, 8, 13, 24, 31] {
            assert_eq!(rotate_bits(v, d).count_ones(), v.count_ones());
        }
    }
}

#[test]
fn rotation_is_a_bijection_for_every_distance() {
    // Two distinct inputs never collide (checked on low-16-bit domain).
    for d in 0..32u32 {
        common::assert_low_bits_bijection(12, &format!("rotate by {d}"), |i| {
            // Rotate then rotate back keeps identity; also check raw injectivity
            // over a window by undoing the rotation.
            rotate_bits(rotate_bits(i, d), (32 - d) % 32)
        });
    }
}

#[test]
fn all_ones_and_zero_are_fixed_points() {
    for d in 0..64u32 {
        assert_eq!(rotate_bits(0, d), 0);
        assert_eq!(rotate_bits(u32::MAX, d), u32::MAX);
        assert_eq!(
            rotate_bits(0xaaaa_aaaa, d),
            if d % 2 == 0 { 0xaaaa_aaaa } else { 0x5555_5555 }
        );
    }
}

#[test]
fn byte_rotation_moves_whole_bytes() {
    assert_eq!(rotate_bytes(0x1234_5678, 1), 0x7812_3456);
    assert_eq!(rotate_bytes(0x1234_5678, 2), 0x5678_1234);
    assert_eq!(rotate_bytes(0x1234_5678, 3), 0x3456_7812);
    assert_eq!(rotate_bytes(0x1234_5678, 4), 0x1234_5678);
    assert_eq!(rotate_bytes(0x0000_00ff, 1), 0xff00_0000);
}

#[test]
fn byte_rotation_matches_little_endian_byte_shuffle() {
    for v in common::random_u32s(8, 4096) {
        let bytes = v.to_le_bytes();
        for d in -9..=9i32 {
            let mut out = [0u8; 4];
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = bytes[(i as i32 + d).rem_euclid(4) as usize];
            }
            assert_eq!(rotate_bytes(v, d), u32::from_le_bytes(out), "{v:#x} by {d}");
        }
    }
}

#[test]
fn negative_byte_distances_rotate_the_other_way() {
    for v in common::random_u32s(9, 1024) {
        assert_eq!(rotate_bytes(v, -1), rotate_bytes(v, 3));
        assert_eq!(rotate_bytes(v, -2), rotate_bytes(v, 2));
        assert_eq!(rotate_bytes(v, -3), rotate_bytes(v, 1));
        assert_eq!(rotate_bytes(v, -1), v.rotate_left(8));
    }
}

#[test]
fn byte_distance_wraps_with_period_four() {
    for v in common::random_u32s(10, 256) {
        for d in -40..40i32 {
            assert_eq!(rotate_bytes(v, d), rotate_bytes(v, d.rem_euclid(4)));
        }
    }
}

#[test]
fn byte_rotation_equals_bit_rotation_by_eight_times_distance() {
    for v in common::random_u32s(11, 1024) {
        for d in 0..8i32 {
            assert_eq!(rotate_bytes(v, d), rotate_bits(v, d as u32 * 8));
        }
    }
}

#[test]
fn four_byte_rotations_are_distinct_for_generic_values() {
    // The per-dimension seeds derived in owen/lookup rely on rotate_bytes(seed,
    // 0..4) producing four different seeds for a generic hash.
    for v in common::random_u32s(12, 4096) {
        let rots = [
            rotate_bytes(v, 0),
            rotate_bytes(v, 1),
            rotate_bytes(v, 2),
            rotate_bytes(v, 3),
        ];
        let all_bytes_equal = v.to_le_bytes().windows(2).all(|w| w[0] == w[1]);
        let two_periodic =
            v.to_le_bytes()[0] == v.to_le_bytes()[2] && v.to_le_bytes()[1] == v.to_le_bytes()[3];
        if !all_bytes_equal && !two_periodic {
            common::assert_all_distinct(&rots, &format!("rotations of {v:#x}"));
        }
    }
}

#[test]
fn functions_are_usable_in_const_context() {
    const A: u32 = rotate_bits(1, 1);
    const B: u32 = rotate_bytes(0x0000_00ff, 1);
    assert_eq!(A, 0x8000_0000);
    assert_eq!(B, 0xff00_0000);
}
