//! Port of `oqmc/rotate.h` — bit/byte rotation, used to extract fresh random
//! values from an existing hash or RNG number.

/// Rotate the bits of a 32-bit integer right by `distance` (wrapping every 32).
///
/// Upstream computes `value >> (distance & 31) | value << ((-distance) & 31)`,
/// which is exactly a right-rotation; `u32::rotate_right` reduces the distance
/// modulo 32 identically.
#[inline]
pub const fn rotate_bits(value: u32, distance: u32) -> u32 {
    value.rotate_right(distance)
}

/// Rotate the bytes of a 4-byte integer by `distance` (wrapping every 4).
#[inline]
pub const fn rotate_bytes(value: u32, distance: i32) -> u32 {
    rotate_bits(value, (distance * 8) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_bits_matches_the_upstream_formula() {
        for v in [0u32, 1, 0x8000_0000, 0x1234_5678, u32::MAX] {
            for d in 0..70u32 {
                let expected = (v >> (d & 31)) | v.wrapping_shl(d.wrapping_neg() & 31);
                assert_eq!(rotate_bits(v, d), expected, "{v:#x} by {d}");
            }
        }
    }

    #[test]
    fn rotate_bytes_moves_whole_bytes() {
        assert_eq!(rotate_bytes(0x1234_5678, 0), 0x1234_5678);
        assert_eq!(rotate_bytes(0x1234_5678, 1), 0x7812_3456);
        assert_eq!(rotate_bytes(0x1234_5678, 2), 0x5678_1234);
        assert_eq!(rotate_bytes(0x1234_5678, 3), 0x3456_7812);
        assert_eq!(rotate_bytes(0x1234_5678, 4), 0x1234_5678);
        assert_eq!(rotate_bytes(0x1234_5678, -1), 0x3456_7812);
    }

    #[test]
    fn four_byte_rotations_of_a_generic_seed_are_distinct() {
        let seed = 0x0102_0304u32;
        let r = [
            rotate_bytes(seed, 0),
            rotate_bytes(seed, 1),
            rotate_bytes(seed, 2),
            rotate_bytes(seed, 3),
        ];
        for a in 0..4 {
            for b in (a + 1)..4 {
                assert_ne!(r[a], r[b]);
            }
        }
    }

    #[test]
    fn functions_are_const() {
        const A: u32 = rotate_bits(2, 1);
        const B: u32 = rotate_bytes(0x100, 1);
        assert_eq!(A, 1);
        assert_eq!(B, 1);
    }
}
