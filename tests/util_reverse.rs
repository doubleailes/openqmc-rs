//! Property tests for `openqmc::reverse` (oqmc/reverse.h).

mod common;

use openqmc::reverse::{reverse_bits16, reverse_bits32};

/// Bit-by-bit reference reversal.
fn slow_reverse32(v: u32) -> u32 {
    let mut out = 0u32;
    for i in 0..32 {
        if v & (1 << i) != 0 {
            out |= 1 << (31 - i);
        }
    }
    out
}

fn slow_reverse16(v: u16) -> u16 {
    let mut out = 0u16;
    for i in 0..16 {
        if v & (1 << i) != 0 {
            out |= 1 << (15 - i);
        }
    }
    out
}

#[test]
fn reverse16_is_exhaustively_correct() {
    for v in 0..=u16::MAX {
        assert_eq!(reverse_bits16(v), slow_reverse16(v), "{v:#06x}");
    }
}

#[test]
fn reverse16_is_an_involution_exhaustively() {
    for v in 0..=u16::MAX {
        assert_eq!(reverse_bits16(reverse_bits16(v)), v);
    }
}

#[test]
fn reverse16_is_a_bijection() {
    let mut seen = vec![false; 65536];
    for v in 0..=u16::MAX {
        let r = reverse_bits16(v) as usize;
        assert!(!seen[r]);
        seen[r] = true;
    }
    assert!(seen.iter().all(|&b| b));
}

#[test]
fn reverse32_matches_reference_on_edges() {
    for v in common::EDGE_U32 {
        assert_eq!(reverse_bits32(v), slow_reverse32(v), "{v:#x}");
    }
}

#[test]
fn reverse32_matches_reference_on_random_values() {
    for v in common::random_u32s(1, 1 << 18) {
        assert_eq!(reverse_bits32(v), slow_reverse32(v), "{v:#x}");
    }
}

#[test]
fn reverse32_is_an_involution() {
    for v in common::EDGE_U32
        .iter()
        .copied()
        .chain(common::random_u32s(2, 1 << 16))
    {
        assert_eq!(reverse_bits32(reverse_bits32(v)), v);
    }
}

#[test]
fn reverse32_of_single_bits() {
    for k in 0..32 {
        assert_eq!(reverse_bits32(1 << k), 1 << (31 - k));
    }
}

#[test]
fn reverse16_of_single_bits() {
    for k in 0..16 {
        assert_eq!(reverse_bits16(1 << k), 1 << (15 - k));
    }
}

#[test]
fn reverse_preserves_popcount() {
    for v in common::random_u32s(3, 1 << 14) {
        assert_eq!(reverse_bits32(v).count_ones(), v.count_ones());
        assert_eq!(
            reverse_bits16(v as u16).count_ones(),
            (v as u16).count_ones()
        );
    }
}

#[test]
fn reverse_swaps_leading_and_trailing_zeros() {
    for v in common::random_u32s(4, 1 << 14) {
        if v == 0 {
            continue;
        }
        assert_eq!(reverse_bits32(v).leading_zeros(), v.trailing_zeros());
        assert_eq!(reverse_bits32(v).trailing_zeros(), v.leading_zeros());
    }
}

#[test]
fn reverse_is_xor_linear() {
    let values = common::random_u32s(5, 4096);
    for w in values.windows(2) {
        assert_eq!(
            reverse_bits32(w[0] ^ w[1]),
            reverse_bits32(w[0]) ^ reverse_bits32(w[1])
        );
        assert_eq!(
            reverse_bits16((w[0] ^ w[1]) as u16),
            reverse_bits16(w[0] as u16) ^ reverse_bits16(w[1] as u16)
        );
    }
}

#[test]
fn reverse_fixes_palindromes() {
    for v in [
        0u32,
        u32::MAX,
        0x8000_0001,
        0xf000_000f,
        0x0ff0_0ff0,
        0x9669_9669,
    ] {
        assert_eq!(reverse_bits32(v), v);
    }
    for v in [0u16, u16::MAX, 0x8001, 0xf00f, 0x9669] {
        assert_eq!(reverse_bits16(v), v);
    }
}

#[test]
fn reverse16_embeds_into_reverse32() {
    // reverse32 of a 16-bit value equals reverse16 shifted into the top half.
    for v in 0..=u16::MAX {
        assert_eq!(reverse_bits32(v as u32), (reverse_bits16(v) as u32) << 16);
    }
}

#[test]
fn reverse32_of_a_top_half_value_lands_in_the_bottom_half() {
    for v in (0..=u16::MAX).step_by(3) {
        assert_eq!(reverse_bits32((v as u32) << 16), reverse_bits16(v) as u32);
    }
}

#[test]
fn radical_inverse_of_a_prefix_is_stratified() {
    // The Van der Corput sequence: reversing 0..2^m fills the top-m-bit strata.
    for m in 0..=16u32 {
        let values: Vec<u32> = (0..(1u32 << m)).map(reverse_bits32).collect();
        common::assert_1d_stratified(&values, m, &format!("vdc m={m}"));
    }
}

#[test]
fn radical_inverse_of_an_aligned_block_is_stratified() {
    for m in 1..=10u32 {
        for block in [1u32, 2, 3, 7, 100, 1000] {
            let start = block << m;
            let values: Vec<u32> = (start..start + (1 << m)).map(reverse_bits32).collect();
            common::assert_1d_stratified(&values, m, &format!("vdc block {block} m={m}"));
        }
    }
}

#[test]
fn functions_are_usable_in_const_context() {
    const A: u32 = reverse_bits32(1);
    const B: u16 = reverse_bits16(1);
    assert_eq!(A, 0x8000_0000);
    assert_eq!(B, 0x8000);
}
