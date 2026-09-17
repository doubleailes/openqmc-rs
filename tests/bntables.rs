//! Tests for `openqmc::bntables` (oqmc/bntables.h) — the bundled blue-noise
//! key/rank tables and the toroidal lookup.

mod common;

use openqmc::bntables::{self, SIZE, TableReturnValue, XBITS, YBITS, table_value};
use openqmc::encode::{EncodeKey, encode_bits16};

fn tables() -> [(&'static str, &'static [u32], &'static [u32]); 3] {
    [
        (
            "sobol",
            bntables::sobol::key_table(),
            bntables::sobol::rank_table(),
        ),
        (
            "lattice",
            bntables::lattice::key_table(),
            bntables::lattice::rank_table(),
        ),
        (
            "pmj",
            bntables::pmj::key_table(),
            bntables::pmj::rank_table(),
        ),
    ]
}

#[test]
fn constants_describe_a_256_by_256_table() {
    assert_eq!(XBITS, 8);
    assert_eq!(YBITS, 8);
    assert_eq!(SIZE, 65536);
    assert_eq!(SIZE, 1 << (XBITS + YBITS));
}

#[test]
fn every_table_has_full_length() {
    for (name, keys, ranks) in tables() {
        assert_eq!(keys.len(), SIZE, "{name} keys");
        assert_eq!(ranks.len(), SIZE, "{name} ranks");
    }
}

#[test]
fn tables_are_cached_and_return_the_same_slice() {
    assert!(std::ptr::eq(
        bntables::sobol::key_table(),
        bntables::sobol::key_table()
    ));
    assert!(std::ptr::eq(
        bntables::sobol::rank_table(),
        bntables::sobol::rank_table()
    ));
    assert!(std::ptr::eq(
        bntables::lattice::key_table(),
        bntables::lattice::key_table()
    ));
    assert!(std::ptr::eq(
        bntables::lattice::rank_table(),
        bntables::lattice::rank_table()
    ));
    assert!(std::ptr::eq(
        bntables::pmj::key_table(),
        bntables::pmj::key_table()
    ));
    assert!(std::ptr::eq(
        bntables::pmj::rank_table(),
        bntables::pmj::rank_table()
    ));
}

#[test]
fn every_key_table_holds_distinct_keys() {
    for (name, keys, _) in tables() {
        common::assert_all_distinct(keys, &format!("{name} keys"));
    }
}

#[test]
fn key_tables_share_the_same_key_set_in_different_orders() {
    let mut sorted: Vec<Vec<u32>> = tables()
        .iter()
        .map(|(_, k, _)| {
            let mut v = k.to_vec();
            v.sort_unstable();
            v
        })
        .collect();
    let last = sorted.pop().unwrap();
    assert!(sorted.iter().all(|s| *s == last), "key multisets differ");
    let [(_, a, _), (_, b, _), (_, c, _)] = tables();
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
}

#[test]
fn keys_are_spread_over_the_full_32_bit_range() {
    for (name, keys, _) in tables() {
        let or = keys.iter().fold(0u32, |a, &k| a | k);
        assert_eq!(or, u32::MAX, "{name}");
        let min = *keys.iter().min().unwrap();
        let max = *keys.iter().max().unwrap();
        assert!(min < 1 << 16, "{name} min {min:#x}");
        assert!(max > 0xffff_0000, "{name} max {max:#x}");
    }
}

#[test]
fn every_rank_fits_in_seven_bits() {
    for (name, _, ranks) in tables() {
        assert!(ranks.iter().all(|&r| r < 128), "{name}");
    }
}

#[test]
fn ranks_fit_in_the_16_bit_sample_index() {
    // Ranks are XORed into a u16 sample id; they must not disturb bits >= 16.
    for (name, _, ranks) in tables() {
        assert!(ranks.iter().all(|&r| r <= u16::MAX as u32), "{name}");
    }
}

#[test]
fn every_rank_value_appears() {
    for (name, _, ranks) in tables() {
        let mut seen = [false; 128];
        for &r in ranks {
            seen[r as usize] = true;
        }
        assert!(
            seen.iter().all(|&b| b),
            "{name}: some rank value never used"
        );
    }
}

#[test]
fn rank_tables_differ_between_samplers() {
    let [(_, _, a), (_, _, b), (_, _, c)] = tables();
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
}

#[test]
fn ranks_are_not_spatially_constant() {
    // Neighbouring pixels should mostly carry different ranks (blue noise).
    for (name, _, ranks) in tables() {
        let mut same = 0usize;
        for y in 0..256 {
            for x in 0..255 {
                if ranks[y * 256 + x] == ranks[y * 256 + x + 1] {
                    same += 1;
                }
            }
        }
        assert!(
            same < 256 * 255 / 10,
            "{name}: {same} equal horizontal neighbours"
        );
    }
}

#[test]
fn table_value_with_zero_shift_reads_the_pixel_directly() {
    for (name, keys, ranks) in tables() {
        for p in (0..=u16::MAX).step_by(13) {
            let t = table_value::<8, 8, 0>(p, 0, keys, ranks);
            assert_eq!(t.key, keys[p as usize], "{name} p={p}");
            assert_eq!(t.rank, ranks[p as usize], "{name} p={p}");
        }
    }
}

#[test]
fn table_value_shift_is_a_toroidal_translation() {
    let (_, keys, ranks) = tables()[0];
    for &(px, py) in &[(0i32, 0i32), (255, 255), (10, 200), (128, 3)] {
        for &(sx, sy) in &[
            (0i32, 0i32),
            (1, 0),
            (0, 1),
            (255, 255),
            (100, 77),
            (200, 200),
        ] {
            let pixel = encode_bits16::<8, 8, 0>(EncodeKey { x: px, y: py, z: 0 });
            let shift = encode_bits16::<8, 8, 0>(EncodeKey { x: sx, y: sy, z: 0 });
            let target = encode_bits16::<8, 8, 0>(EncodeKey {
                x: (px + sx) % 256,
                y: (py + sy) % 256,
                z: 0,
            });
            let t = table_value::<8, 8, 0>(pixel, shift, keys, ranks);
            assert_eq!(t.key, keys[target as usize]);
            assert_eq!(t.rank, ranks[target as usize]);
        }
    }
}

#[test]
fn table_value_over_all_pixels_is_a_permutation_of_the_table_for_any_shift() {
    let (_, keys, ranks) = tables()[1];
    for shift in [0u16, 1, 0x0100, 0xffff, 0x1234] {
        let mut got: Vec<u32> = (0..=u16::MAX)
            .map(|p| table_value::<8, 8, 0>(p, shift, keys, ranks).key)
            .collect();
        got.sort_unstable();
        let mut want = keys.to_vec();
        want.sort_unstable();
        assert_eq!(got, want, "shift {shift:#x}");
    }
}

#[test]
fn table_value_shift_composes_additively() {
    let (_, keys, ranks) = tables()[2];
    for p in (0..=u16::MAX).step_by(101) {
        for &(a, b) in &[(0x0001u16, 0x0100u16), (0x0a0a, 0x0505), (0xff00, 0x00ff)] {
            let ax = a & 0xff;
            let ay = a >> 8;
            let bx = b & 0xff;
            let by = b >> 8;
            let sum = ((ax + bx) & 0xff) | (((ay + by) & 0xff) << 8);
            let via_target = table_value::<8, 8, 0>(p, sum, keys, ranks);
            let once = table_value::<8, 8, 0>(p, a, keys, ranks);
            // Shift the pixel by `a` then by `b`: same as shifting by the sum.
            let px = p & 0xff;
            let py = p >> 8;
            let moved = ((px + ax) & 0xff) | (((py + ay) & 0xff) << 8);
            let twice = table_value::<8, 8, 0>(moved, b, keys, ranks);
            assert_eq!(twice.key, via_target.key);
            assert_eq!(twice.rank, via_target.rank);
            assert_eq!(once.key, keys[moved as usize]);
        }
    }
}

#[test]
fn table_value_works_with_other_layouts() {
    // Any layout is accepted; with 16/0/0 the shift is a plain 1D wrap.
    let (_, keys, ranks) = tables()[0];
    for p in [0u16, 1, 100, 65535] {
        for s in [0u16, 1, 65535] {
            let t = table_value::<16, 0, 0>(p, s, keys, ranks);
            let idx = p.wrapping_add(s) as usize;
            assert_eq!(t.key, keys[idx]);
            assert_eq!(t.rank, ranks[idx]);
        }
    }
}

#[test]
fn table_return_value_is_copy_and_debug() {
    let v = TableReturnValue { key: 1, rank: 2 };
    let w = v;
    assert_eq!(v.key, w.key);
    assert_eq!(v.rank, w.rank);
    let s = format!("{v:?}");
    assert!(s.contains("key") && s.contains("rank"));
}

#[test]
fn neighbouring_pixels_get_different_keys() {
    for (name, keys, ranks) in tables() {
        let mut same = 0;
        for p in 0..65535u16 {
            let a = table_value::<8, 8, 0>(p, 0, keys, ranks);
            let b = table_value::<8, 8, 0>(p + 1, 0, keys, ranks);
            if a.key == b.key {
                same += 1;
            }
        }
        assert_eq!(same, 0, "{name}");
    }
}
