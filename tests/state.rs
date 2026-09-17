//! Property tests for `openqmc::state` (oqmc/state.h) — the shared 8-byte state
//! and its domain-tree operations.

mod common;

use openqmc::encode::{EncodeKey, encode_bits16};
use openqmc::pcg;
use openqmc::state::{
    MAX_INDEX_BIT_SIZE, MAX_INDEX_SIZE, SPATIAL_ENCODE_BITS_X, SPATIAL_ENCODE_BITS_Y, State64Bit,
    compute_index_id, compute_index_key,
};

#[test]
fn constants_are_consistent() {
    assert_eq!(MAX_INDEX_BIT_SIZE, 16);
    assert_eq!(MAX_INDEX_SIZE, 1 << MAX_INDEX_BIT_SIZE);
    assert_eq!(SPATIAL_ENCODE_BITS_X + SPATIAL_ENCODE_BITS_Y, 16);
}

#[test]
fn state_is_eight_bytes_copy_eq_debug() {
    assert_eq!(std::mem::size_of::<State64Bit>(), 8);
    let s = State64Bit::new(1, 2, 3, 4);
    let c = s;
    assert_eq!(s, c);
    assert!(format!("{s:?}").contains("State64Bit"));
}

#[test]
fn index_key_and_id_split_the_index() {
    for i in [0i32, 1, 65535, 65536, 65537, 1 << 20, i32::MAX] {
        assert_eq!((compute_index_key(i) << 16) | compute_index_id(i), i);
        assert!(compute_index_id(i) < 65536);
        assert!(compute_index_id(i) >= 0);
        assert_eq!(compute_index_key(i), i / 65536);
    }
}

#[test]
fn index_id_is_periodic_and_key_is_a_step_function() {
    for base in [0i32, 1, 12345] {
        for k in 0..8 {
            let i = base + k * 65536;
            assert_eq!(compute_index_id(i), base);
            assert_eq!(compute_index_key(i), k);
        }
    }
}

#[test]
fn index_key_id_are_const() {
    const K: i32 = compute_index_key(0x0007_0003);
    const I: i32 = compute_index_id(0x0007_0003);
    assert_eq!(K, 7);
    assert_eq!(I, 3);
}

#[test]
fn pattern_id_is_init_seed_of_frame_plus_index_key() {
    for frame in [0i32, 1, 7, -1, 1000] {
        for index in [0i32, 5, 65535, 65536, 1 << 20] {
            let s = State64Bit::new(0, 0, frame, index);
            assert_eq!(
                s.pattern_id,
                pcg::init_seed((frame + compute_index_key(index)) as u32)
            );
        }
    }
}

#[test]
fn sample_id_is_the_low_16_bits_of_the_index() {
    for index in [0i32, 1, 255, 65535, 65536, 65537, 1 << 20, i32::MAX] {
        let s = State64Bit::new(0, 0, 0, index);
        assert_eq!(s.sample_id as i32, index & 0xffff);
    }
}

#[test]
fn pixel_id_is_the_8_8_encoding_of_the_pixel() {
    for x in [0i32, 1, 255, 256, -1, 1000] {
        for y in [0i32, 3, 255, 256, -7, 999] {
            let s = State64Bit::new(x, y, 0, 0);
            assert_eq!(
                s.pixel_id,
                encode_bits16::<8, 8, 0>(EncodeKey { x, y, z: 0 })
            );
            assert_eq!(s.pixel_id as i32, (x & 0xff) | ((y & 0xff) << 8));
        }
    }
}

#[test]
fn pixels_tile_with_period_256() {
    for x in 0..256 {
        let a = State64Bit::new(x, 5, 0, 0);
        let b = State64Bit::new(x + 256, 5, 0, 0);
        let c = State64Bit::new(x - 256, 5 + 512, 0, 0);
        assert_eq!(a, b);
        assert_eq!(a, c);
    }
}

#[test]
fn pixel_only_affects_pixel_id() {
    let base = State64Bit::new(0, 0, 3, 9);
    for x in 0..256 {
        for y in (0..256).step_by(17) {
            let s = State64Bit::new(x, y, 3, 9);
            assert_eq!(s.pattern_id, base.pattern_id);
            assert_eq!(s.sample_id, base.sample_id);
        }
    }
}

#[test]
fn all_65536_pixels_have_distinct_pixel_ids() {
    let ids: Vec<u16> = (0..256)
        .flat_map(|y| (0..256).map(move |x| State64Bit::new(x, y, 0, 0).pixel_id))
        .collect();
    common::assert_all_distinct(&ids, "pixel ids");
}

#[test]
fn frame_only_affects_pattern_id() {
    let base = State64Bit::new(4, 5, 0, 9);
    let mut patterns = Vec::new();
    for frame in 0..256 {
        let s = State64Bit::new(4, 5, frame, 9);
        assert_eq!(s.sample_id, base.sample_id);
        assert_eq!(s.pixel_id, base.pixel_id);
        patterns.push(s.pattern_id);
    }
    common::assert_all_distinct(&patterns, "pattern ids across frames");
}

#[test]
fn consecutive_frames_differ_by_one_in_pattern_id() {
    // init_seed is an additive seed, so frames map to consecutive raw states.
    for frame in [0i32, 10, 1000] {
        let a = State64Bit::new(0, 0, frame, 0);
        let b = State64Bit::new(0, 0, frame + 1, 0);
        assert_eq!(b.pattern_id.wrapping_sub(a.pattern_id), 1);
    }
}

#[test]
fn frame_and_index_key_alias_as_documented_upstream() {
    // frame f with index i + 2^16 shares its pattern with frame f + 1 index i.
    for f in [0i32, 3, 100] {
        for i in [0i32, 7, 65535] {
            let a = State64Bit::new(1, 2, f, i + 65536);
            let b = State64Bit::new(1, 2, f + 1, i);
            assert_eq!(a, b);
        }
    }
}

#[test]
fn indices_below_2_16_share_a_pattern() {
    let base = State64Bit::new(1, 2, 3, 0);
    for i in (0..65536).step_by(257) {
        let s = State64Bit::new(1, 2, 3, i);
        assert_eq!(s.pattern_id, base.pattern_id);
        assert_eq!(s.pixel_id, base.pixel_id);
        assert_eq!(s.sample_id as i32, i);
    }
}

#[test]
fn crossing_2_16_changes_the_pattern() {
    let a = State64Bit::new(1, 2, 3, 65535);
    let b = State64Bit::new(1, 2, 3, 65536);
    assert_ne!(a.pattern_id, b.pattern_id);
    assert_eq!(b.sample_id, 0);
    assert_eq!(a.sample_id, 65535);
}

#[test]
fn max_index_is_accepted() {
    let s = State64Bit::new(0, 0, 0, i32::MAX);
    assert_eq!(s.sample_id, 0xffff);
    assert_eq!(s.pattern_id, pcg::init_seed(0x7fff));
}

#[test]
fn negative_pixels_and_frames_are_accepted() {
    let s = State64Bit::new(-1, -1, -1, 0);
    assert_eq!(s.pixel_id, 0xffff);
    assert_eq!(s.pattern_id, pcg::init_seed(u32::MAX));
}

#[test]
fn pixel_decorrelate_is_new_domain_of_pixel_id() {
    for x in [0i32, 7, 255] {
        for y in [0i32, 1, 128] {
            let s = State64Bit::new(x, y, 2, 3);
            assert_eq!(s.pixel_decorrelate(), s.new_domain(s.pixel_id as i32));
        }
    }
}

#[test]
fn pixel_decorrelate_preserves_sample_and_pixel_ids() {
    for x in 0..64 {
        let s = State64Bit::new(x, x * 3, 1, 77);
        let d = s.pixel_decorrelate();
        assert_eq!(d.sample_id, s.sample_id);
        assert_eq!(d.pixel_id, s.pixel_id);
        assert_ne!(d.pattern_id, s.pattern_id);
    }
}

#[test]
fn pixel_decorrelate_makes_all_pixels_distinct() {
    let patterns: Vec<u32> = (0..256)
        .flat_map(|y| {
            (0..256).map(move |x| State64Bit::new(x, y, 0, 0).pixel_decorrelate().pattern_id)
        })
        .collect();
    common::assert_all_distinct(&patterns, "decorrelated patterns");
}

#[test]
fn pixel_decorrelate_is_idempotent_only_in_ids() {
    let s = State64Bit::new(3, 4, 5, 6);
    let once = s.pixel_decorrelate();
    let twice = once.pixel_decorrelate();
    assert_ne!(once.pattern_id, twice.pattern_id);
    assert_eq!(once.sample_id, twice.sample_id);
    assert_eq!(once.pixel_id, twice.pixel_id);
}

#[test]
fn new_domain_is_a_pcg_transition_of_pattern_plus_key() {
    let s = State64Bit::new(3, 4, 5, 6);
    for key in common::EDGE_I32 {
        let d = s.new_domain(key);
        assert_eq!(
            d.pattern_id,
            pcg::state_transition(s.pattern_id.wrapping_add(key as u32))
        );
        assert_eq!(d.sample_id, s.sample_id);
        assert_eq!(d.pixel_id, s.pixel_id);
    }
}

#[test]
fn new_domain_keys_are_distinct_over_a_wide_range() {
    let s = State64Bit::new(3, 4, 5, 6);
    let patterns: Vec<u32> = (-5000..5000).map(|k| s.new_domain(k).pattern_id).collect();
    common::assert_all_distinct(&patterns, "domain patterns");
}

#[test]
fn new_domain_never_returns_to_the_parent_pattern() {
    let s = State64Bit::new(3, 4, 5, 6);
    for k in -4096..4096 {
        assert_ne!(s.new_domain(k).pattern_id, s.pattern_id, "key {k}");
    }
}

#[test]
fn new_domain_is_injective_in_the_parent_pattern() {
    // Distinct parent patterns give distinct children for the same key.
    let children: Vec<u32> = (0..4096)
        .map(|f| State64Bit::new(0, 0, f, 0).new_domain(7).pattern_id)
        .collect();
    common::assert_all_distinct(&children, "children of distinct parents");
}

#[test]
fn new_domain_keys_wrap_at_32_bits() {
    // key as u32 wraps, so key and key + 2^32 are the same key; i32 range only
    // lets us check the two's complement equivalence.
    let s = State64Bit::new(3, 4, 5, 6);
    assert_eq!(
        s.new_domain(-1).pattern_id,
        pcg::state_transition(s.pattern_id.wrapping_add(u32::MAX))
    );
    assert_eq!(
        s.new_domain(i32::MIN).pattern_id,
        pcg::state_transition(s.pattern_id.wrapping_add(0x8000_0000))
    );
}

#[test]
fn new_domain_chain_order_matters() {
    let s = State64Bit::new(3, 4, 5, 6);
    assert_ne!(s.new_domain(1).new_domain(2), s.new_domain(2).new_domain(1));
}

#[test]
fn new_domain_depth_produces_distinct_patterns() {
    let mut s = State64Bit::new(3, 4, 5, 6);
    let mut seen = vec![s.pattern_id];
    for _ in 0..4096 {
        s = s.new_domain(0);
        seen.push(s.pattern_id);
    }
    common::assert_all_distinct(&seen, "chain of new_domain(0)");
}

#[test]
fn new_domain_split_combines_the_index() {
    for index in [0i32, 1, 100, 65535] {
        let s = State64Bit::new(1, 2, 3, index);
        for size in [1i32, 2, 4, 16, 1000] {
            for sub in 0..size.min(8) {
                let combined = index * size + sub;
                let d = s.new_domain_split(9, size, sub);
                let expected = s.new_domain(9).new_domain(compute_index_key(combined));
                assert_eq!(d.pattern_id, expected.pattern_id);
                assert_eq!(d.sample_id as i32, compute_index_id(combined));
                assert_eq!(d.pixel_id, s.pixel_id);
            }
        }
    }
}

#[test]
fn new_domain_split_of_size_one_index_zero_keeps_the_sample_id() {
    for index in [0i32, 1, 4096, 65535] {
        let s = State64Bit::new(1, 2, 3, index);
        let d = s.new_domain_split(5, 1, 0);
        assert_eq!(d.sample_id, s.sample_id);
        assert_eq!(d.pattern_id, s.new_domain(5).new_domain(0).pattern_id);
    }
}

#[test]
fn new_domain_split_sub_indices_are_distinct_states() {
    let s = State64Bit::new(1, 2, 3, 7);
    let size = 64;
    let states: Vec<(u32, u16)> = (0..size)
        .map(|i| {
            let d = s.new_domain_split(2, size, i);
            (d.pattern_id, d.sample_id)
        })
        .collect();
    common::assert_all_distinct(&states, "split states");
}

#[test]
fn new_domain_split_sample_ids_tile_without_gaps_within_a_pattern() {
    // Splitting every base index i < 2^16 / size by `size` covers every sample
    // id exactly once with a single pattern id.
    let size = 16i32;
    let mut seen = vec![false; 65536];
    let mut pattern = None;
    for i in 0..(65536 / size) {
        let s = State64Bit::new(1, 2, 3, i);
        for j in 0..size {
            let d = s.new_domain_split(4, size, j);
            match pattern {
                None => pattern = Some(d.pattern_id),
                Some(p) => assert_eq!(p, d.pattern_id),
            }
            assert!(!seen[d.sample_id as usize]);
            seen[d.sample_id as usize] = true;
        }
    }
    assert!(seen.iter().all(|&b| b));
}

#[test]
fn new_domain_split_overflow_moves_into_a_new_pattern() {
    let s = State64Bit::new(1, 2, 3, 65535);
    let a = s.new_domain_split(4, 2, 0); // combined = 131070 -> key 1, id 65534
    let b = s.new_domain_split(4, 2, 1); // combined = 131071 -> key 1, id 65535
    assert_eq!(a.pattern_id, b.pattern_id);
    assert_eq!(a.sample_id, 65534);
    assert_eq!(b.sample_id, 65535);
    let small = State64Bit::new(1, 2, 3, 0).new_domain_split(4, 2, 0);
    assert_ne!(small.pattern_id, a.pattern_id);
}

#[test]
fn new_domain_distrib_replaces_the_sample_id() {
    for base_index in [0i32, 9, 65535] {
        let s = State64Bit::new(1, 2, 3, base_index);
        for index in [0i32, 1, 65535, 65536, 1 << 20] {
            let d = s.new_domain_distrib(6, index);
            let expected = s
                .new_domain(6)
                .new_domain(compute_index_key(index))
                .new_domain(s.sample_id as i32);
            assert_eq!(d.pattern_id, expected.pattern_id);
            assert_eq!(d.sample_id as i32, compute_index_id(index));
            assert_eq!(d.pixel_id, s.pixel_id);
        }
    }
}

#[test]
fn new_domain_distrib_patterns_depend_on_the_parent_sample_id() {
    let patterns: Vec<u32> = (0..4096)
        .map(|i| {
            State64Bit::new(1, 2, 3, i)
                .new_domain_distrib(6, 0)
                .pattern_id
        })
        .collect();
    common::assert_all_distinct(&patterns, "distrib patterns across parent samples");
}

#[test]
fn new_domain_distrib_shares_a_pattern_across_its_own_indices() {
    let s = State64Bit::new(1, 2, 3, 42);
    let base = s.new_domain_distrib(6, 0);
    for index in 0..65536 {
        let d = s.new_domain_distrib(6, index);
        assert_eq!(d.pattern_id, base.pattern_id);
        assert_eq!(d.sample_id as i32, index);
    }
    assert_ne!(s.new_domain_distrib(6, 65536).pattern_id, base.pattern_id);
}

#[test]
fn the_three_domain_kinds_never_collide() {
    let s = State64Bit::new(5, 7, 2, 3);
    let mut all = Vec::new();
    for k in 0..512 {
        all.push(s.new_domain(k).pattern_id);
        all.push(s.new_domain_distrib(k, 0).pattern_id);
        all.push(s.new_domain_split(k, 11, 0).pattern_id);
    }
    common::assert_all_distinct(&all, "domain kinds");
}

#[test]
fn draw_rnd_prefixes_are_consistent() {
    let s = State64Bit::new(5, 7, 2, 3);
    let four = s.draw_rnd::<4>();
    assert_eq!(&four[..3], &s.draw_rnd::<3>()[..]);
    assert_eq!(&four[..2], &s.draw_rnd::<2>()[..]);
    assert_eq!(&four[..1], &s.draw_rnd::<1>()[..]);
}

#[test]
fn draw_rnd_matches_a_pcg_stream_seeded_with_pattern_plus_sample() {
    for i in [0i32, 1, 999, 65535] {
        let s = State64Bit::new(5, 7, 2, i);
        let mut state = s.pattern_id.wrapping_add(s.sample_id as u32);
        let expected: [u32; 4] = std::array::from_fn(|_| pcg::rng(&mut state));
        assert_eq!(s.draw_rnd::<4>(), expected);
    }
}

#[test]
fn rng_matches_draw_rnd() {
    for i in [0i32, 1, 999, 65535] {
        let s = State64Bit::new(5, 7, 2, i);
        let mut r = s.rng();
        let block = s.draw_rnd::<4>();
        for &b in &block {
            assert_eq!(r.next_u32(), b);
        }
    }
}

#[test]
fn draw_rnd_is_independent_of_pixel_id() {
    // The state does not include pixel_id in its rnd seed; callers decorrelate.
    let a = State64Bit::new(0, 0, 2, 3);
    let b = State64Bit::new(200, 100, 2, 3);
    assert_eq!(a.draw_rnd::<4>(), b.draw_rnd::<4>());
    assert_ne!(
        a.pixel_decorrelate().draw_rnd::<4>(),
        b.pixel_decorrelate().draw_rnd::<4>()
    );
}

#[test]
fn draw_rnd_differs_across_sample_ids() {
    let outs: Vec<[u32; 4]> = (0..4096)
        .map(|i| State64Bit::new(0, 0, 2, i).draw_rnd::<4>())
        .collect();
    common::assert_all_distinct(&outs, "rnd across samples");
}

#[test]
fn draw_rnd_differs_across_domains() {
    let s = State64Bit::new(0, 0, 2, 3);
    let outs: Vec<[u32; 4]> = (0..4096).map(|k| s.new_domain(k).draw_rnd::<4>()).collect();
    common::assert_all_distinct(&outs, "rnd across domains");
}

#[test]
fn draw_rnd_is_uniform_over_many_samples() {
    let n = 1 << 16;
    let mut buckets = [0u32; 16];
    for i in 0..n {
        let v = State64Bit::new(0, 0, 0, i).draw_rnd::<1>()[0];
        buckets[(v >> 28) as usize] += 1;
    }
    let expected = n as f64 / 16.0;
    for (b, &c) in buckets.iter().enumerate() {
        assert!(
            (c as f64 - expected).abs() / expected < 0.05,
            "bucket {b}: {c}"
        );
    }
}

#[test]
fn draw_rnd_seed_collisions_are_deliberate_and_predictable() {
    // pattern + sample is the seed, so pattern+1 with sample-1 collides: this is
    // the upstream behaviour and why samplers pixel-decorrelate first.
    let a = State64Bit::new(0, 0, 0, 5);
    let b = State64Bit {
        pattern_id: a.pattern_id.wrapping_add(1),
        sample_id: 4,
        pixel_id: 0,
    };
    assert_eq!(a.draw_rnd::<4>(), b.draw_rnd::<4>());
}

#[test]
fn struct_fields_are_public_and_constructible() {
    let s = State64Bit {
        pattern_id: 1,
        sample_id: 2,
        pixel_id: 3,
    };
    assert_eq!(s.new_domain(0).sample_id, 2);
    assert_eq!(s.new_domain(0).pixel_id, 3);
    assert_eq!(s.new_domain(0).pattern_id, pcg::state_transition(1));
}
