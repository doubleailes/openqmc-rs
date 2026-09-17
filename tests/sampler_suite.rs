//! Behavioural suite run against every one of the six samplers through the
//! public `Sampler` API. Each `sampler_suite!` invocation stamps out the same
//! set of tests for one sampler type.

mod common;

use openqmc::float::uint_to_float;
use openqmc::range::uint_to_range;
use openqmc::sampler::{Sampler, SamplerImpl};
use openqmc::state::State64Bit;
use openqmc::{
    LatticeBnSampler, LatticeSampler, PmjBnSampler, PmjSampler, SobolBnSampler, SobolSampler,
};

fn assert_traits<T: Copy + Send + Sync + std::fmt::Debug + 'static>() {}

/// Blocks drawn from distinct domains must be distinct when the whole 32-bit
/// pattern id reaches the sequence. The blue-noise samplers only see the
/// pattern through a 16-bit toroidal table shift, so there we allow (and
/// bound) birthday-paradox collisions.
fn assert_domains_distinct_for(blocks: &[[u32; 4]], pattern_bits: u32, what: &str) {
    if pattern_bits >= 32 {
        common::assert_all_distinct(blocks, what);
        return;
    }
    let n = blocks.len();
    let mut sorted = blocks.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let collisions = n - sorted.len();
    let expected = (n as f64) * (n as f64 - 1.0) / 2.0 / (1u64 << pattern_bits) as f64;
    assert!(
        (collisions as f64) <= 3.0 * expected + 8.0,
        "{what}: {collisions} collisions, expected about {expected:.1}"
    );
    assert!(
        sorted.len() * 20 >= n * 19,
        "{what}: fewer than 95% distinct"
    );
}

/// Integrate `f` over the unit square with the first two dimensions of domain
/// `0` of `n` consecutive sample indices at pixel (3, 5), frame 0.
fn integrate2d<T: SamplerImpl>(n: u32, f: impl Fn(f64, f64) -> f64) -> f64 {
    let mut acc = 0.0;
    for i in 0..n {
        let [x, y] = Sampler::<T>::new(3, 5, 0, i as i32)
            .new_domain(0)
            .draw_sample_f32::<2>();
        acc += f(x as f64, y as f64);
    }
    acc / n as f64
}

macro_rules! sampler_suite {
    ($modname:ident, $sampler:ty, decorrelated_state: $decorrelated:expr, pattern_bits: $pattern_bits:expr) => {
        mod $modname {
            use super::*;

            type S = $sampler;

            const PX: i32 = 12;
            const PY: i32 = 34;
            const FRAME: i32 = 2;
            const INDEX: i32 = 7;

            /// Bits of the domain pattern that reach the drawn sample values.
            const PATTERN_BITS: u32 = $pattern_bits;

            fn root() -> S {
                S::new(PX, PY, FRAME, INDEX)
            }

            fn assert_domains_distinct(blocks: &[[u32; 4]], what: &str) {
                assert_domains_distinct_for(blocks, PATTERN_BITS, what);
            }

            fn block_at(x: i32, y: i32, f: i32, i: i32, d: i32) -> [u32; 4] {
                S::new(x, y, f, i).new_domain(d).draw_sample::<4>()
            }

            #[test]
            fn is_copy_send_sync_static_debug() {
                assert_traits::<S>();
                let s = format!("{:?}", root());
                assert!(!s.is_empty());
            }

            #[test]
            fn is_a_small_value_type() {
                assert!(std::mem::size_of::<S>() <= 64);
                assert!(std::mem::size_of::<S>() >= 8);
            }

            #[test]
            fn construction_is_deterministic() {
                let a = root().new_domain(1);
                let b = root().new_domain(1);
                assert_eq!(a.draw_sample::<4>(), b.draw_sample::<4>());
                assert_eq!(a.draw_rnd::<4>(), b.draw_rnd::<4>());
                assert_eq!(a.rng().next_u32(), b.rng().next_u32());
            }

            #[test]
            fn copies_draw_identically() {
                let a = root().new_domain(3);
                let b = a;
                assert_eq!(a.draw_sample::<4>(), b.draw_sample::<4>());
                assert_eq!(a.draw_rnd::<4>(), b.draw_rnd::<4>());
            }

            #[test]
            fn drawing_does_not_mutate_the_sampler() {
                let a = root().new_domain(3);
                let first = a.draw_sample::<4>();
                let _ = a.draw_rnd::<4>();
                let _ = a.rng().next_u32();
                let _ = a.new_domain(9).draw_sample::<4>();
                assert_eq!(a.draw_sample::<4>(), first);
            }

            #[test]
            fn draw_sample_prefixes_are_consistent() {
                for d in 0..64 {
                    let s = root().new_domain(d);
                    let four = s.draw_sample::<4>();
                    assert_eq!(&four[..3], &s.draw_sample::<3>()[..]);
                    assert_eq!(&four[..2], &s.draw_sample::<2>()[..]);
                    assert_eq!(&four[..1], &s.draw_sample::<1>()[..]);
                }
            }

            #[test]
            fn draw_rnd_prefixes_are_consistent() {
                for d in 0..64 {
                    let s = root().new_domain(d);
                    let four = s.draw_rnd::<4>();
                    assert_eq!(&four[..3], &s.draw_rnd::<3>()[..]);
                    assert_eq!(&four[..2], &s.draw_rnd::<2>()[..]);
                    assert_eq!(&four[..1], &s.draw_rnd::<1>()[..]);
                }
            }

            #[test]
            fn draw_sample_f32_is_uint_to_float_of_draw_sample() {
                for d in 0..64 {
                    let s = root().new_domain(d);
                    let ints = s.draw_sample::<4>();
                    let floats = s.draw_sample_f32::<4>();
                    for k in 0..4 {
                        assert_eq!(floats[k], uint_to_float(ints[k]));
                    }
                }
            }

            #[test]
            fn draw_rnd_f32_is_uint_to_float_of_draw_rnd() {
                for d in 0..64 {
                    let s = root().new_domain(d);
                    let ints = s.draw_rnd::<4>();
                    let floats = s.draw_rnd_f32::<4>();
                    for k in 0..4 {
                        assert_eq!(floats[k], uint_to_float(ints[k]));
                    }
                }
            }

            #[test]
            fn draw_sample_range_is_uint_to_range_of_draw_sample() {
                for d in 0..32 {
                    let s = root().new_domain(d);
                    let ints = s.draw_sample::<4>();
                    for range in [1u32, 2, 3, 10, 256, 65537, u32::MAX] {
                        let r = s.draw_sample_range::<4>(range);
                        for k in 0..4 {
                            assert_eq!(r[k], uint_to_range(ints[k], range));
                            assert!(r[k] < range);
                        }
                    }
                }
            }

            #[test]
            fn draw_rnd_range_is_uint_to_range_of_draw_rnd() {
                for d in 0..32 {
                    let s = root().new_domain(d);
                    let ints = s.draw_rnd::<4>();
                    for range in [1u32, 2, 3, 10, 256, 65537, u32::MAX] {
                        let r = s.draw_rnd_range::<4>(range);
                        for k in 0..4 {
                            assert_eq!(r[k], uint_to_range(ints[k], range));
                            assert!(r[k] < range);
                        }
                    }
                }
            }

            #[test]
            fn range_one_always_draws_zero() {
                for d in 0..64 {
                    let s = root().new_domain(d);
                    assert_eq!(s.draw_sample_range::<4>(1), [0; 4]);
                    assert_eq!(s.draw_rnd_range::<4>(1), [0; 4]);
                }
            }

            #[test]
            fn float_draws_are_in_the_unit_interval() {
                for i in 0..512 {
                    for d in 0..4 {
                        let s = S::new(PX, PY, FRAME, i).new_domain(d);
                        for v in s
                            .draw_sample_f32::<4>()
                            .iter()
                            .chain(s.draw_rnd_f32::<4>().iter())
                        {
                            assert!((0.0..1.0).contains(v), "i={i} d={d}: {v}");
                        }
                    }
                }
            }

            #[test]
            fn range_draws_hit_every_bucket() {
                let range = 8u32;
                let mut seen = vec![false; range as usize];
                for i in 0..256 {
                    let r = S::new(PX, PY, FRAME, i)
                        .new_domain(0)
                        .draw_sample_range::<1>(range);
                    seen[r[0] as usize] = true;
                }
                assert!(seen.iter().all(|&b| b));
            }

            #[test]
            fn rng_first_outputs_equal_draw_rnd() {
                for d in 0..64 {
                    let s = root().new_domain(d);
                    let mut r = s.rng();
                    let block = s.draw_rnd::<4>();
                    for &b in &block {
                        assert_eq!(r.next_u32(), b);
                    }
                }
            }

            #[test]
            fn rng_f32_matches_rnd_f32() {
                let s = root().new_domain(5);
                let mut r = s.rng();
                let f = s.draw_rnd_f32::<4>();
                for &v in &f {
                    assert_eq!(r.next_f32(), v);
                }
            }

            #[test]
            fn rng_streams_are_reproducible_and_long() {
                let s = root().new_domain(5);
                let mut a = s.rng();
                let mut b = s.rng();
                let outs: Vec<u32> = (0..10_000).map(|_| a.next_u32()).collect();
                for &o in &outs {
                    assert_eq!(b.next_u32(), o);
                }
                common::assert_all_distinct(&outs, "rng stream");
            }

            #[test]
            fn rng_streams_differ_across_domains() {
                let firsts: Vec<u32> = (0..2048)
                    .map(|d| root().new_domain(d).rng().next_u32())
                    .collect();
                common::assert_all_distinct(&firsts, "rng first outputs across domains");
            }

            #[test]
            fn rng_streams_differ_across_indices() {
                let firsts: Vec<u32> = (0..2048)
                    .map(|i| S::new(PX, PY, FRAME, i).rng().next_u32())
                    .collect();
                common::assert_all_distinct(&firsts, "rng first outputs across indices");
            }

            #[test]
            fn sample_and_rnd_blocks_are_different_streams() {
                let differing = (0..256)
                    .filter(|&d| {
                        let s = root().new_domain(d);
                        s.draw_sample::<4>() != s.draw_rnd::<4>()
                    })
                    .count();
                assert_eq!(differing, 256);
            }

            #[test]
            fn new_domain_keys_give_distinct_blocks() {
                let blocks: Vec<[u32; 4]> = (0..2048)
                    .map(|d| root().new_domain(d).draw_sample::<4>())
                    .collect();
                assert_domains_distinct(&blocks, "blocks across domain keys");
            }

            #[test]
            fn new_domain_negative_keys_are_valid_and_distinct() {
                let blocks: Vec<[u32; 4]> = (-1024..1024)
                    .map(|d| root().new_domain(d).draw_sample::<4>())
                    .collect();
                assert_domains_distinct(&blocks, "blocks across signed domain keys");
                let _ = root().new_domain(i32::MIN).draw_sample::<4>();
                let _ = root().new_domain(i32::MAX).draw_sample::<4>();
            }

            #[test]
            fn new_domain_rnd_blocks_are_distinct() {
                let blocks: Vec<[u32; 4]> = (0..2048)
                    .map(|d| root().new_domain(d).draw_rnd::<4>())
                    .collect();
                common::assert_all_distinct(&blocks, "rnd blocks across domain keys");
            }

            #[test]
            fn new_domain_does_not_change_the_parent() {
                let r = root();
                let before = r.draw_sample::<4>();
                let _ = r.new_domain(1);
                let _ = r.new_domain_split(1, 4, 2);
                let _ = r.new_domain_distrib(1, 2);
                let _ = r.new_domain_chain(1, 2);
                assert_eq!(r.draw_sample::<4>(), before);
            }

            #[test]
            fn new_domain_chain_is_nested_new_domain() {
                for k in 0..16 {
                    for i in 0..16 {
                        let a = root().new_domain_chain(k, i);
                        let b = root().new_domain(k).new_domain(i);
                        assert_eq!(a.draw_sample::<4>(), b.draw_sample::<4>());
                        assert_eq!(a.draw_rnd::<4>(), b.draw_rnd::<4>());
                    }
                }
            }

            #[test]
            fn new_domain_order_matters() {
                let a = root().new_domain(1).new_domain(2).draw_sample::<4>();
                let b = root().new_domain(2).new_domain(1).draw_sample::<4>();
                assert_ne!(a, b);
            }

            #[test]
            fn deep_domain_chains_stay_distinct() {
                let mut s = root();
                let mut blocks = vec![s.draw_sample::<4>()];
                for _ in 0..1024 {
                    s = s.new_domain(0);
                    blocks.push(s.draw_sample::<4>());
                }
                assert_domains_distinct(&blocks, "blocks down a domain chain");
            }

            #[test]
            fn different_pixels_give_distinct_blocks() {
                let blocks: Vec<[u32; 4]> =
                    (0..256).map(|x| block_at(x, PY, FRAME, INDEX, 0)).collect();
                common::assert_all_distinct(&blocks, "blocks across a pixel row");
                let blocks: Vec<[u32; 4]> =
                    (0..256).map(|y| block_at(PX, y, FRAME, INDEX, 0)).collect();
                common::assert_all_distinct(&blocks, "blocks across a pixel column");
            }

            #[test]
            fn a_pixel_patch_gives_distinct_blocks() {
                let blocks: Vec<[u32; 4]> = (0..32)
                    .flat_map(|y| (0..32).map(move |x| block_at(x, y, FRAME, INDEX, 0)))
                    .collect();
                common::assert_all_distinct(&blocks, "blocks across a 32x32 patch");
            }

            #[test]
            fn different_pixels_give_distinct_rnd_blocks() {
                let blocks: Vec<[u32; 4]> = (0..256)
                    .map(|x| S::new(x, PY, FRAME, INDEX).draw_rnd::<4>())
                    .collect();
                common::assert_all_distinct(&blocks, "rnd blocks across a pixel row");
            }

            #[test]
            fn pixels_tile_with_period_256() {
                // The pixel id is 8+8 bits, so the pattern repeats every 256
                // pixels along each axis (matches upstream).
                for &(x, y) in &[(0, 0), (255, 255), (PX, PY), (100, 200)] {
                    let base = block_at(x, y, FRAME, INDEX, 0);
                    assert_eq!(block_at(x + 256, y, FRAME, INDEX, 0), base);
                    assert_eq!(block_at(x, y + 256, FRAME, INDEX, 0), base);
                    assert_eq!(block_at(x - 256, y - 512, FRAME, INDEX, 0), base);
                    assert_eq!(block_at(x + 1024, y + 768, FRAME, INDEX, 0), base);
                }
            }

            #[test]
            fn adjacent_pixels_do_not_tile_early() {
                for x in 0..255 {
                    assert_ne!(
                        block_at(x, 0, FRAME, INDEX, 0),
                        block_at(x + 1, 0, FRAME, INDEX, 0)
                    );
                }
            }

            #[test]
            fn negative_pixel_coordinates_are_accepted() {
                let a = block_at(-1, -1, FRAME, INDEX, 0);
                let b = block_at(255, 255, FRAME, INDEX, 0);
                assert_eq!(a, b);
                let _ = block_at(i32::MIN, i32::MAX, FRAME, INDEX, 0);
            }

            #[test]
            fn different_frames_give_distinct_blocks() {
                let blocks: Vec<[u32; 4]> =
                    (0..2048).map(|f| block_at(PX, PY, f, INDEX, 0)).collect();
                assert_domains_distinct(&blocks, "blocks across frames");
            }

            #[test]
            fn negative_frames_are_accepted_and_distinct() {
                let blocks: Vec<[u32; 4]> =
                    (-512..512).map(|f| block_at(PX, PY, f, INDEX, 0)).collect();
                assert_domains_distinct(&blocks, "blocks across signed frames");
            }

            #[test]
            fn different_indices_give_distinct_blocks() {
                let blocks: Vec<[u32; 4]> =
                    (0..4096).map(|i| block_at(PX, PY, FRAME, i, 0)).collect();
                common::assert_all_distinct(&blocks, "blocks across indices");
            }

            #[test]
            fn different_indices_give_distinct_rnd_blocks() {
                let blocks: Vec<[u32; 4]> = (0..4096)
                    .map(|i| S::new(PX, PY, FRAME, i).draw_rnd::<4>())
                    .collect();
                common::assert_all_distinct(&blocks, "rnd blocks across indices");
            }

            #[test]
            fn frame_and_high_index_bits_alias_as_upstream() {
                for f in [0, 3, 100] {
                    for i in [0, 7, 65535] {
                        assert_eq!(
                            block_at(PX, PY, f, i + 65536, 0),
                            block_at(PX, PY, f + 1, i, 0)
                        );
                    }
                }
            }

            #[test]
            fn large_indices_are_accepted() {
                for i in [65536, 65537, 1 << 20, 1 << 30, i32::MAX - 1, i32::MAX] {
                    let s = S::new(PX, PY, FRAME, i).new_domain(0);
                    let f = s.draw_sample_f32::<4>();
                    assert!(f.iter().all(|v| (0.0..1.0).contains(v)));
                    let _ = s.draw_rnd::<4>();
                }
            }

            #[test]
            fn first_dimension_visits_every_16_bit_stratum_once_over_a_full_index_block() {
                let vals: Vec<u32> = (0..65536)
                    .map(|i| block_at(PX, PY, FRAME, i, 0)[0])
                    .collect();
                common::assert_1d_stratified(&vals, 16, "dim 0 over 2^16 indices");
            }

            #[test]
            fn every_dimension_is_stratified_over_index_prefixes() {
                for m in 0..=10u32 {
                    let rows: Vec<[u32; 4]> = (0..(1 << m))
                        .map(|i| block_at(PX, PY, FRAME, i, 1))
                        .collect();
                    for d in 0..4 {
                        let vals: Vec<u32> = rows.iter().map(|r| r[d]).collect();
                        common::assert_1d_stratified(&vals, m, &format!("dim {d} m={m}"));
                    }
                }
            }

            #[test]
            fn every_dimension_is_stratified_over_aligned_index_blocks() {
                for m in 1..=8u32 {
                    for block in [1i32, 3, 17, 255] {
                        let start = block << m;
                        let rows: Vec<[u32; 4]> = (start..start + (1 << m))
                            .map(|i| block_at(PX, PY, FRAME, i, 1))
                            .collect();
                        for d in 0..4 {
                            let vals: Vec<u32> = rows.iter().map(|r| r[d]).collect();
                            common::assert_1d_stratified(
                                &vals,
                                m,
                                &format!("dim {d} block {block} m={m}"),
                            );
                        }
                    }
                }
            }

            #[test]
            fn stratification_holds_at_several_pixels_and_domains() {
                for &(x, y, d) in &[(0, 0, 0), (255, 255, 1), (77, 191, 12345), (-5, 1000, -9)] {
                    let m = 8u32;
                    let rows: Vec<[u32; 4]> =
                        (0..(1 << m)).map(|i| block_at(x, y, FRAME, i, d)).collect();
                    for dim in 0..4 {
                        let vals: Vec<u32> = rows.iter().map(|r| r[dim]).collect();
                        common::assert_1d_stratified(
                            &vals,
                            m,
                            &format!("pixel ({x},{y}) domain {d} dim {dim}"),
                        );
                    }
                }
            }

            #[test]
            fn split_domain_matches_a_root_at_the_combined_index() {
                // root(i).split(k, size, j) == root(i*size + j).domain(k).domain(0)
                // while the combined index stays below 2^16.
                for i in [0, 1, 5, 100] {
                    for size in [1, 2, 3, 8, 64] {
                        for j in 0..size.min(6) {
                            let a = S::new(PX, PY, FRAME, i).new_domain_split(9, size, j);
                            let b = S::new(PX, PY, FRAME, i * size + j)
                                .new_domain(9)
                                .new_domain(0);
                            assert_eq!(
                                a.draw_sample::<4>(),
                                b.draw_sample::<4>(),
                                "i={i} size={size} j={j}"
                            );
                            assert_eq!(
                                a.draw_rnd::<4>(),
                                b.draw_rnd::<4>(),
                                "i={i} size={size} j={j}"
                            );
                        }
                    }
                }
            }

            #[test]
            fn split_domain_sub_indices_give_distinct_blocks() {
                let size = 256;
                let blocks: Vec<[u32; 4]> = (0..size)
                    .map(|j| root().new_domain_split(2, size, j).draw_sample::<4>())
                    .collect();
                common::assert_all_distinct(&blocks, "split blocks");
            }

            #[test]
            fn split_domain_sub_indices_are_stratified() {
                for m in 1..=8u32 {
                    let size = 1 << m;
                    let rows: Vec<[u32; 4]> = (0..size)
                        .map(|j| root().new_domain_split(2, size, j).draw_sample::<4>())
                        .collect();
                    for d in 0..4 {
                        let vals: Vec<u32> = rows.iter().map(|r| r[d]).collect();
                        common::assert_1d_stratified(&vals, m, &format!("split dim {d} m={m}"));
                    }
                }
            }

            #[test]
            fn split_domain_across_base_indices_and_sub_indices_is_stratified() {
                // 2^a base indices x 2^b sub indices tile 2^(a+b) sample ids.
                let (a, b) = (4u32, 4u32);
                let size = 1 << b;
                let mut vals = Vec::new();
                for i in 0..(1 << a) {
                    for j in 0..size {
                        vals.push(
                            S::new(PX, PY, FRAME, i)
                                .new_domain_split(2, size, j)
                                .draw_sample::<1>()[0],
                        );
                    }
                }
                common::assert_1d_stratified(&vals, a + b, "split tiling");
            }

            #[test]
            fn split_domain_with_size_one_is_a_double_domain() {
                for i in [0, 9, 65535] {
                    let a = S::new(PX, PY, FRAME, i).new_domain_split(4, 1, 0);
                    let b = S::new(PX, PY, FRAME, i).new_domain(4).new_domain(0);
                    assert_eq!(a.draw_sample::<4>(), b.draw_sample::<4>());
                }
            }

            #[test]
            fn split_domain_differs_from_plain_domain() {
                let a = root().new_domain_split(4, 2, 0).draw_sample::<4>();
                let b = root().new_domain(4).draw_sample::<4>();
                assert_ne!(a, b);
            }

            #[test]
            fn split_domains_with_different_keys_differ() {
                let blocks: Vec<[u32; 4]> = (0..512)
                    .map(|k| root().new_domain_split(k, 4, 1).draw_sample::<4>())
                    .collect();
                assert_domains_distinct(&blocks, "split blocks across keys");
            }

            #[test]
            fn distrib_domain_matches_a_root_at_the_distributed_index() {
                // root(i).distrib(k, j) == root(j).domain(k).domain(0).domain(i)
                for i in [0, 1, 5, 4000, 65535] {
                    for j in [0, 1, 2, 999, 65535] {
                        let a = S::new(PX, PY, FRAME, i).new_domain_distrib(9, j);
                        let b = S::new(PX, PY, FRAME, j)
                            .new_domain(9)
                            .new_domain(0)
                            .new_domain(i);
                        assert_eq!(a.draw_sample::<4>(), b.draw_sample::<4>(), "i={i} j={j}");
                        assert_eq!(a.draw_rnd::<4>(), b.draw_rnd::<4>(), "i={i} j={j}");
                    }
                }
            }

            #[test]
            fn distrib_domain_indices_give_distinct_blocks() {
                let blocks: Vec<[u32; 4]> = (0..2048)
                    .map(|j| root().new_domain_distrib(2, j).draw_sample::<4>())
                    .collect();
                common::assert_all_distinct(&blocks, "distrib blocks");
            }

            #[test]
            fn distrib_domain_indices_are_stratified() {
                for m in 0..=10u32 {
                    let rows: Vec<[u32; 4]> = (0..(1 << m))
                        .map(|j| root().new_domain_distrib(2, j).draw_sample::<4>())
                        .collect();
                    for d in 0..4 {
                        let vals: Vec<u32> = rows.iter().map(|r| r[d]).collect();
                        common::assert_1d_stratified(&vals, m, &format!("distrib dim {d} m={m}"));
                    }
                }
            }

            #[test]
            fn distrib_domains_from_different_roots_differ() {
                let blocks: Vec<[u32; 4]> = (0..1024)
                    .map(|i| {
                        S::new(PX, PY, FRAME, i)
                            .new_domain_distrib(2, 0)
                            .draw_sample::<4>()
                    })
                    .collect();
                assert_domains_distinct(&blocks, "distrib blocks across roots");
            }

            #[test]
            fn distrib_domain_large_indices_are_accepted() {
                for j in [65536, 1 << 20, i32::MAX] {
                    let f = root().new_domain_distrib(2, j).draw_sample_f32::<4>();
                    assert!(f.iter().all(|v| (0.0..1.0).contains(v)));
                }
            }

            #[test]
            fn chain_domain_indices_give_distinct_blocks() {
                let blocks: Vec<[u32; 4]> = (0..2048)
                    .map(|j| root().new_domain_chain(2, j).draw_sample::<4>())
                    .collect();
                assert_domains_distinct(&blocks, "chain blocks");
            }

            #[test]
            fn the_three_split_kinds_differ_from_each_other() {
                let split = root().new_domain_split(3, 8, 1).draw_sample::<4>();
                let distrib = root().new_domain_distrib(3, 1).draw_sample::<4>();
                let chain = root().new_domain_chain(3, 1).draw_sample::<4>();
                assert_ne!(split, distrib);
                assert_ne!(split, chain);
                assert_ne!(distrib, chain);
            }

            #[test]
            fn warm_cache_is_idempotent_and_does_not_change_draws() {
                let before = root().new_domain(0).draw_sample::<4>();
                S::warm_cache();
                S::warm_cache();
                assert_eq!(root().new_domain(0).draw_sample::<4>(), before);
            }

            #[test]
            fn all_top_12_bits_are_exercised_within_a_4096_index_block() {
                // Every sampler permutes the top 12 bits over an aligned block
                // of 2^12 indices. (The lattice varies *only* those bits within
                // such a block; see the cross-sampler suite.)
                for d in 0..4 {
                    let mut or = 0u32;
                    let mut and = u32::MAX;
                    for i in 0..4096 {
                        let v = block_at(PX, PY, FRAME, i, 0)[d] >> 20;
                        or |= v;
                        and &= v;
                    }
                    assert_eq!(or, 0xfff, "dim {d}");
                    assert_eq!(and, 0, "dim {d}");
                }
            }

            #[test]
            fn all_32_bits_are_exercised_across_domains() {
                for d in 0..4 {
                    let mut or = 0u32;
                    let mut and = u32::MAX;
                    for k in 0..4096 {
                        let v = root().new_domain(k).draw_sample::<4>()[d];
                        or |= v;
                        and &= v;
                    }
                    assert_eq!(or, u32::MAX, "dim {d}");
                    assert_eq!(and, 0, "dim {d}");
                }
            }

            #[test]
            fn dimension_means_are_close_to_half() {
                let n = 4096;
                let mut sums = [0.0f64; 4];
                for i in 0..n {
                    let f = S::new(PX, PY, FRAME, i)
                        .new_domain(0)
                        .draw_sample_f32::<4>();
                    for d in 0..4 {
                        sums[d] += f[d] as f64;
                    }
                }
                for (d, s) in sums.iter().enumerate() {
                    let mean = s / n as f64;
                    assert!((mean - 0.5).abs() < 2e-3, "dim {d}: mean {mean}");
                }
            }

            #[test]
            fn rnd_means_are_close_to_half() {
                let n = 1 << 16;
                let mut sum = 0.0f64;
                for i in 0..n {
                    sum += S::new(PX, PY, FRAME, i).draw_rnd_f32::<1>()[0] as f64;
                }
                let mean = sum / n as f64;
                assert!((mean - 0.5).abs() < 5e-3, "rnd mean {mean}");
            }

            #[test]
            fn integrates_a_smooth_function_accurately() {
                let exact = 2.0 / 3.0;
                let est = integrate2d::<<S as SamplerType>::Impl>(4096, |x, y| x * x + y * y);
                assert!((est - exact).abs() < 3e-3, "estimate {est}");
            }

            #[test]
            fn integration_error_shrinks_with_more_samples() {
                let exact = 2.0 / 3.0;
                let coarse = (integrate2d::<<S as SamplerType>::Impl>(64, |x, y| x * x + y * y)
                    - exact)
                    .abs();
                let fine = (integrate2d::<<S as SamplerType>::Impl>(4096, |x, y| x * x + y * y)
                    - exact)
                    .abs();
                assert!(fine <= coarse + 1e-9, "coarse {coarse} fine {fine}");
            }

            #[test]
            fn concurrent_draws_match_the_main_thread() {
                let expected: Vec<[u32; 4]> =
                    (0..512).map(|i| block_at(PX, PY, FRAME, i, 3)).collect();
                let handles: Vec<_> = (0..8)
                    .map(|_| {
                        std::thread::spawn(move || {
                            (0..512)
                                .map(|i| block_at(PX, PY, FRAME, i, 3))
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                for h in handles {
                    assert_eq!(h.join().unwrap(), expected);
                }
            }

            #[test]
            fn samplers_can_be_sent_across_threads() {
                let s = root().new_domain(4);
                let expected = s.draw_sample::<4>();
                let got = std::thread::spawn(move || s.draw_sample::<4>())
                    .join()
                    .unwrap();
                assert_eq!(got, expected);
            }

            #[test]
            fn state_is_pixel_decorrelated_as_declared() {
                // Cache-free samplers decorrelate the state per pixel up front
                // (pixel domain first, then the caller's domain); blue-noise
                // samplers keep the state correlated and add the pixel domain
                // only when drawing the low-quality stream (caller's domain
                // first, then the pixel domain).
                for &(x, y, d) in &[(0, 0, 1), (1, 0, 2), (255, 255, -7), (PX, PY, 12345)] {
                    let st = State64Bit::new(x, y, FRAME, INDEX);
                    let pixel = st.pixel_id as i32;
                    let expected = if $decorrelated {
                        st.new_domain(pixel).new_domain(d).draw_rnd::<4>()
                    } else {
                        st.new_domain(d).new_domain(pixel).draw_rnd::<4>()
                    };
                    assert_eq!(
                        S::new(x, y, FRAME, INDEX).new_domain(d).draw_rnd::<4>(),
                        expected
                    );
                }
                let a = S::new(0, 0, FRAME, INDEX).draw_rnd::<4>();
                let b = S::new(1, 0, FRAME, INDEX).draw_rnd::<4>();
                assert_ne!(a, b);
            }
        }
    };
}

/// Maps a `Sampler<T>` alias back to its `T` so the macro can name the impl.
trait SamplerType {
    type Impl: SamplerImpl;
}
impl<T: SamplerImpl> SamplerType for Sampler<T> {
    type Impl = T;
}

sampler_suite!(sobol, SobolSampler, decorrelated_state: true, pattern_bits: 32);
sampler_suite!(lattice, LatticeSampler, decorrelated_state: true, pattern_bits: 32);
sampler_suite!(pmj, PmjSampler, decorrelated_state: true, pattern_bits: 32);
sampler_suite!(sobolbn, SobolBnSampler, decorrelated_state: false, pattern_bits: 16);
sampler_suite!(latticebn, LatticeBnSampler, decorrelated_state: false, pattern_bits: 16);
sampler_suite!(pmjbn, PmjBnSampler, decorrelated_state: false, pattern_bits: 16);
