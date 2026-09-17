//! Cross-checks between the sampler API and the public building blocks it is
//! assembled from, plus relationships *between* the six samplers.

mod common;

use openqmc::bntables::{self, table_value};
use openqmc::lookup::shuffled_scrambled_lookup;
use openqmc::owen::shuffled_scrambled_sobol;
use openqmc::pcg;
use openqmc::rank1::shuffled_rotated_lattice;
use openqmc::sampler::{Sampler, SamplerImpl};
use openqmc::state::{MAX_INDEX_SIZE, State64Bit};
use openqmc::stochastic::stochastic_pmj_init;
use openqmc::{
    LatticeBnSampler, LatticeSampler, PmjBnSampler, PmjSampler, SobolBnSampler, SobolSampler,
};

const CASES: [(i32, i32, i32, i32, i32); 8] = [
    (0, 0, 0, 0, 0),
    (12, 34, 2, 7, 0),
    (255, 255, 1, 65535, 3),
    (300, -7, -1, 65536, -2),
    (77, 191, 9, 1 << 20, 12345),
    (1, 2, 3, i32::MAX, 5),
    (128, 64, 0, 4096, i32::MIN),
    (5, 5, 5, 5, i32::MAX),
];

fn pmj_table() -> Vec<[u32; 4]> {
    let mut t = vec![[0u32; 4]; MAX_INDEX_SIZE];
    stochastic_pmj_init(MAX_INDEX_SIZE, &mut t);
    t
}

fn decorrelated(x: i32, y: i32, f: i32, i: i32, d: i32) -> State64Bit {
    State64Bit::new(x, y, f, i)
        .pixel_decorrelate()
        .new_domain(d)
}

fn correlated(x: i32, y: i32, f: i32, i: i32, d: i32) -> State64Bit {
    State64Bit::new(x, y, f, i).new_domain(d)
}

#[test]
fn sobol_sampler_is_the_owen_core_on_a_decorrelated_state() {
    for (x, y, f, i, d) in CASES {
        let st = decorrelated(x, y, f, i, d);
        let expected =
            shuffled_scrambled_sobol::<4>(st.sample_id as u32, pcg::output(st.pattern_id));
        assert_eq!(
            SobolSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_sample::<4>(),
            expected
        );
    }
}

#[test]
fn lattice_sampler_is_the_rank1_core_on_a_decorrelated_state() {
    for (x, y, f, i, d) in CASES {
        let st = decorrelated(x, y, f, i, d);
        let expected = shuffled_rotated_lattice::<4>(st.sample_id as u32, st.pattern_id);
        assert_eq!(
            LatticeSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_sample::<4>(),
            expected
        );
    }
}

#[test]
fn pmj_sampler_is_a_lookup_into_the_stochastic_table() {
    let table = pmj_table();
    for (x, y, f, i, d) in CASES {
        let st = decorrelated(x, y, f, i, d);
        let expected = shuffled_scrambled_lookup::<4, 4>(
            st.sample_id as u32,
            pcg::output(st.pattern_id),
            &table,
        );
        assert_eq!(
            PmjSampler::new(x, y, f, i).new_domain(d).draw_sample::<4>(),
            expected
        );
    }
}

#[test]
fn sobolbn_sampler_is_the_owen_core_keyed_by_the_blue_noise_table() {
    for (x, y, f, i, d) in CASES {
        let st = correlated(x, y, f, i, d);
        let t = table_value::<8, 8, 0>(
            st.pixel_id,
            pcg::output(st.pattern_id) as u16,
            bntables::sobol::key_table(),
            bntables::sobol::rank_table(),
        );
        let expected = shuffled_scrambled_sobol::<4>(st.sample_id as u32 ^ t.rank, t.key);
        assert_eq!(
            SobolBnSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_sample::<4>(),
            expected
        );
    }
}

#[test]
fn latticebn_sampler_is_the_rank1_core_keyed_by_the_blue_noise_table() {
    for (x, y, f, i, d) in CASES {
        let st = correlated(x, y, f, i, d);
        let t = table_value::<8, 8, 0>(
            st.pixel_id,
            pcg::output(st.pattern_id) as u16,
            bntables::lattice::key_table(),
            bntables::lattice::rank_table(),
        );
        let expected = shuffled_rotated_lattice::<4>(st.sample_id as u32 ^ t.rank, t.key);
        assert_eq!(
            LatticeBnSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_sample::<4>(),
            expected
        );
    }
}

#[test]
fn pmjbn_sampler_is_a_table_lookup_keyed_by_the_blue_noise_table() {
    let table = pmj_table();
    for (x, y, f, i, d) in CASES {
        let st = correlated(x, y, f, i, d);
        let t = table_value::<8, 8, 0>(
            st.pixel_id,
            pcg::output(st.pattern_id) as u16,
            bntables::pmj::key_table(),
            bntables::pmj::rank_table(),
        );
        let expected =
            shuffled_scrambled_lookup::<4, 4>(st.sample_id as u32 ^ t.rank, t.key, &table);
        assert_eq!(
            PmjBnSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_sample::<4>(),
            expected
        );
    }
}

#[test]
fn cache_free_samplers_draw_rnd_from_the_decorrelated_state() {
    for (x, y, f, i, d) in CASES {
        let st = decorrelated(x, y, f, i, d);
        assert_eq!(
            SobolSampler::new(x, y, f, i).new_domain(d).draw_rnd::<4>(),
            st.draw_rnd::<4>()
        );
        assert_eq!(
            LatticeSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_rnd::<4>(),
            st.draw_rnd::<4>()
        );
        assert_eq!(
            PmjSampler::new(x, y, f, i).new_domain(d).draw_rnd::<4>(),
            st.draw_rnd::<4>()
        );
        assert_eq!(
            SobolSampler::new(x, y, f, i).new_domain(d).rng().next_u32(),
            st.rng().next_u32()
        );
    }
}

#[test]
fn blue_noise_samplers_draw_rnd_from_a_pixel_domain_of_the_state() {
    for (x, y, f, i, d) in CASES {
        let st = correlated(x, y, f, i, d);
        let expected = st.new_domain(st.pixel_id as i32).draw_rnd::<4>();
        assert_eq!(
            SobolBnSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_rnd::<4>(),
            expected
        );
        assert_eq!(
            LatticeBnSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_rnd::<4>(),
            expected
        );
        assert_eq!(
            PmjBnSampler::new(x, y, f, i).new_domain(d).draw_rnd::<4>(),
            expected
        );
        assert_eq!(
            SobolBnSampler::new(x, y, f, i)
                .new_domain(d)
                .rng()
                .next_u32(),
            expected[0]
        );
    }
}

#[test]
fn cache_free_samplers_share_one_rnd_stream_per_domain() {
    for (x, y, f, i, d) in CASES {
        let reference = SobolSampler::new(x, y, f, i).new_domain(d).draw_rnd::<4>();
        assert_eq!(
            LatticeSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_rnd::<4>(),
            reference
        );
        assert_eq!(
            PmjSampler::new(x, y, f, i).new_domain(d).draw_rnd::<4>(),
            reference
        );
    }
}

#[test]
fn blue_noise_samplers_share_one_rnd_stream_per_domain() {
    for (x, y, f, i, d) in CASES {
        let reference = SobolBnSampler::new(x, y, f, i)
            .new_domain(d)
            .draw_rnd::<4>();
        assert_eq!(
            LatticeBnSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_rnd::<4>(),
            reference
        );
        assert_eq!(
            PmjBnSampler::new(x, y, f, i).new_domain(d).draw_rnd::<4>(),
            reference
        );
    }
}

#[test]
fn cache_free_and_blue_noise_rnd_streams_differ_by_domain_order() {
    // Cache-free: new_domain(pixel).new_domain(d). Blue-noise:
    // new_domain(d).new_domain(pixel). The two orders give different seeds
    // unless the derived domain is the root itself.
    for (x, y, f, i, d) in CASES {
        let free = SobolSampler::new(x, y, f, i).new_domain(d).draw_rnd::<4>();
        let blue = SobolBnSampler::new(x, y, f, i)
            .new_domain(d)
            .draw_rnd::<4>();
        let st = State64Bit::new(x, y, f, i);
        if st.pixel_id as i32 == d {
            // Same key twice: the order cannot matter.
            assert_eq!(free, blue, "case {:?}", (x, y, f, i, d));
        } else {
            assert_ne!(free, blue, "case {:?}", (x, y, f, i, d));
        }
        assert_eq!(
            free,
            st.new_domain(st.pixel_id as i32)
                .new_domain(d)
                .draw_rnd::<4>()
        );
        assert_eq!(
            blue,
            st.new_domain(d)
                .new_domain(st.pixel_id as i32)
                .draw_rnd::<4>()
        );
    }
    // At the root (no derived domain) both families agree.
    for (x, y, f, i, _) in CASES {
        assert_eq!(
            SobolSampler::new(x, y, f, i).draw_rnd::<4>(),
            SobolBnSampler::new(x, y, f, i).draw_rnd::<4>()
        );
    }
}

#[test]
fn the_six_samplers_produce_six_different_high_quality_sequences() {
    for (x, y, f, i, d) in CASES {
        let blocks = [
            SobolSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_sample::<4>(),
            LatticeSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_sample::<4>(),
            PmjSampler::new(x, y, f, i).new_domain(d).draw_sample::<4>(),
            SobolBnSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_sample::<4>(),
            LatticeBnSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_sample::<4>(),
            PmjBnSampler::new(x, y, f, i)
                .new_domain(d)
                .draw_sample::<4>(),
        ];
        common::assert_all_distinct(&blocks, &format!("case {:?}", (x, y, f, i, d)));
    }
}

#[test]
fn blue_noise_variants_differ_from_their_base_at_every_pixel_of_a_row() {
    let mut same = 0;
    for x in 0..256 {
        if SobolSampler::new(x, 0, 0, 0)
            .new_domain(0)
            .draw_sample::<4>()
            == SobolBnSampler::new(x, 0, 0, 0)
                .new_domain(0)
                .draw_sample::<4>()
        {
            same += 1;
        }
        if LatticeSampler::new(x, 0, 0, 0)
            .new_domain(0)
            .draw_sample::<4>()
            == LatticeBnSampler::new(x, 0, 0, 0)
                .new_domain(0)
                .draw_sample::<4>()
        {
            same += 1;
        }
        if PmjSampler::new(x, 0, 0, 0).new_domain(0).draw_sample::<4>()
            == PmjBnSampler::new(x, 0, 0, 0)
                .new_domain(0)
                .draw_sample::<4>()
        {
            same += 1;
        }
    }
    assert_eq!(same, 0);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn sampler_sizes_are_state_plus_table_references() {
    use std::mem::size_of;
    assert_eq!(size_of::<SobolSampler>(), 8);
    assert_eq!(size_of::<LatticeSampler>(), 8);
    assert_eq!(size_of::<PmjSampler>(), 8 + 16);
    assert_eq!(size_of::<SobolBnSampler>(), 8 + 2 * 16);
    assert_eq!(size_of::<LatticeBnSampler>(), 8 + 2 * 16);
    assert_eq!(size_of::<PmjBnSampler>(), 8 + 3 * 16);
}

#[test]
fn sampler_wrapper_adds_no_size_over_its_impl() {
    use std::mem::size_of;
    assert_eq!(
        size_of::<SobolSampler>(),
        size_of::<openqmc::sobol::SobolImpl>()
    );
    assert_eq!(
        size_of::<LatticeSampler>(),
        size_of::<openqmc::lattice::LatticeImpl>()
    );
    assert_eq!(size_of::<PmjSampler>(), size_of::<openqmc::pmj::PmjImpl>());
    assert_eq!(
        size_of::<SobolBnSampler>(),
        size_of::<openqmc::sobolbn::SobolBnImpl>()
    );
    assert_eq!(
        size_of::<LatticeBnSampler>(),
        size_of::<openqmc::latticebn::LatticeBnImpl>()
    );
    assert_eq!(
        size_of::<PmjBnSampler>(),
        size_of::<openqmc::pmjbn::PmjBnImpl>()
    );
}

#[test]
fn impls_are_reachable_through_the_trait_directly() {
    // The trait surface matches the wrapper for every sampler.
    fn check<T: SamplerImpl>(x: i32, y: i32, f: i32, i: i32, d: i32) {
        let imp = T::from_pixel(x, y, f, i).new_domain(d);
        let wrapped = Sampler::<T>::new(x, y, f, i).new_domain(d);
        assert_eq!(imp.draw_block(), wrapped.draw_sample::<4>());
        assert_eq!(imp.draw_rnd_block(), wrapped.draw_rnd::<4>());
        assert_eq!(imp.rng().next_u32(), wrapped.rng().next_u32());
        assert_eq!(
            imp.new_domain_split(1, 4, 2).draw_block(),
            wrapped.new_domain_split(1, 4, 2).draw_sample::<4>()
        );
        assert_eq!(
            imp.new_domain_distrib(1, 2).draw_block(),
            wrapped.new_domain_distrib(1, 2).draw_sample::<4>()
        );
        T::warm_cache();
    }
    for (x, y, f, i, d) in CASES {
        check::<openqmc::sobol::SobolImpl>(x, y, f, i, d);
        check::<openqmc::lattice::LatticeImpl>(x, y, f, i, d);
        check::<openqmc::pmj::PmjImpl>(x, y, f, i, d);
        check::<openqmc::sobolbn::SobolBnImpl>(x, y, f, i, d);
        check::<openqmc::latticebn::LatticeBnImpl>(x, y, f, i, d);
        check::<openqmc::pmjbn::PmjBnImpl>(x, y, f, i, d);
    }
}

#[test]
fn a_generic_integrator_accepts_every_sampler() {
    fn estimate<T: SamplerImpl>(n: i32) -> f64 {
        let mut acc = 0.0;
        for i in 0..n {
            let [x, y] = Sampler::<T>::new(9, 9, 0, i)
                .new_domain(0)
                .draw_sample_f32::<2>();
            acc += if (x * x + y * y) < 1.0 { 1.0 } else { 0.0 };
        }
        4.0 * acc / n as f64
    }
    let n = 1 << 14;
    let pis = [
        estimate::<openqmc::sobol::SobolImpl>(n),
        estimate::<openqmc::lattice::LatticeImpl>(n),
        estimate::<openqmc::pmj::PmjImpl>(n),
        estimate::<openqmc::sobolbn::SobolBnImpl>(n),
        estimate::<openqmc::latticebn::LatticeBnImpl>(n),
        estimate::<openqmc::pmjbn::PmjBnImpl>(n),
    ];
    for (k, pi) in pis.iter().enumerate() {
        assert!(
            (pi - std::f64::consts::PI).abs() < 0.02,
            "sampler {k}: {pi}"
        );
    }
}

#[test]
fn lattice_low_16_bits_are_constant_within_an_index_block() {
    // A rank-1 lattice over 2^16 indices carries exactly 16 bits of variation
    // per dimension; the low bits are fixed by the pattern's shuffle and shift.
    for d in 0..4 {
        let first = LatticeSampler::new(3, 4, 0, 0)
            .new_domain(0)
            .draw_sample::<4>()[d]
            & 0xffff;
        for i in (0..65536).step_by(97) {
            let v = LatticeSampler::new(3, 4, 0, i)
                .new_domain(0)
                .draw_sample::<4>()[d]
                & 0xffff;
            assert_eq!(v, first, "dim {d} index {i}");
        }
    }
}

#[test]
fn sobol_and_pmj_low_bits_vary_within_an_index_block() {
    for d in 0..4 {
        let sobol: Vec<u32> = (0..4096)
            .map(|i| {
                SobolSampler::new(3, 4, 0, i)
                    .new_domain(0)
                    .draw_sample::<4>()[d]
                    & 0xffff
            })
            .collect();
        let pmj: Vec<u32> = (0..4096)
            .map(|i| PmjSampler::new(3, 4, 0, i).new_domain(0).draw_sample::<4>()[d] & 0xffff)
            .collect();
        assert_eq!(sobol.iter().fold(0, |a, &v| a | v), 0xffff, "sobol dim {d}");
        assert_eq!(pmj.iter().fold(0, |a, &v| a | v), 0xffff, "pmj dim {d}");
        assert_eq!(sobol.iter().fold(0xffff, |a, &v| a & v), 0, "sobol dim {d}");
        assert_eq!(pmj.iter().fold(0xffff, |a, &v| a & v), 0, "pmj dim {d}");
    }
}

#[test]
fn blue_noise_samplers_see_the_pattern_only_through_a_16_bit_shift() {
    // Two domains whose pcg::output(pattern) agree in the low 16 bits draw the
    // same high-quality block at the same pixel and sample index.
    let (x, y, f, i) = (12, 34, 2, 7);
    let base = State64Bit::new(x, y, f, i);
    let mut by_shift: std::collections::HashMap<u16, i32> = std::collections::HashMap::new();
    let mut coincidences = 0;
    for k in 0..20000 {
        let shift = pcg::output(base.new_domain(k).pattern_id) as u16;
        if let Some(&other) = by_shift.get(&shift) {
            coincidences += 1;
            assert_eq!(
                SobolBnSampler::new(x, y, f, i)
                    .new_domain(k)
                    .draw_sample::<4>(),
                SobolBnSampler::new(x, y, f, i)
                    .new_domain(other)
                    .draw_sample::<4>()
            );
            assert_eq!(
                LatticeBnSampler::new(x, y, f, i)
                    .new_domain(k)
                    .draw_sample::<4>(),
                LatticeBnSampler::new(x, y, f, i)
                    .new_domain(other)
                    .draw_sample::<4>()
            );
            assert_eq!(
                PmjBnSampler::new(x, y, f, i)
                    .new_domain(k)
                    .draw_sample::<4>(),
                PmjBnSampler::new(x, y, f, i)
                    .new_domain(other)
                    .draw_sample::<4>()
            );
            // The low-quality stream still tells them apart.
            assert_ne!(
                SobolBnSampler::new(x, y, f, i)
                    .new_domain(k)
                    .draw_rnd::<4>(),
                SobolBnSampler::new(x, y, f, i)
                    .new_domain(other)
                    .draw_rnd::<4>()
            );
        } else {
            by_shift.insert(shift, k);
        }
    }
    assert!(
        coincidences > 100,
        "only {coincidences} shift coincidences in 20000 domains"
    );
}

#[test]
fn cache_free_samplers_never_collide_across_20000_domains() {
    let blocks: Vec<[u32; 4]> = (0..20000)
        .map(|k| {
            SobolSampler::new(12, 34, 2, 7)
                .new_domain(k)
                .draw_sample::<4>()
        })
        .collect();
    common::assert_all_distinct(&blocks, "sobol domains");
    let blocks: Vec<[u32; 4]> = (0..20000)
        .map(|k| {
            LatticeSampler::new(12, 34, 2, 7)
                .new_domain(k)
                .draw_sample::<4>()
        })
        .collect();
    common::assert_all_distinct(&blocks, "lattice domains");
    let blocks: Vec<[u32; 4]> = (0..20000)
        .map(|k| {
            PmjSampler::new(12, 34, 2, 7)
                .new_domain(k)
                .draw_sample::<4>()
        })
        .collect();
    common::assert_all_distinct(&blocks, "pmj domains");
}

#[test]
fn blue_noise_pixels_with_the_same_table_entry_share_a_pattern() {
    // Pixel p with shift s reads table[p + s]; pixel p + 1 with shift s - 1
    // reads the same entry, so the two draw identical blocks.
    let (f, i) = (2, 7);
    for x in 0..255 {
        let a = State64Bit::new(x, 10, f, i);
        let shift_a = pcg::output(a.new_domain(0).pattern_id) as u16;
        // Find a key whose shift is exactly one less in x for pixel x + 1.
        let want = ((shift_a & 0xff).wrapping_sub(1) & 0xff) | (shift_a & 0xff00);
        let b = State64Bit::new(x + 1, 10, f, i);
        let found = (0..200000).find(|&k| pcg::output(b.new_domain(k).pattern_id) as u16 == want);
        if let Some(k) = found {
            assert_eq!(
                SobolBnSampler::new(x, 10, f, i)
                    .new_domain(0)
                    .draw_sample::<4>(),
                SobolBnSampler::new(x + 1, 10, f, i)
                    .new_domain(k)
                    .draw_sample::<4>(),
                "x={x} k={k}"
            );
            return;
        }
    }
    panic!("no matching shift found");
}

#[test]
fn first_two_dimensions_of_sobol_family_are_02_nets_at_the_sampler_level() {
    for m in 0..=10u32 {
        let sobol: Vec<[u32; 2]> = (0..(1 << m))
            .map(|i| {
                SobolSampler::new(5, 6, 1, i)
                    .new_domain(2)
                    .draw_sample::<2>()
            })
            .collect();
        common::assert_02_net(&sobol, m, &format!("sobol m={m}"));
        let sobolbn: Vec<[u32; 2]> = (0..(1 << m))
            .map(|i| {
                SobolBnSampler::new(5, 6, 1, i)
                    .new_domain(2)
                    .draw_sample::<2>()
            })
            .collect();
        common::assert_02_net(&sobolbn, m, &format!("sobolbn m={m}"));
    }
}

#[test]
fn both_dimension_pairs_of_pmj_family_are_02_nets_at_the_sampler_level() {
    for m in 0..=10u32 {
        let rows: Vec<[u32; 4]> = (0..(1 << m))
            .map(|i| PmjSampler::new(5, 6, 1, i).new_domain(2).draw_sample::<4>())
            .collect();
        let first: Vec<[u32; 2]> = rows.iter().map(|r| [r[0], r[1]]).collect();
        let second: Vec<[u32; 2]> = rows.iter().map(|r| [r[2], r[3]]).collect();
        common::assert_02_net(&first, m, &format!("pmj (0,1) m={m}"));
        common::assert_02_net(&second, m, &format!("pmj (2,3) m={m}"));
        let rows: Vec<[u32; 4]> = (0..(1 << m))
            .map(|i| {
                PmjBnSampler::new(5, 6, 1, i)
                    .new_domain(2)
                    .draw_sample::<4>()
            })
            .collect();
        let first: Vec<[u32; 2]> = rows.iter().map(|r| [r[0], r[1]]).collect();
        let second: Vec<[u32; 2]> = rows.iter().map(|r| [r[2], r[3]]).collect();
        common::assert_02_net(&first, m, &format!("pmjbn (0,1) m={m}"));
        common::assert_02_net(&second, m, &format!("pmjbn (2,3) m={m}"));
    }
}

#[test]
fn split_and_distrib_preserve_02_nets_for_sobol() {
    let m = 8u32;
    let root = SobolSampler::new(5, 6, 1, 3);
    let split: Vec<[u32; 2]> = (0..(1 << m))
        .map(|j| root.new_domain_split(1, 1 << m, j).draw_sample::<2>())
        .collect();
    common::assert_02_net(&split, m, "sobol split");
    let distrib: Vec<[u32; 2]> = (0..(1 << m))
        .map(|j| root.new_domain_distrib(1, j).draw_sample::<2>())
        .collect();
    common::assert_02_net(&distrib, m, "sobol distrib");
}

#[test]
fn blue_noise_error_has_less_low_frequency_energy_than_white_noise() {
    // Render a 64x64 tile at 1 spp of a smooth integrand; the blue-noise
    // sampler's per-pixel error should have less energy after a 4x4 box blur
    // than the pixel-decorrelated (white noise) base sampler.
    fn low_freq_energy<T: SamplerImpl>() -> f64 {
        const W: usize = 64;
        let mut err = vec![0.0f64; W * W];
        for y in 0..W {
            for x in 0..W {
                let [u, v] = Sampler::<T>::new(x as i32, y as i32, 0, 0)
                    .new_domain(0)
                    .draw_sample_f32::<2>();
                err[y * W + x] = (u as f64 + v as f64) - 1.0;
            }
        }
        let mut energy = 0.0;
        for by in 0..W / 4 {
            for bx in 0..W / 4 {
                let mut s = 0.0;
                for y in 0..4 {
                    for x in 0..4 {
                        s += err[(by * 4 + y) * W + bx * 4 + x];
                    }
                }
                energy += (s / 16.0).powi(2);
            }
        }
        energy
    }
    let white = low_freq_energy::<openqmc::sobol::SobolImpl>();
    let blue = low_freq_energy::<openqmc::sobolbn::SobolBnImpl>();
    assert!(blue < white * 0.5, "blue {blue} vs white {white}");
    let white = low_freq_energy::<openqmc::pmj::PmjImpl>();
    let blue = low_freq_energy::<openqmc::pmjbn::PmjBnImpl>();
    assert!(blue < white * 0.5, "blue {blue} vs white {white}");
    let white = low_freq_energy::<openqmc::lattice::LatticeImpl>();
    let blue = low_freq_energy::<openqmc::latticebn::LatticeBnImpl>();
    assert!(blue < white * 0.5, "blue {blue} vs white {white}");
}

#[test]
fn golden_smoke_values_from_the_upstream_generator() {
    // A handful of the upstream golden vectors, restated here so this suite
    // stands alone if the generated file is ever regenerated.
    assert_eq!(pcg::hash(0), 0x07bb_2fe2);
    assert_eq!(pcg::hash(0xffff_ffff), 0xe62a_4902);
    assert_eq!(
        shuffled_scrambled_sobol::<4>(0, 0x07bb_2fe2),
        [0xb56e_e684, 0xf47f_54b7, 0x2166_db28, 0x12b5_7966]
    );
    assert_eq!(
        shuffled_rotated_lattice::<4>(1, 0x07bb_2fe2),
        [0xc3fd_8aed, 0x722c_5f92, 0xfc34_2a34, 0x2127_7911]
    );
}
