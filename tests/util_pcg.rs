//! Property tests for `openqmc::pcg` (oqmc/pcg.h) — PCG-RXS-M-XS-32 as a
//! sequential PRNG and as a stateless hash.

mod common;

use openqmc::pcg::{Rng, hash, init, init_seed, output, rng, state_transition};

const LCG_MUL: u32 = 747796405;
const LCG_INC: u32 = 2891336453;

#[test]
fn state_transition_is_the_documented_lcg() {
    for s in common::EDGE_U32
        .iter()
        .copied()
        .chain(common::random_u32s(1, 4096))
    {
        assert_eq!(
            state_transition(s),
            s.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC)
        );
    }
}

#[test]
fn state_transition_is_a_bijection() {
    // Odd multiplier => invertible mod 2^32. Check on a large prefix plus its
    // algebraic inverse.
    let inv = {
        // Newton iteration for the modular inverse of an odd number mod 2^32.
        let mut x = LCG_MUL;
        for _ in 0..5 {
            x = x.wrapping_mul(2u32.wrapping_sub(LCG_MUL.wrapping_mul(x)));
        }
        x
    };
    assert_eq!(LCG_MUL.wrapping_mul(inv), 1);
    for s in common::random_u32s(2, 1 << 16) {
        let next = state_transition(s);
        assert_eq!(next.wrapping_sub(LCG_INC).wrapping_mul(inv), s);
    }
}

#[test]
fn state_transition_has_no_collisions_on_a_million_states() {
    let outs: Vec<u32> = (0..(1u32 << 20)).map(state_transition).collect();
    common::assert_all_distinct(&outs, "state_transition");
}

#[test]
fn state_transition_has_full_period_on_low_bits() {
    // An LCG with these constants has full period 2^32, hence period 2^k on the
    // low k bits.
    for k in 1..=12u32 {
        let period = 1u32 << k;
        let mask = period - 1;
        let mut s = 0u32;
        let mut seen = vec![false; period as usize];
        for _ in 0..period {
            let low = (s & mask) as usize;
            assert!(!seen[low], "low {k} bits repeated early");
            seen[low] = true;
            s = state_transition(s);
        }
        assert_eq!(s & mask, 0);
    }
}

#[test]
fn output_is_a_bijection() {
    let outs: Vec<u32> = (0..(1u32 << 20)).map(output).collect();
    common::assert_all_distinct(&outs, "output");
    let outs: Vec<u32> = common::random_u32s(3, 1 << 18)
        .into_iter()
        .map(output)
        .collect();
    common::assert_all_distinct(&outs, "output(random)");
}

#[test]
fn output_fixes_zero_only_rarely() {
    assert_eq!(output(0), 0);
    let fixed = (1..(1u32 << 16)).filter(|&s| output(s) == s).count();
    assert!(fixed < 8, "{fixed} fixed points");
}

#[test]
fn output_matches_the_rxs_m_xs_steps() {
    for s in common::random_u32s(4, 4096) {
        let mut x = s;
        x ^= x >> (4 + (x >> 28));
        x = x.wrapping_mul(277803737);
        x ^= x >> 22;
        assert_eq!(output(s), x);
    }
}

#[test]
fn init_is_transition_of_zero() {
    assert_eq!(init(), state_transition(0));
    assert_eq!(init(), LCG_INC);
}

#[test]
fn init_seed_adds_the_seed() {
    for seed in common::EDGE_U32 {
        assert_eq!(init_seed(seed), init().wrapping_add(seed));
    }
    assert_eq!(init_seed(0), init());
    assert_eq!(init_seed(1).wrapping_sub(init_seed(0)), 1);
}

#[test]
fn init_seed_is_a_bijection() {
    let outs: Vec<u32> = (0..(1u32 << 18)).map(init_seed).collect();
    common::assert_all_distinct(&outs, "init_seed");
}

#[test]
fn hash_is_output_of_transition() {
    for k in common::EDGE_U32
        .iter()
        .copied()
        .chain(common::random_u32s(5, 4096))
    {
        assert_eq!(hash(k), output(state_transition(k)));
    }
}

#[test]
fn hash_has_no_collisions_on_a_million_keys() {
    let outs: Vec<u32> = (0..(1u32 << 20)).map(hash).collect();
    common::assert_all_distinct(&outs, "hash");
}

#[test]
fn hash_has_no_collisions_on_random_keys() {
    let outs: Vec<u32> = common::random_u32s(6, 1 << 18)
        .into_iter()
        .map(hash)
        .collect();
    common::assert_all_distinct(&outs, "hash(random)");
}

#[test]
fn hash_flips_roughly_half_the_bits_on_single_bit_changes() {
    let mut total = 0u64;
    let mut count = 0u64;
    for k in common::random_u32s(7, 4096) {
        for b in 0..32 {
            total += (hash(k) ^ hash(k ^ (1 << b))).count_ones() as u64;
            count += 1;
        }
    }
    let avg = total as f64 / count as f64;
    assert!((avg - 16.0).abs() < 0.5, "avalanche {avg}");
}

#[test]
fn hash_output_bits_are_balanced() {
    let n = 1u32 << 16;
    let mut ones = [0u32; 32];
    for k in 0..n {
        let h = hash(k);
        for (b, slot) in ones.iter_mut().enumerate() {
            *slot += (h >> b) & 1;
        }
    }
    for (b, &c) in ones.iter().enumerate() {
        let frac = c as f64 / n as f64;
        assert!((frac - 0.5).abs() < 0.02, "bit {b}: {frac}");
    }
}

#[test]
fn hash_equals_first_rng_output_from_that_state() {
    for k in common::random_u32s(8, 4096) {
        let mut s = k;
        assert_eq!(rng(&mut s), hash(k));
        assert_eq!(s, state_transition(k));
    }
}

#[test]
fn rng_advances_state_by_one_transition_per_call() {
    let mut s = 12345u32;
    for _ in 0..100 {
        let before = s;
        let out = rng(&mut s);
        assert_eq!(s, state_transition(before));
        assert_eq!(out, output(s));
    }
}

#[test]
fn rng_stream_matches_manual_iteration() {
    let mut a = Rng::new(0xC0FF_EE00);
    let mut s = 0xC0FF_EE00u32;
    for _ in 0..1000 {
        assert_eq!(a.next_u32(), rng(&mut s));
    }
}

#[test]
fn rng_struct_is_copy_and_copies_diverge_independently() {
    let mut a = Rng::new(42);
    let b = a;
    let mut b2 = b;
    let first_a = a.next_u32();
    let first_b = b2.next_u32();
    assert_eq!(first_a, first_b);
    let _ = a.next_u32();
    let mut b3 = b;
    assert_eq!(b3.next_u32(), first_a);
}

#[test]
fn rng_debug_is_printable() {
    let s = format!("{:?}", Rng::new(7));
    assert!(s.contains("Rng"));
}

#[test]
fn rng_next_f32_is_in_unit_interval() {
    let mut r = Rng::new(0xabcd);
    for _ in 0..100_000 {
        let f = r.next_f32();
        assert!((0.0..1.0).contains(&f));
    }
}

#[test]
fn rng_next_f32_is_uint_to_float_of_next_u32() {
    let mut a = Rng::new(99);
    let mut b = Rng::new(99);
    for _ in 0..1000 {
        assert_eq!(a.next_f32(), openqmc::float::uint_to_float(b.next_u32()));
    }
}

#[test]
fn rng_next_2d_consumes_two_values_in_order() {
    let mut a = Rng::new(5);
    let mut b = Rng::new(5);
    for _ in 0..500 {
        let [x, y] = a.next_2d();
        assert_eq!(x, b.next_f32());
        assert_eq!(y, b.next_f32());
    }
}

#[test]
fn rng_streams_with_different_seeds_differ() {
    let mut first: Vec<u32> = Vec::new();
    for seed in common::PRIMES {
        first.push(Rng::new(seed).next_u32());
    }
    common::assert_all_distinct(&first, "first outputs");
}

#[test]
fn rng_stream_has_no_short_cycle() {
    let mut r = Rng::new(1);
    let outs: Vec<u32> = (0..(1 << 16)).map(|_| r.next_u32()).collect();
    common::assert_all_distinct(&outs, "rng stream");
}

#[test]
fn rng_mean_is_close_to_half() {
    let mut r = Rng::new(0x1234);
    let n = 1 << 18;
    let sum: f64 = (0..n).map(|_| r.next_f32() as f64).sum();
    let mean = sum / n as f64;
    assert!((mean - 0.5).abs() < 0.005, "mean {mean}");
}

#[test]
fn rng_variance_is_close_to_one_twelfth() {
    let mut r = Rng::new(0x5678);
    let n = 1 << 18;
    let mut sum = 0.0f64;
    let mut sq = 0.0f64;
    for _ in 0..n {
        let x = r.next_f32() as f64;
        sum += x;
        sq += x * x;
    }
    let mean = sum / n as f64;
    let var = sq / n as f64 - mean * mean;
    assert!((var - 1.0 / 12.0).abs() < 0.002, "variance {var}");
}

#[test]
fn rng_histogram_is_flat() {
    let mut r = Rng::new(0x9abc);
    let buckets = 64usize;
    let n = 1u64 << 18;
    let mut counts = vec![0u64; buckets];
    for _ in 0..n {
        counts[(r.next_f32() * buckets as f32) as usize] += 1;
    }
    let expected = n as f64 / buckets as f64;
    for (i, &c) in counts.iter().enumerate() {
        assert!(
            (c as f64 - expected).abs() / expected < 0.06,
            "bucket {i}: {c}"
        );
    }
}

#[test]
fn rng_consecutive_pairs_are_uncorrelated() {
    let mut r = Rng::new(0xdef0);
    let n = 1 << 17;
    let mut sxy = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for _ in 0..n {
        let x = r.next_f32() as f64 - 0.5;
        let y = r.next_f32() as f64 - 0.5;
        sxy += x * y;
        sx += x * x;
        sy += y * y;
    }
    let corr = sxy / (sx * sy).sqrt();
    assert!(corr.abs() < 0.01, "correlation {corr}");
}

#[test]
fn rng_seed_zero_is_valid() {
    let mut r = Rng::new(0);
    let a = r.next_u32();
    let b = r.next_u32();
    assert_eq!(a, hash(0));
    assert_ne!(a, b);
}

#[test]
fn functions_are_usable_in_const_context() {
    const A: u32 = state_transition(1);
    const B: u32 = output(1);
    const C: u32 = init();
    const D: u32 = init_seed(3);
    const E: u32 = hash(3);
    const R: Rng = Rng::new(3);
    assert_eq!(A, state_transition(1));
    assert_eq!(B, output(1));
    assert_eq!(C, init());
    assert_eq!(D, init_seed(3));
    assert_eq!(E, hash(3));
    let mut r = R;
    assert_eq!(r.next_u32(), hash(3));
}
