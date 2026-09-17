//! Port of `oqmc/sampler.h` — the public, static-polymorphic sampler interface.
//!
//! Upstream `SamplerInterface<Impl>` composes an internal implementation and
//! exposes a uniform domain-tree API. We model that with the [`SamplerImpl`]
//! trait plus the generic [`Sampler`] wrapper. Each impl computes a full
//! 4-dimensional block; the wrapper's const-generic `draw_*::<N>` methods
//! return the first `N` values, which is identical to computing `N` directly in
//! every upstream draw path (each output dimension is independent of the
//! requested depth).
//!
//! Divergence from the C++ headers: upstream takes a caller-allocated `void*`
//! cache, a design driven by GPU memory management (see `sampler.h`). This CPU
//! port instead builds any required tables in lazily-initialised process
//! globals, so every `Sampler<T>` stays a small `Copy + Send + 'static` value.

use crate::float::uint_to_float;
use crate::pcg;
use crate::range::uint_to_range;

/// Internal sampler implementation. Not called directly — use [`Sampler`].
pub trait SamplerImpl: Copy {
    /// Construct from pixel/frame/sample indices (index must be `>= 0`).
    fn from_pixel(x: i32, y: i32, frame: i32, index: i32) -> Self;
    /// Derive a child domain (independent 4D pattern).
    fn new_domain(&self, key: i32) -> Self;
    /// Derive a fixed-rate split domain.
    fn new_domain_split(&self, key: i32, size: i32, index: i32) -> Self;
    /// Derive an adaptive-rate (local) split domain.
    fn new_domain_distrib(&self, key: i32, index: i32) -> Self;
    /// Compute the 4D high-quality sample block for this domain.
    fn draw_block(&self) -> [u32; 4];
    /// Compute the 4D low-quality pseudo-random block for this domain.
    fn draw_rnd_block(&self) -> [u32; 4];
    /// A sequential PRNG stream for this domain (for unbounded draws).
    fn rng(&self) -> pcg::Rng;
    /// Force any lazily-built cache to initialise. Optional; draws trigger it
    /// anyway. Default no-op for cache-free samplers.
    fn warm_cache() {}
}

#[inline]
fn take<const N: usize>(block: [u32; 4]) -> [u32; N] {
    const { assert!(N >= 1 && N <= 4, "Draw size must be within [1, 4]") };
    let mut out = [0u32; N];
    out.copy_from_slice(&block[..N]);
    out
}

/// The public sampler. Different `Impl`s are interchangeable at every call site.
///
/// A sampler is immutable: derive children with `new_domain*`, draw values with
/// `draw_*`. Deriving a domain is cheap (an LCG state transition); drawing is
/// where the sequence work happens.
#[derive(Clone, Copy, Debug)]
pub struct Sampler<T: SamplerImpl> {
    imp: T,
}

impl<T: SamplerImpl> Sampler<T> {
    /// Eagerly initialise any lazily-built cache for this sampler type.
    #[inline]
    pub fn warm_cache() {
        T::warm_cache();
    }

    /// Construct from pixel coordinate, frame and sample index (`index >= 0`).
    #[inline]
    pub fn new(x: i32, y: i32, frame: i32, index: i32) -> Self {
        debug_assert!(index >= 0);
        Self {
            imp: T::from_pixel(x, y, frame, index),
        }
    }

    /// Derive a child domain with an independent 4D pattern.
    #[inline]
    pub fn new_domain(&self, key: i32) -> Self {
        Self {
            imp: self.imp.new_domain(key),
        }
    }

    /// Derive a split domain with a fixed sample-rate multiplier.
    #[inline]
    pub fn new_domain_split(&self, key: i32, size: i32, index: i32) -> Self {
        debug_assert!(size > 0 && index >= 0);
        Self {
            imp: self.imp.new_domain_split(key, size, index),
        }
    }

    /// Derive a split domain with a local (adaptive) distribution.
    #[inline]
    pub fn new_domain_distrib(&self, key: i32, index: i32) -> Self {
        debug_assert!(index >= 0);
        Self {
            imp: self.imp.new_domain_distrib(key, index),
        }
    }

    /// Derive a split domain with a global (adaptive) distribution
    /// (`new_domain(key).new_domain(index)`).
    #[inline]
    pub fn new_domain_chain(&self, key: i32, index: i32) -> Self {
        debug_assert!(index >= 0);
        self.new_domain(key).new_domain(index)
    }

    /// Draw `N` (1..=4) high-quality integer sample values, each in `[0, 2^32)`.
    #[inline]
    pub fn draw_sample<const N: usize>(&self) -> [u32; N] {
        take(self.imp.draw_block())
    }

    /// Draw `N` high-quality sample values within `[0, range)`.
    #[inline]
    pub fn draw_sample_range<const N: usize>(&self, range: u32) -> [u32; N] {
        debug_assert!(range > 0);
        let block = self.draw_sample::<N>();
        let mut out = [0u32; N];
        for i in 0..N {
            out[i] = uint_to_range(block[i], range);
        }
        out
    }

    /// Draw `N` high-quality sample values within `[0, 1)`.
    #[inline]
    pub fn draw_sample_f32<const N: usize>(&self) -> [f32; N] {
        let block = self.draw_sample::<N>();
        let mut out = [0.0f32; N];
        for i in 0..N {
            out[i] = uint_to_float(block[i]);
        }
        out
    }

    /// Draw `N` low-quality pseudo-random integer values, each in `[0, 2^32)`.
    #[inline]
    pub fn draw_rnd<const N: usize>(&self) -> [u32; N] {
        take(self.imp.draw_rnd_block())
    }

    /// Draw `N` low-quality pseudo-random values within `[0, range)`.
    #[inline]
    pub fn draw_rnd_range<const N: usize>(&self, range: u32) -> [u32; N] {
        debug_assert!(range > 0);
        let block = self.draw_rnd::<N>();
        let mut out = [0u32; N];
        for i in 0..N {
            out[i] = uint_to_range(block[i], range);
        }
        out
    }

    /// Draw `N` low-quality pseudo-random values within `[0, 1)`.
    #[inline]
    pub fn draw_rnd_f32<const N: usize>(&self) -> [f32; N] {
        let block = self.draw_rnd::<N>();
        let mut out = [0.0f32; N];
        for i in 0..N {
            out[i] = uint_to_float(block[i]);
        }
        out
    }

    /// A sequential PRNG stream seeded from this domain, for unbounded random
    /// draws (Russian roulette, delta tracking) where QMC stratification does not
    /// apply.
    #[inline]
    pub fn rng(&self) -> pcg::Rng {
        self.imp.rng()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::state::State64Bit;

    /// A minimal deterministic implementation to exercise the wrapper alone.
    #[derive(Clone, Copy, Debug)]
    struct Dummy {
        state: State64Bit,
    }

    impl SamplerImpl for Dummy {
        fn from_pixel(x: i32, y: i32, frame: i32, index: i32) -> Self {
            Self {
                state: State64Bit::new(x, y, frame, index),
            }
        }
        fn new_domain(&self, key: i32) -> Self {
            Self {
                state: self.state.new_domain(key),
            }
        }
        fn new_domain_split(&self, key: i32, size: i32, index: i32) -> Self {
            Self {
                state: self.state.new_domain_split(key, size, index),
            }
        }
        fn new_domain_distrib(&self, key: i32, index: i32) -> Self {
            Self {
                state: self.state.new_domain_distrib(key, index),
            }
        }
        fn draw_block(&self) -> [u32; 4] {
            let p = self.state.pattern_id;
            [p, p ^ 1, p ^ 2, p ^ 3]
        }
        fn draw_rnd_block(&self) -> [u32; 4] {
            self.state.draw_rnd::<4>()
        }
        fn rng(&self) -> pcg::Rng {
            self.state.rng()
        }
    }

    type DummySampler = Sampler<Dummy>;

    #[test]
    fn take_returns_the_prefix() {
        let block = [10u32, 20, 30, 40];
        assert_eq!(take::<1>(block), [10]);
        assert_eq!(take::<2>(block), [10, 20]);
        assert_eq!(take::<3>(block), [10, 20, 30]);
        assert_eq!(take::<4>(block), block);
    }

    #[test]
    fn default_warm_cache_is_a_no_op() {
        Dummy::warm_cache();
        DummySampler::warm_cache();
    }

    #[test]
    fn draw_sample_is_take_of_draw_block() {
        let s = DummySampler::new(1, 2, 3, 4).new_domain(5);
        let block = s.imp.draw_block();
        assert_eq!(s.draw_sample::<4>(), block);
        assert_eq!(s.draw_sample::<2>(), [block[0], block[1]]);
        assert_eq!(s.draw_sample::<1>(), [block[0]]);
    }

    #[test]
    fn draw_rnd_is_take_of_draw_rnd_block() {
        let s = DummySampler::new(1, 2, 3, 4).new_domain(5);
        let block = s.imp.draw_rnd_block();
        assert_eq!(s.draw_rnd::<4>(), block);
        assert_eq!(s.draw_rnd::<3>(), [block[0], block[1], block[2]]);
    }

    #[test]
    fn conversions_apply_per_element() {
        let s = DummySampler::new(1, 2, 3, 4).new_domain(5);
        let block = s.draw_sample::<4>();
        let f = s.draw_sample_f32::<4>();
        let r = s.draw_sample_range::<4>(1000);
        for k in 0..4 {
            assert_eq!(f[k], uint_to_float(block[k]));
            assert_eq!(r[k], uint_to_range(block[k], 1000));
        }
        let block = s.draw_rnd::<4>();
        let f = s.draw_rnd_f32::<4>();
        let r = s.draw_rnd_range::<4>(7);
        for k in 0..4 {
            assert_eq!(f[k], uint_to_float(block[k]));
            assert_eq!(r[k], uint_to_range(block[k], 7));
        }
    }

    #[test]
    fn domain_methods_forward_to_the_impl() {
        let s = DummySampler::new(1, 2, 3, 4);
        assert_eq!(s.new_domain(7).imp.state, s.imp.state.new_domain(7));
        assert_eq!(
            s.new_domain_split(7, 4, 1).imp.state,
            s.imp.state.new_domain_split(7, 4, 1)
        );
        assert_eq!(
            s.new_domain_distrib(7, 1).imp.state,
            s.imp.state.new_domain_distrib(7, 1)
        );
        assert_eq!(
            s.new_domain_chain(7, 1).imp.state,
            s.imp.state.new_domain(7).new_domain(1)
        );
    }

    #[test]
    fn rng_forwards_to_the_impl() {
        let s = DummySampler::new(1, 2, 3, 4);
        assert_eq!(s.rng().next_u32(), s.imp.rng().next_u32());
        assert_eq!(s.rng().next_u32(), s.draw_rnd::<1>()[0]);
    }

    #[test]
    fn wrapper_is_copy_and_debug() {
        let s = DummySampler::new(1, 2, 3, 4);
        let t = s;
        assert_eq!(s.draw_sample::<4>(), t.draw_sample::<4>());
        assert!(format!("{s:?}").contains("Sampler"));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn negative_index_is_rejected_in_debug() {
        let _ = DummySampler::new(0, 0, 0, -1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn zero_split_size_is_rejected_in_debug() {
        let _ = DummySampler::new(0, 0, 0, 0).new_domain_split(0, 0, 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn negative_split_index_is_rejected_in_debug() {
        let _ = DummySampler::new(0, 0, 0, 0).new_domain_split(0, 2, -1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn negative_distrib_index_is_rejected_in_debug() {
        let _ = DummySampler::new(0, 0, 0, 0).new_domain_distrib(0, -1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn negative_chain_index_is_rejected_in_debug() {
        let _ = DummySampler::new(0, 0, 0, 0).new_domain_chain(0, -1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn zero_range_is_rejected_in_debug() {
        let _ = DummySampler::new(0, 0, 0, 0).draw_sample_range::<1>(0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn zero_rnd_range_is_rejected_in_debug() {
        let _ = DummySampler::new(0, 0, 0, 0).draw_rnd_range::<1>(0);
    }
}
