//! Port of `oqmc/pmj.h` — the progressive multi-jittered (0,2) sampler.
//!
//! A base 4D pattern for all 2^16 sample indices is pre-computed once (see
//! [`crate::stochastic`]), then looked up and XOR-scrambled at runtime. The
//! first pair of dimensions integrates as well as the Sobol sampler; the second
//! pair is a randomisation of the first (the (0,2) table only produces two
//! dimensions).
//!
//! Upstream requires a caller-allocated cache; this port builds it lazily in a
//! process global (see [`crate::sampler`] for the rationale).

use std::sync::OnceLock;

use crate::lookup::shuffled_scrambled_lookup;
use crate::pcg;
use crate::sampler::{Sampler, SamplerImpl};
use crate::state::{MAX_INDEX_SIZE, State64Bit};
use crate::stochastic::stochastic_pmj_init;

/// The pre-computed 4D PMJ pattern table (one entry per sample index). Shared
/// with the blue-noise PMJ variant.
pub(crate) fn pmj_cache() -> &'static [[u32; 4]] {
    static CACHE: OnceLock<Box<[[u32; 4]]>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut table = vec![[0u32; 4]; MAX_INDEX_SIZE];
        stochastic_pmj_init(MAX_INDEX_SIZE, &mut table);
        table.into_boxed_slice()
    })
}

/// Implementation behind [`PmjSampler`]. Use it through [`Sampler`].
#[derive(Clone, Copy, Debug)]
pub struct PmjImpl {
    state: State64Bit,
    cache: &'static [[u32; 4]],
}

impl SamplerImpl for PmjImpl {
    #[inline]
    fn from_pixel(x: i32, y: i32, frame: i32, index: i32) -> Self {
        Self {
            state: State64Bit::new(x, y, frame, index).pixel_decorrelate(),
            cache: pmj_cache(),
        }
    }

    #[inline]
    fn new_domain(&self, key: i32) -> Self {
        Self {
            state: self.state.new_domain(key),
            cache: self.cache,
        }
    }

    #[inline]
    fn new_domain_split(&self, key: i32, size: i32, index: i32) -> Self {
        Self {
            state: self.state.new_domain_split(key, size, index),
            cache: self.cache,
        }
    }

    #[inline]
    fn new_domain_distrib(&self, key: i32, index: i32) -> Self {
        Self {
            state: self.state.new_domain_distrib(key, index),
            cache: self.cache,
        }
    }

    #[inline]
    fn draw_block(&self) -> [u32; 4] {
        shuffled_scrambled_lookup::<4, 4>(
            self.state.sample_id as u32,
            pcg::output(self.state.pattern_id),
            self.cache,
        )
    }

    #[inline]
    fn draw_rnd_block(&self) -> [u32; 4] {
        self.state.draw_rnd::<4>()
    }

    #[inline]
    fn rng(&self) -> pcg::Rng {
        self.state.rng()
    }

    fn warm_cache() {
        pmj_cache();
    }
}

/// Progressive multi-jittered (0,2) sampler (Helmer et al.).
///
/// Draws from a lazily-built 2^16-entry 4D table, so per-draw cost is a cheap
/// lookup. The first dimension pair integrates like Sobol; the second pair is a
/// randomisation of the first. Call `PmjSampler::warm_cache()` to build the
/// table up front instead of on first draw. See
/// [`PmjBnSampler`](crate::PmjBnSampler) for the blue-noise variant.
///
/// ```
/// use openqmc::PmjSampler;
///
/// PmjSampler::warm_cache();
/// let root = PmjSampler::new(12, 34, 0, 7);
/// let [u, v] = root.new_domain(0).draw_sample_f32::<2>();
/// assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
/// ```
pub type PmjSampler = Sampler<PmjImpl>;
