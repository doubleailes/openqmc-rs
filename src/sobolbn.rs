//! Port of `oqmc/sobolbn.h` — the blue-noise variant of the Sobol sampler.
//! Same sequence as [`SobolSampler`](crate::SobolSampler), plus spatial
//! blue-noise dithering between pixels (progressive pixel sampling). Note it
//! does *not* pixel-decorrelate the state; the per-pixel (key, rank) table
//! lookup provides the decorrelation.

use crate::bntables::{self, table_value};
use crate::owen::shuffled_scrambled_sobol;
use crate::pcg;
use crate::sampler::{Sampler, SamplerImpl};
use crate::state::State64Bit;

/// Implementation behind [`SobolBnSampler`]. Use it through [`Sampler`].
#[derive(Clone, Copy, Debug)]
pub struct SobolBnImpl {
    state: State64Bit,
    key_table: &'static [u32],
    rank_table: &'static [u32],
}

impl SamplerImpl for SobolBnImpl {
    #[inline]
    fn from_pixel(x: i32, y: i32, frame: i32, index: i32) -> Self {
        Self {
            state: State64Bit::new(x, y, frame, index),
            key_table: bntables::sobol::key_table(),
            rank_table: bntables::sobol::rank_table(),
        }
    }

    #[inline]
    fn new_domain(&self, key: i32) -> Self {
        Self {
            state: self.state.new_domain(key),
            ..*self
        }
    }

    #[inline]
    fn new_domain_split(&self, key: i32, size: i32, index: i32) -> Self {
        Self {
            state: self.state.new_domain_split(key, size, index),
            ..*self
        }
    }

    #[inline]
    fn new_domain_distrib(&self, key: i32, index: i32) -> Self {
        Self {
            state: self.state.new_domain_distrib(key, index),
            ..*self
        }
    }

    #[inline]
    fn draw_block(&self) -> [u32; 4] {
        let t = table_value::<8, 8, 0>(
            self.state.pixel_id,
            pcg::output(self.state.pattern_id) as u16,
            self.key_table,
            self.rank_table,
        );
        shuffled_scrambled_sobol::<4>(self.state.sample_id as u32 ^ t.rank, t.key)
    }

    #[inline]
    fn draw_rnd_block(&self) -> [u32; 4] {
        self.state
            .new_domain(self.state.pixel_id as i32)
            .draw_rnd::<4>()
    }

    #[inline]
    fn rng(&self) -> pcg::Rng {
        self.state.new_domain(self.state.pixel_id as i32).rng()
    }

    fn warm_cache() {
        bntables::sobol::key_table();
        bntables::sobol::rank_table();
    }
}

/// Blue-noise variant of the [`SobolSampler`](crate::SobolSampler).
///
/// Same sequence, plus a per-pixel key/rank table lookup that distributes the
/// residual error as blue noise across the image — easier on the eye and on
/// denoisers. A good default if you are unsure which sampler to pick. The
/// tables decode lazily on first draw; call `SobolBnSampler::warm_cache()` to
/// pay that cost up front.
///
/// ```
/// use openqmc::SobolBnSampler;
///
/// let root = SobolBnSampler::new(12, 34, 0, 7);
/// let [u, v] = root.new_domain(0).draw_sample_f32::<2>();
/// assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
/// ```
pub type SobolBnSampler = Sampler<SobolBnImpl>;
