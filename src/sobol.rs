//! Port of `oqmc/sobol.h` — the Owen-scrambled Sobol sampler.
//!
//! No cache: all samples are computed on the fly. The per-draw cost is higher
//! than the other samplers, but Owen scrambling gives excellent random error
//! cancellation and a very high rate of integration for smooth functions.

use crate::owen::shuffled_scrambled_sobol;
use crate::pcg;
use crate::sampler::{Sampler, SamplerImpl};
use crate::state::State64Bit;

/// Implementation behind [`SobolSampler`]. Use it through [`Sampler`].
#[derive(Clone, Copy, Debug)]
pub struct SobolImpl {
    state: State64Bit,
}

impl SamplerImpl for SobolImpl {
    #[inline]
    fn from_pixel(x: i32, y: i32, frame: i32, index: i32) -> Self {
        Self {
            state: State64Bit::new(x, y, frame, index).pixel_decorrelate(),
        }
    }

    #[inline]
    fn new_domain(&self, key: i32) -> Self {
        Self {
            state: self.state.new_domain(key),
        }
    }

    #[inline]
    fn new_domain_split(&self, key: i32, size: i32, index: i32) -> Self {
        Self {
            state: self.state.new_domain_split(key, size, index),
        }
    }

    #[inline]
    fn new_domain_distrib(&self, key: i32, index: i32) -> Self {
        Self {
            state: self.state.new_domain_distrib(key, index),
        }
    }

    #[inline]
    fn draw_block(&self) -> [u32; 4] {
        shuffled_scrambled_sobol::<4>(
            self.state.sample_id as u32,
            pcg::output(self.state.pattern_id),
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
}

/// Owen-scrambled Sobol sampler (Burley 2020, Ahmed 2024).
///
/// Cache-free; every sample is computed on the fly. Highest rate of integration
/// for smooth functions of the six samplers, at the highest per-draw cost. See
/// [`SobolBnSampler`](crate::SobolBnSampler) for the blue-noise variant.
///
/// ```
/// use openqmc::SobolSampler;
///
/// let root = SobolSampler::new(12, 34, 0, 7);
/// let [u, v] = root.new_domain(0).draw_sample_f32::<2>();
/// assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
/// ```
pub type SobolSampler = Sampler<SobolImpl>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_pixel_decorrelates_the_state() {
        let imp = SobolImpl::from_pixel(3, 4, 5, 6);
        assert_eq!(imp.state, State64Bit::new(3, 4, 5, 6).pixel_decorrelate());
    }

    #[test]
    fn domain_methods_forward_to_the_state() {
        let imp = SobolImpl::from_pixel(3, 4, 5, 6);
        assert_eq!(imp.new_domain(1).state, imp.state.new_domain(1));
        assert_eq!(
            imp.new_domain_split(1, 2, 1).state,
            imp.state.new_domain_split(1, 2, 1)
        );
        assert_eq!(
            imp.new_domain_distrib(1, 2).state,
            imp.state.new_domain_distrib(1, 2)
        );
    }

    #[test]
    fn draw_block_is_the_core_seeded_with_output_of_pattern() {
        let imp = SobolImpl::from_pixel(3, 4, 5, 6).new_domain(2);
        assert_eq!(
            imp.draw_block(),
            shuffled_scrambled_sobol::<4>(
                imp.state.sample_id as u32,
                pcg::output(imp.state.pattern_id)
            )
        );
        assert_eq!(imp.draw_rnd_block(), imp.state.draw_rnd::<4>());
        assert_eq!(imp.rng().next_u32(), imp.state.rng().next_u32());
    }

    #[test]
    fn impl_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<SobolImpl>(), 8);
        assert_eq!(std::mem::size_of::<SobolSampler>(), 8);
    }
}
