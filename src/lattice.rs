//! Port of `oqmc/lattice.h` — the rank-1 lattice sampler.
//!
//! No cache: samples are computed on the fly with a low per-draw cost. Runtime
//! performance is high, though the rate of integration per pixel can be lower
//! than the Sobol or PMJ samplers.

use crate::pcg;
use crate::rank1::shuffled_rotated_lattice;
use crate::sampler::{Sampler, SamplerImpl};
use crate::state::State64Bit;

/// Implementation behind [`LatticeSampler`]. Use it through [`Sampler`].
#[derive(Clone, Copy, Debug)]
pub struct LatticeImpl {
    state: State64Bit,
}

impl SamplerImpl for LatticeImpl {
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
        // The lattice takes the raw pattern id (it applies pcg::output / pcg::rng
        // internally for the shuffle seed and the toroidal shifts).
        shuffled_rotated_lattice::<4>(self.state.sample_id as u32, self.state.pattern_id)
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

/// Rank-1 lattice sampler (Hickernell et al.).
///
/// Cache-free and the cheapest per draw of the six samplers; the rate of
/// integration per pixel can be lower than Sobol or PMJ. Prefer it when draw
/// cost dominates. See [`LatticeBnSampler`](crate::LatticeBnSampler) for the
/// blue-noise variant.
///
/// ```
/// use openqmc::LatticeSampler;
///
/// let root = LatticeSampler::new(12, 34, 0, 7);
/// let [u, v] = root.new_domain(0).draw_sample_f32::<2>();
/// assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
/// ```
pub type LatticeSampler = Sampler<LatticeImpl>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_pixel_decorrelates_the_state() {
        let imp = LatticeImpl::from_pixel(3, 4, 5, 6);
        assert_eq!(imp.state, State64Bit::new(3, 4, 5, 6).pixel_decorrelate());
    }

    #[test]
    fn draw_block_is_the_core_on_the_raw_pattern() {
        let imp = LatticeImpl::from_pixel(3, 4, 5, 6).new_domain(2);
        assert_eq!(
            imp.draw_block(),
            shuffled_rotated_lattice::<4>(imp.state.sample_id as u32, imp.state.pattern_id)
        );
        assert_eq!(imp.draw_rnd_block(), imp.state.draw_rnd::<4>());
        assert_eq!(imp.rng().next_u32(), imp.state.rng().next_u32());
    }

    #[test]
    fn domain_methods_forward_to_the_state() {
        let imp = LatticeImpl::from_pixel(3, 4, 5, 6);
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
    fn impl_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<LatticeImpl>(), 8);
    }
}
