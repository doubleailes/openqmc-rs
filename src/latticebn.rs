//! Port of `oqmc/latticebn.h` — the blue-noise variant of the rank-1 lattice
//! sampler. Same sequence as [`LatticeSampler`](crate::LatticeSampler), plus
//! spatial blue-noise dithering between pixels. State is not
//! pixel-decorrelated (the table lookup does that).

use crate::bntables::{self, table_value};
use crate::pcg;
use crate::rank1::shuffled_rotated_lattice;
use crate::sampler::{Sampler, SamplerImpl};
use crate::state::State64Bit;

/// Implementation behind [`LatticeBnSampler`]. Use it through [`Sampler`].
#[derive(Clone, Copy, Debug)]
pub struct LatticeBnImpl {
    state: State64Bit,
    key_table: &'static [u32],
    rank_table: &'static [u32],
}

impl SamplerImpl for LatticeBnImpl {
    #[inline]
    fn from_pixel(x: i32, y: i32, frame: i32, index: i32) -> Self {
        Self {
            state: State64Bit::new(x, y, frame, index),
            key_table: bntables::lattice::key_table(),
            rank_table: bntables::lattice::rank_table(),
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
        shuffled_rotated_lattice::<4>(self.state.sample_id as u32 ^ t.rank, t.key)
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
        bntables::lattice::key_table();
        bntables::lattice::rank_table();
    }
}

/// Blue-noise variant of the [`LatticeSampler`](crate::LatticeSampler).
///
/// Same sequence, plus a per-pixel key/rank table lookup that distributes the
/// residual error as blue noise across the image. The tables decode lazily on
/// first draw; call `LatticeBnSampler::warm_cache()` to pay that cost up front.
///
/// ```
/// use openqmc::LatticeBnSampler;
///
/// let root = LatticeBnSampler::new(12, 34, 0, 7);
/// let [u, v] = root.new_domain(0).draw_sample_f32::<2>();
/// assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
/// ```
pub type LatticeBnSampler = Sampler<LatticeBnImpl>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_pixel_keeps_the_state_correlated() {
        let imp = LatticeBnImpl::from_pixel(3, 4, 5, 6);
        assert_eq!(imp.state, State64Bit::new(3, 4, 5, 6));
        assert!(std::ptr::eq(imp.key_table, bntables::lattice::key_table()));
        assert!(std::ptr::eq(
            imp.rank_table,
            bntables::lattice::rank_table()
        ));
    }

    #[test]
    fn domain_methods_keep_the_tables() {
        let imp = LatticeBnImpl::from_pixel(3, 4, 5, 6);
        for child in [
            imp.new_domain(1),
            imp.new_domain_split(1, 2, 1),
            imp.new_domain_distrib(1, 2),
        ] {
            assert!(std::ptr::eq(child.key_table, imp.key_table));
            assert!(std::ptr::eq(child.rank_table, imp.rank_table));
        }
        assert_eq!(imp.new_domain(1).state, imp.state.new_domain(1));
    }

    #[test]
    fn draw_block_reads_the_table_shifted_by_the_pattern() {
        let imp = LatticeBnImpl::from_pixel(3, 4, 5, 6).new_domain(2);
        let t = table_value::<8, 8, 0>(
            imp.state.pixel_id,
            pcg::output(imp.state.pattern_id) as u16,
            imp.key_table,
            imp.rank_table,
        );
        assert_eq!(
            imp.draw_block(),
            shuffled_rotated_lattice::<4>(imp.state.sample_id as u32 ^ t.rank, t.key)
        );
    }

    #[test]
    fn rnd_paths_decorrelate_by_pixel() {
        let imp = LatticeBnImpl::from_pixel(3, 4, 5, 6).new_domain(2);
        let expected = imp.state.new_domain(imp.state.pixel_id as i32);
        assert_eq!(imp.draw_rnd_block(), expected.draw_rnd::<4>());
        assert_eq!(imp.rng().next_u32(), expected.rng().next_u32());
    }

    #[test]
    fn warm_cache_decodes_the_tables() {
        LatticeBnImpl::warm_cache();
        LatticeBnSampler::warm_cache();
        assert_eq!(bntables::lattice::key_table().len(), bntables::SIZE);
    }
}
