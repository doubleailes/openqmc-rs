//! Port of `oqmc/bntables.h` — pre-computed, optimised blue-noise key/rank
//! tables used to give the base samplers a spatial blue-noise error
//! distribution (generalising Belcour & Heitz). A single table pair serves all
//! domains: the lookup is shifted toroidally per domain by a random offset.
//!
//! The upstream tables ship as text headers; this port bundles them as
//! little-endian binary blobs (converted 1:1 from `include/oqmc/data/**`) and
//! decodes them lazily into process globals.

use std::sync::OnceLock;

use crate::encode::{EncodeKey, decode_bits16, encode_bits16};

/// Pixel-x precision for the tables (256 pixels).
pub const XBITS: u32 = 8;
/// Pixel-y precision for the tables (256 pixels).
pub const YBITS: u32 = 8;
/// Table length (2^16).
pub const SIZE: usize = 1 << (XBITS + YBITS);

/// A key/rank pair used to randomise a sequence.
#[derive(Clone, Copy, Debug)]
pub struct TableReturnValue {
    /// Scramble key for the sequence at this pixel.
    pub key: u32,
    /// Sample-index rank (XORed into the index) for this pixel.
    pub rank: u32,
}

/// Look up a key/rank pair, shifting the pixel coordinate toroidally by `shift`.
#[inline]
pub fn table_value<const XB: u32, const YB: u32, const ZB: u32>(
    pixel: u16,
    shift: u16,
    key_table: &[u32],
    rank_table: &[u32],
) -> TableReturnValue {
    let p = decode_bits16::<XB, YB, ZB>(pixel);
    let s = decode_bits16::<XB, YB, ZB>(shift);
    let index = encode_bits16::<XB, YB, ZB>(EncodeKey {
        x: p.x + s.x,
        y: p.y + s.y,
        z: p.z + s.z,
    }) as usize;

    TableReturnValue {
        key: key_table[index],
        rank: rank_table[index],
    }
}

fn decode_table(bytes: &'static [u8]) -> Box<[u32]> {
    debug_assert_eq!(bytes.len(), SIZE * 4);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

macro_rules! table_pair {
    ($module:ident, $keys:literal, $ranks:literal) => {
        #[doc = concat!("Optimised blue-noise key/rank tables for the ", stringify!($module), " sampler, decoded lazily from the bundled blobs.")]
        pub mod $module {
            use super::{OnceLock, decode_table};

            /// Optimised blue-noise key table.
            pub fn key_table() -> &'static [u32] {
                static T: OnceLock<Box<[u32]>> = OnceLock::new();
                T.get_or_init(|| decode_table(include_bytes!($keys)))
            }

            /// Optimised blue-noise rank table.
            pub fn rank_table() -> &'static [u32] {
                static T: OnceLock<Box<[u32]>> = OnceLock::new();
                T.get_or_init(|| decode_table(include_bytes!($ranks)))
            }
        }
    };
}

table_pair!(sobol, "data/sobol_keys.bin", "data/sobol_ranks.bin");
table_pair!(lattice, "data/lattice_keys.bin", "data/lattice_ranks.bin");
table_pair!(pmj, "data/pmj_keys.bin", "data/pmj_ranks.bin");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_decode_to_full_length() {
        assert_eq!(sobol::key_table().len(), SIZE);
        assert_eq!(sobol::rank_table().len(), SIZE);
        assert_eq!(lattice::key_table().len(), SIZE);
        assert_eq!(pmj::rank_table().len(), SIZE);
    }

    #[test]
    fn tables_decode_little_endian() {
        let raw: &[u8] = include_bytes!("data/sobol_keys.bin");
        let first = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let last = u32::from_le_bytes([
            raw[raw.len() - 4],
            raw[raw.len() - 3],
            raw[raw.len() - 2],
            raw[raw.len() - 1],
        ]);
        assert_eq!(sobol::key_table()[0], first);
        assert_eq!(sobol::key_table()[SIZE - 1], last);
    }

    #[test]
    fn all_six_blobs_have_the_expected_size() {
        assert_eq!(include_bytes!("data/sobol_keys.bin").len(), SIZE * 4);
        assert_eq!(include_bytes!("data/sobol_ranks.bin").len(), SIZE * 4);
        assert_eq!(include_bytes!("data/lattice_keys.bin").len(), SIZE * 4);
        assert_eq!(include_bytes!("data/lattice_ranks.bin").len(), SIZE * 4);
        assert_eq!(include_bytes!("data/pmj_keys.bin").len(), SIZE * 4);
        assert_eq!(include_bytes!("data/pmj_ranks.bin").len(), SIZE * 4);
    }

    #[test]
    fn tables_are_decoded_once() {
        assert!(std::ptr::eq(pmj::key_table(), pmj::key_table()));
        assert!(std::ptr::eq(lattice::rank_table(), lattice::rank_table()));
    }

    #[test]
    fn ranks_are_below_128() {
        for t in [
            sobol::rank_table(),
            lattice::rank_table(),
            pmj::rank_table(),
        ] {
            assert!(t.iter().all(|&r| r < 128));
        }
    }

    #[test]
    fn table_value_with_zero_shift_is_a_direct_index() {
        let keys = sobol::key_table();
        let ranks = sobol::rank_table();
        for p in [0u16, 1, 255, 256, 65535] {
            let t = table_value::<XBITS, YBITS, 0>(p, 0, keys, ranks);
            assert_eq!(t.key, keys[p as usize]);
            assert_eq!(t.rank, ranks[p as usize]);
        }
    }

    #[test]
    fn table_value_shift_wraps_toroidally() {
        let keys = sobol::key_table();
        let ranks = sobol::rank_table();
        // pixel (255, 255) shifted by (1, 1) lands on (0, 0).
        let t = table_value::<XBITS, YBITS, 0>(0xffff, 0x0101, keys, ranks);
        assert_eq!(t.key, keys[0]);
        assert_eq!(t.rank, ranks[0]);
    }
}
