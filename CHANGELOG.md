# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-09

**Breaking: sample values change.** `SobolSampler` and `SobolBnSampler` now
produce different values in dimensions 2 and 3, following upstream OpenQMC
`b6a204e` ("Replace Sobol matrices with SZ sequences"). Renders using either
sampler will not reproduce images made with 0.2.x. Dimensions 0 and 1 are
unchanged. PMJ, lattice and their blue-noise variants are unaffected. Output
is again bit-for-bit identical to upstream, now at upstream HEAD (`d6c40b5`)
rather than `b6a204e~1`.

### Changed

- **Behaviour change:** the generator matrices behind `SobolSampler`,
  `SobolBnSampler` and `owen::shuffled_scrambled_sobol` are now the SZ
  sequences of Ahmed et al., *SZ Sequences: Binary-Constructed
  (0, 2^q)-Sequences* (SIGGRAPH Asia 2025). They replace the classic Sobol
  (Joe & Kuo) direction numbers. Every pair of the four dimensions is now a
  (0, 2)-sequence in base 4, and the four together are a (0, 4)-sequence in
  base 16. Before, the pairs (1, 3) and (2, 3) failed a 16×16 grid at 256
  points. The blue-noise tables still apply, because dimensions 0 and 1 are
  unchanged.
- **Breaking API:** `owen::sobol_reversed_index` is renamed
  `owen::sz_reversed_basis`, matching upstream's `oqmc::szReversedBasis`, and
  it returns the new SZ values. There is no deprecated alias, because the old
  name would silently return different numbers.
- The core is still evaluated from compile-time byte tables, now built from
  the SZ matrices. On the matched benchmark, Sobol and Sobol-BN draws are
  1.4–1.6× faster than upstream's C++ at default flags, and about 3× faster
  with `target-cpu=native`. Upstream dropped its SIMD Sobol paths in the same
  commit. See `tools/README.md`.

### Added

- `owen::sz_to_reversed`, a port of upstream's `oqmc::szToReversed`. It is
  the SZ program on a natural-order index, so
  `sz_to_reversed(i, d) == sz_reversed_basis(reverse_bits16(i), d)`.
- Exhaustive tests for the SZ core. The byte tables are checked against both
  of upstream's shift-mask-xor programs. Further tests cover the eq. 13
  chaining relation (`P · SZ[2] == SZ[3]`), the upstream host-path draw,
  base-4 (0, 2)-sequence stratification for every dimension pair, and the
  base-16 (0, 4)-sequence property of all four dimensions.
- A large dependency-free test suite (about 800 new tests) covering every
  module: exhaustive and property tests for the utility headers (`encode`,
  `float`, `range`, `rotate`, `reverse`, `permute`, `pcg`), (0,2)-net and
  stratification checks for the Sobol, lattice and PMJ cores, checks on the
  bundled blue-noise tables, `State64Bit` domain-tree algebra, a behavioural
  suite stamped out for all six samplers, and cross-checks that rebuild each
  sampler's output from the public building blocks.
- `benches/upstream.rs` and `tools/bench_ref.cpp`: a matched Rust / C++
  benchmark using upstream's own `benchmark` workload, so the port can be timed
  against the reference headers. Both sides report `init_us` as a cold start,
  timing each sampler in a fresh process. `tools/README.md` records the
  results and explains the `target-cpu=native` speedup (GFNI bit reversal).

### Fixed

- `tools/README.md`: upstream now requires C++17, so the reference generator is
  compiled with `-std=c++17` (was `-std=c++14`, which no longer builds).

## [0.2.4] - 2026-09-26

Performance. No API or behavioural changes — sample values remain bit-for-bit
identical to upstream OpenQMC and to every previous release.

### Changed

- `sobol_reversed_index` (the GF(2) direction-matrix product behind
  `SobolSampler` and `SobolBnSampler`) evaluates the product a byte at a time
  from two compile-time 256-entry tables per dimension instead of a 16-step
  loop over the index bits. The product is linear over GF(2), so the two
  lookups XORed together are exactly the loop's result; a new exhaustive test
  checks every index in every dimension against the loop, and the upstream
  golden vectors are unchanged. In crust-render, where Sobol draws were
  16–24% of render time, this cut render instructions by 10–18% and render
  time by 6–17%.

## [0.2.0] - 2026-08-04

Release polish. No API or behavioural changes — sample values remain bit-for-bit
identical to upstream OpenQMC and to 0.1.x.

### Added

- Full crates.io metadata: repository, documentation, readme, keywords,
  categories, authors, and an explicit MSRV (`rust-version = "1.85"`).
- This changelog.
- Module-level documentation for every module and doc comments for every public
  item; documentation is now enforced with `#![warn(missing_docs)]` and
  `#![deny(rustdoc::broken_intra_doc_links)]`.
- `#![forbid(unsafe_code)]` (the crate already contained no `unsafe`).
- GitHub Actions CI (rustfmt, clippy, tests, rustdoc, MSRV check) and a
  publish-on-tag release workflow.
- README badges, MSRV note, and crates.io installation instructions.

### Changed

- The `tools/` reference-generator sources are no longer packaged into the
  published crate.

## [0.1.1] - 2026-08-04

### Changed

- Version bump; no functional changes.

## [0.1.0] - 2026-08-04

### Added

- Initial release: a faithful Rust port of
  [AcademySoftwareFoundation/openqmc](https://github.com/AcademySoftwareFoundation/openqmc).
- Six interchangeable samplers behind the pass-by-value domain-tree `Sampler`
  API: `SobolSampler`, `LatticeSampler`, `PmjSampler` and their blue-noise
  variants `SobolBnSampler`, `LatticeBnSampler`, `PmjBnSampler`.
- Bundled upstream blue-noise key/rank tables (`src/data/*.bin`).
- Golden-vector test suite generated from the upstream C++ headers
  (`tests/golden_upstream.rs`, regenerable via `tools/ref_gen.cpp`),
  verifying bit-for-bit parity.

[Unreleased]: https://github.com/doubleailes/openqmc-rs/compare/0.3.0...HEAD
[0.3.0]: https://github.com/doubleailes/openqmc-rs/compare/0.2.4...0.3.0
[0.2.4]: https://github.com/doubleailes/openqmc-rs/compare/0.2.3...0.2.4
[0.2.0]: https://github.com/doubleailes/openqmc-rs/releases/tag/0.2.0
