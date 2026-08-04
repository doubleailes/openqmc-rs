# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/doubleailes/openqmc-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/doubleailes/openqmc-rs/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/doubleailes/openqmc-rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/doubleailes/openqmc-rs/releases/tag/v0.1.0
