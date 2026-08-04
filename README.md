# openqmc-rs

A faithful Rust port of [AcademySoftwareFoundation/openqmc][upstream] — quasi-Monte Carlo
samplers for rendering, with a pass-by-value domain-tree API and blue-noise variants.

The port reproduces the upstream algorithms and their sample values **bit-for-bit**, verified
against golden vectors generated directly from the C++ headers (see
[Golden vectors](#golden-vectors)).

[upstream]: https://github.com/AcademySoftwareFoundation/openqmc

## Why QMC

Monte Carlo integrators (path tracers, volume integrators) converge faster when their samples
are stratified rather than merely random. OpenQMC provides low-discrepancy sequences that stay
well-distributed *and* stay decorrelated between pixels, plus blue-noise variants that shape the
residual error into a spatial pattern the eye — and denoisers — handle much better than white
noise.

## Samplers

Six samplers, all interchangeable behind the same [`Sampler`] API:

| Sampler | Construction | Cache | Notes |
| --- | --- | --- | --- |
| `SobolSampler` | Owen-scrambled Sobol (Burley 2020, Ahmed 2024) | none | Highest rate of integration for smooth functions; highest per-draw cost. |
| `LatticeSampler` | rank-1 lattice (Hickernell et al.) | none | Cheapest per draw; lower per-pixel rate of integration. |
| `PmjSampler` | progressive multi-jittered (0,2) (Helmer et al.) | 2^16 × 4D table, built lazily | First dimension pair integrates like Sobol; second pair is a randomisation of the first. |
| `SobolBnSampler` | as Sobol | blue-noise tables | Blue-noise variant: same sequence, spatially dithered between pixels. |
| `LatticeBnSampler` | as Lattice | blue-noise tables | " |
| `PmjBnSampler` | as PMJ | PMJ + blue-noise tables | " |

The blue-noise variants add a per-pixel key/rank table lookup (generalising Belcour & Heitz) that
distributes error as blue noise across the image. They do not pixel-decorrelate the state — the
table lookup provides that.

If you are unsure, start with `SobolBnSampler`; switch to `LatticeSampler` if draw cost dominates.

## Usage

Add the dependency:

```toml
[dependencies]
openqmc-rs = { git = "https://github.com/doubleailes/openqmc-rs" }
```

The crate's library name is `openqmc`, so import it as `use openqmc::...;`.

Construct a root sampler per (pixel, frame, sample index), derive independent 4-dimensional
sub-patterns with `new_domain`, and draw up to four dimensions per domain:

```rust
use openqmc::SobolSampler;

for index in 0..spp {
    let root = SobolSampler::new(pixel_x, pixel_y, frame, index);

    // Domain 0: pixel jitter + lens position.
    let camera = root.new_domain(0);
    let [jitter_x, jitter_y, lens_u, lens_v] = camera.draw_sample_f32::<4>();

    // Domain 1: first bounce direction. Independent of domain 0's pattern.
    let [u, v] = root.new_domain(1).draw_sample_f32::<2>();

    // Unbounded / incidental randomness — no stratification to preserve.
    let mut rng = root.new_domain(2).rng();
    let roulette = rng.next_f32();
}
```

Every sampler is `Copy + Send + 'static` and 8 bytes of state, so pass it by value down the
integrator; there is nothing to borrow and nothing to reset.

### Drawing

Each domain yields one 4D block. Pick the flavour you need:

| Method | Range | Quality |
| --- | --- | --- |
| `draw_sample::<N>()` | `[0, 2^32)` | stratified |
| `draw_sample_range::<N>(range)` | `[0, range)` | stratified |
| `draw_sample_f32::<N>()` | `[0, 1)` | stratified |
| `draw_rnd::<N>()` / `_range` / `_f32` | as above | pseudo-random |
| `rng()` → `pcg::Rng` | unbounded stream | pseudo-random |

`N` must be in `1..=4` and is checked at compile time. Use the stratified draws for the
dimensions that matter to convergence, and `draw_rnd` / `rng` for decisions where stratification
buys nothing (Russian roulette, delta tracking).

### Domains

Need more than four dimensions, or a variable number of samples in a sub-integral? Derive more
domains rather than drawing more values:

- `new_domain(key)` — an independent 4D pattern. The standard way to pad dimensions.
- `new_domain_split(key, size, index)` — a fixed sample-rate multiplier: `size` sub-samples per
  parent sample (e.g. 4 shadow rays per camera sample).
- `new_domain_distrib(key, index)` — an adaptive (local) sample rate, when the count varies per
  parent sample.
- `new_domain_chain(key, index)` — an adaptive rate with a global distribution.

Sample indices up to `2^16` are stratified within a pattern; larger indices are handled by
folding the high bits into the domain key, so `index` may exceed `2^16` without extra care.

### Caches

`PmjSampler`, `PmjBnSampler` and the blue-noise samplers need pre-computed tables. Upstream takes
a caller-allocated `void*`; this port builds them in lazily-initialised process globals instead
(the `void*` is a GPU memory-management concern that does not apply here). Tables initialise on
first draw; call `Sampler::<T>::warm_cache()` — e.g. `PmjSampler::warm_cache()` — to pay that cost
up front instead of on a render thread.

## Module map

Modules map one-to-one onto the upstream `oqmc/*.h` headers:

| Module | Upstream header | Contents |
| --- | --- | --- |
| `sampler`, `state` | `sampler.h`, `state.h` | `Sampler`, `SamplerImpl`, the 8-byte `State64Bit` and its domain-tree mutations |
| `pcg` | `pcg.h` | PCG-RXS-M-XS-32, as both a sequential `Rng` and a stateless `hash` |
| `owen`, `rank1`, `stochastic`, `lookup` | `owen.h`, `rank1.h`, `stochastic.h`, `lookup.h` | the sequence cores |
| `bntables` | `bntables.h` | blue-noise key/rank tables |
| `encode`, `float`, `permute`, `range`, `reverse`, `rotate` | leaf headers | bit/encoding utilities |
| `sobol`, `lattice`, `pmj`, `*bn` | matching headers | the six sampler implementations |

The two deliberate divergences from the C++ (the lazy global caches, and surfacing the sequential
PRNG as a named `pcg::Rng` type) are described in `src/sampler.rs` and `src/pcg.rs`.

## Testing

```bash
cargo test
```

This runs the unit tests mirroring upstream's `src/tests/*.cpp` structural invariants, plus the
golden-vector suite.

### Golden vectors

`tests/golden_upstream.rs` is generated by `tools/ref_gen.cpp`, which compiles against the
upstream C++ headers and emits the expected values as Rust tests. It covers `pcg`, the `owen` and
`rank1` cores, `State64Bit`, the PMJ table, and full 4D draws for all six samplers across a grid
of `(x, y, frame, index, domain)`. See [`tools/README.md`](tools/README.md) for how to regenerate
it.

The bundled blue-noise blobs under `src/data/` are the upstream optimised tables
(`include/oqmc/data/**`) converted verbatim to little-endian binary — 65 536 `u32` entries each.

## References

- Burley, *Practical Hash-based Owen Scrambling* (2020); Ahmed (2024)
- Hickernell et al., *Weighted Compound Integration Rules with Higher Order Convergence for all N*
- Helmer et al., *Stochastic Generation of (t,s) Sample Sequences*
- Jarzynski & Olano, *Hash Functions for GPU Rendering*
- Belcour & Heitz, blue-noise sample-set optimisation
- Sobol direction matrices originate from MIT-licensed code by Leonhard Gruenschloss

## License

Apache-2.0, matching upstream. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the full
attribution of the ported algorithms and bundled data.
