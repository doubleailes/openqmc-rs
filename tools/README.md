# Reference generator

`ref_gen.cpp` regenerates the golden vectors in `../tests/golden_upstream.rs`
directly from the upstream C++ headers, so the Rust port can be checked
bit-for-bit against the original.

## Regenerating

Clone the upstream headers and compile against them (scalar path, so the values
match the portable Rust implementation):

```bash
git clone --depth 1 https://github.com/AcademySoftwareFoundation/openqmc.git
g++ -std=c++17 -O2 -DOQMC_FORCE_SCALAR -I openqmc/include \
    ref_gen.cpp -o ref_gen
./ref_gen > ../tests/golden_upstream.rs
cargo fmt   # the generated file must be reformatted to keep CI's fmt check green
```

Then `cargo test -p openqmc-rs --test golden_upstream` must pass. The generator
emits Rust `#[test]`s whose expected values are the upstream outputs; the test
bodies recompute the same quantities through this crate's public API and assert
equality. It covers `pcg`, `owen`/`rank1` cores, `State64Bit`, the PMJ table,
and full draws for all six samplers (Sobol/Lattice/PMJ + blue-noise variants)
across a grid of `(x, y, frame, index, domain)`.

The blue-noise `.bin` blobs under `../src/data/` are likewise a verbatim
little-endian conversion of `openqmc/include/oqmc/data/**/*.txt` (65 536 `u32`
entries each).

## Benchmark against the C++ reference

`bench_ref.cpp` and `../benches/upstream.rs` time the same workload as
upstream's own `src/tools/lib/benchmark.cpp`. For each sampler, they run 2^15
sample indices. Each index builds `Sampler(0, 0, 0, i)`, then 64 times calls
`newDomain(0)` and draws 4 floats, for 256 dimensions. Each prints CSV:
`sampler,init_us,samples_us,ns_per_draw`. The draw time is the median of
`reps` runs, and a "draw" is one 4D `drawSample` call.

```bash
g++ -std=c++17 -O3 -I openqmc/include bench_ref.cpp -o bench_ref
./bench_ref all 9            # or a single sampler: ./bench_ref sobol 9
cargo bench --bench upstream -- all 9
```

Add `-DOQMC_FORCE_SCALAR` to time upstream's scalar path. Add `-march=native`,
or `RUSTFLAGS="-C target-cpu=native"` for Rust, to time AVX2 builds.

**Pin upstream to the commit the port tracks.** Upstream `b6a204e`
(2026-09-04, "Replace Sobol matrices with SZ sequences") changed Sobol and
Sobol-BN output, and this port predates it. Benchmark against its parent so
both sides compute identical values:

```bash
git -C openqmc fetch --unshallow && git -C openqmc checkout b6a204e~1
```

At that commit, a 64-bit hash of every value the benchmark draws (4096 indices ×
64 domains) is identical between Rust and C++ for all six samplers. This holds
for the scalar, SSE2 and AVX2 builds.

### Results (2026-11-09)

Run on a shared 4-vCPU Intel Xeon @ 2.10 GHz (AVX2) cloud VM, with rustc 1.97,
g++ 13.3 and clang 18.1, all at `-O3` / release. Each number is ns per 4D draw:
the best of 3 rounds, each round the median of 9 runs. Run-to-run noise on this
host is about ±20%, so treat differences under about 1.2× as parity. "C++" is
the faster of g++ and clang.

| sampler   | Rust | C++ scalar | C++ SSE2 | Rust native | C++ AVX2 |
| --------- | ---: | ---------: | -------: | ----------: | -------: |
| pmj       |  5.8 |       10.0 |      8.2 |     **3.5** |      7.3 |
| pmjbn     | 12.4 |       10.3 |     12.5 |     **6.0** |     11.3 |
| sobol     | 18.8 |       27.5 |     19.5 |     **7.5** |     18.5 |
| sobolbn   | 22.6 |       31.2 |     26.0 |    **11.0** |     25.6 |
| lattice   |  6.8 |        7.6 |      6.8 |         7.4 |  **5.8** |
| latticebn | 12.4 |       10.6 |     12.3 |        11.4 | **10.6** |

- With default flags (x86-64 baseline, SSE2), the port is at parity with
  upstream's SIMD build. It is faster on `pmj` (about 1.4×), and roughly level
  on everything else.
- With `target-cpu=native`, Rust is 2–2.5× faster than upstream's AVX2 path on
  the PMJ and Sobol families. The lattice samplers stay at parity, with C++
  slightly ahead.
- Cache initialisation is one-time per process in Rust (lazy globals). Building
  the PMJ table costs about the same in both, roughly 0.3–1.3 ms. Loading the
  blue-noise tables takes about 0.3 ms in Rust, against 0.02–0.07 ms in C++,
  because Rust decodes the bundled little-endian blobs into a `Box<[u32]>`.
