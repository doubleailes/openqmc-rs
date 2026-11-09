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

The checked-in vectors were generated from upstream `d6c40b5` (2026-11-09),
which includes `b6a204e` ("Replace Sobol matrices with SZ sequences").

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
`reps` runs, and a "draw" is one 4D `drawSample` call. `init_us` is a cold
start on both sides: the first cache initialisation in a process that runs only
that sampler. `all` re-runs the binary once per sampler for this reason. Rust's
caches are process-global and shared (PMJ-BN reuses the PMJ table), so timing
them in one process would leave some of the cost out.

```bash
g++ -std=c++17 -O3 -I openqmc/include bench_ref.cpp -o bench_ref
./bench_ref all 9            # or a single sampler: ./bench_ref sobol 9
cargo bench --bench upstream -- all 9
```

Add `-DOQMC_FORCE_SCALAR` to time upstream's scalar path. Add `-march=native`,
or `RUSTFLAGS="-C target-cpu=native"` for Rust, to build for the host CPU.

**Benchmark against upstream HEAD.** Upstream `b6a204e` (2026-09-04, "Replace
Sobol matrices with SZ sequences") changed Sobol and Sobol-BN output, and this
port tracks it. Earlier releases of this crate (up to 0.2.4) match `b6a204e~1`
instead. To time one of those, check that commit out
(`git -C openqmc fetch --unshallow && git -C openqmc checkout b6a204e~1`).

At upstream `d6c40b5` (HEAD on 2026-11-09), a 64-bit hash of every value the
benchmark draws (4096 indices × 64 domains) is identical between Rust and C++
for all six samplers. This holds for the scalar, SSE2 and `-march=native` C++ builds, and
for the baseline and `target-cpu=native` Rust builds.

### Results (2026-11-09, upstream `d6c40b5`)

Run on a shared 4-vCPU Intel Xeon cloud VM (Granite Rapids: AVX-512 and GFNI),
with rustc 1.97, g++ 13.3 and clang 18.1, all at `-O3` / release. Each number is ns per 4D draw:
the best of 3 rounds, each round the median of 9 runs. Run-to-run noise on this
host is about ±20%, so treat differences under about 1.2× as parity. "C++" is
the faster of g++ and clang.

| sampler   | Rust | C++ scalar | C++ SSE2 | Rust native | C++ native |
| --------- | ---: | ---------: | -------: | ----------: | -------: |
| pmj       |  7.0 |       10.6 |     10.2 |     **3.9** |      9.2 |
| pmjbn     | 12.3 |       15.6 |     15.2 |     **6.4** |     13.6 |
| sobol     | 18.9 |       26.9 |     26.7 |     **9.3** |     26.7 |
| sobolbn   | 22.5 |       35.6 |     35.4 |    **11.4** |     34.7 |
| lattice   |  6.8 |        8.7 |      8.7 |     7.4     | **7.4**  |
| latticebn | 13.2 |       13.6 |     13.5 |    **11.5** |     12.3 |

- Upstream `b6a204e` removed the SSE, AVX and Neon paths for the Sobol core, so
  upstream's Sobol and Sobol-BN cost the same in every C++ build (about 27 and
  35 ns). Before it, the AVX2 build ran Sobol at about 18.5 ns.
- This port evaluates the SZ matrix product from byte tables, as it did for
  the old matrices. With default flags it is 1.4–1.6× faster than upstream on
  the Sobol family. With `target-cpu=native` it is about 3× faster (see
  [below](#where-the-target-cpunative-gain-comes-from)). A port of
  upstream's new host path (reverse once, shift-mask-xor programs, dimension 3
  chained from 2) measured 23 / 29 ns at baseline and about 18 ns native for
  `sobol`, so the tables stay.
- PMJ is 1.3–1.5× faster than upstream at baseline and 2–2.5× faster native.
  The lattice samplers are at parity.
- Cold-start cache initialisation is at parity. Building the PMJ table takes
  about 1.3–2.1 ms in both, and PMJ-BN, which also needs the blue-noise tables,
  about 1.5–2.0 ms. Loading the blue-noise tables alone takes about 0.3 ms in
  both. An earlier version of this table reported 0.02–0.08 ms for C++. That
  figure was the median of repeated initialisations within one process, so it
  was not a cold start.

### Where the `target-cpu=native` gain comes from

A default Rust build targets baseline x86-64, which has only SSE2. The gain
from `native` is almost entirely in **bit reversal**. The Owen scrambling used
by the Sobol and PMJ samplers calls `reverse_bits32` / `reverse_bits16`
(`src/reverse.rs`), which compile to LLVM's `bitreverse` intrinsic. LLVM lowers
that intrinsic according to the target features:

| target features   | `bitreverse` lowering                          |
| ----------------- | ---------------------------------------------- |
| SSE2 (baseline)   | scalar shift-and-mask ladder, about 15 ops     |
| SSSE3             | `pshufb` nibble lookup                         |
| GFNI              | one `gf2p8affineqb`, plus `bswap`              |

Upstream hand-rolls the reversal with masks, so g++ and clang emit the scalar
ladder at any `-march`. That is why the C++ columns barely move.

The table below isolates each feature with Rust builds of this crate, on the
same host and upstream commit as the results above. Each number is ns per draw,
the best of 3 rounds:

| sampler   | x86-64 | +SSSE3/4.1 | +GFNI | v3   | v3+GFNI | v4   | v4+GFNI | native | native−GFNI |
| --------- | -----: | ---------: | ----: | ---: | ------: | ---: | ------: | -----: | ----------: |
| pmj       |    5.8 |        5.1 |   4.7 |  5.0 |     4.1 |  4.6 |     3.4 |    3.2 |         4.2 |
| sobol     |   15.9 |       11.9 |  13.3 | 11.5 |     9.2 | 11.0 |     8.2 |    7.9 |        10.2 |
| sobolbn   |   19.1 |       13.6 |  14.7 | 13.5 |    10.8 | 12.6 |     9.6 |    9.7 |        12.6 |
| lattice   |    6.0 |        6.3 |   5.8 |  6.3 |     6.1 |  6.3 |     6.3 |    6.3 |         6.2 |

- `x86-64-v4` plus GFNI matches `native`, and `native` minus GFNI falls back to
  v4 speed. The rest of the gain is v4's 512-bit vectors, worth about 10%.
- Lattice samplers don't reverse bits, so no feature level helps them.
- A `native` binary only runs on CPUs with the build host's features. For a
  portable speedup, `-C target-cpu=x86-64-v2` (SSSE3, any x86-64 CPU since about
  2009) gets most of the Sobol gain. Adding `-C target-feature=+gfni` gets
  near-native speed on Ice Lake / Zen 4 and newer.
