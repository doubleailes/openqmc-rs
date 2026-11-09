//! Rust half of the C++ comparison benchmark (`tools/bench_ref.cpp`).
//!
//! Same workload as upstream's `src/tools/lib/benchmark.cpp`: for each of 2^15
//! sample indices, build `Sampler::new(0, 0, 0, i)`, then 64 times chain
//! `new_domain(0)` and draw 4 floats (256 dimensions). Prints one CSV line per
//! sampler, `name,init_us,samples_us,ns_per_draw`, with the draw time the
//! median over `reps` runs.
//!
//! `init_us` is a cold start: the first `warm_cache()` in a process that runs
//! only that sampler. Caches are process-global and shared (PMJ-BN reuses the
//! PMJ table), so `all` re-runs this binary once per sampler instead of timing
//! them in one process. `tools/bench_ref.cpp` measures init the same way.
//!
//! ```text
//! cargo bench --bench upstream -- [sampler|all] [reps]
//! ```

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use openqmc::lattice::LatticeImpl;
use openqmc::latticebn::LatticeBnImpl;
use openqmc::pmj::PmjImpl;
use openqmc::pmjbn::PmjBnImpl;
use openqmc::sobol::SobolImpl;
use openqmc::sobolbn::SobolBnImpl;
use openqmc::{Sampler, SamplerImpl};

const SAMPLES: i32 = 1 << 15;
const DIMS: i32 = 256;
const SAMPLERS: [&str; 6] = ["pmj", "pmjbn", "sobol", "sobolbn", "lattice", "latticebn"];

#[inline(never)]
fn run_loop<T: SamplerImpl>(nsamples: i32, ndims: i32) {
    for i in 0..nsamples {
        let mut domain = Sampler::<T>::new(0, 0, 0, i);
        for _ in (0..ndims).step_by(4) {
            domain = domain.new_domain(0);
            black_box(domain.draw_sample_f32::<4>());
        }
    }
}

fn time_us(f: impl FnOnce()) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64() * 1e6
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn run<T: SamplerImpl>(name: &str, reps: usize) {
    let init = time_us(Sampler::<T>::warm_cache);
    run_loop::<T>(1024, DIMS); // warm-up
    let samples = median(
        (0..reps)
            .map(|_| time_us(|| run_loop::<T>(black_box(SAMPLES), black_box(DIMS))))
            .collect(),
    );
    let draws = f64::from(SAMPLES) * f64::from(DIMS / 4);
    println!(
        "{name},{init:.1},{samples:.1},{:.3}",
        samples * 1000.0 / draws
    );
}

fn main() {
    // `cargo bench` appends `--bench`; ignore flags.
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .collect();
    let which = args.first().map_or("all", String::as_str);
    let reps: usize = args.get(1).and_then(|r| r.parse().ok()).unwrap_or(9);
    // Internal: set by the `all` driver so each child skips the header.
    let child = std::env::args().any(|a| a == "--child");

    if !child {
        println!("sampler,init_us,samples_us,ns_per_draw");
    }
    match which {
        "all" => {
            // One fresh process per sampler, so each init is a cold start.
            let exe = std::env::current_exe().expect("current_exe");
            for name in SAMPLERS {
                let status = Command::new(&exe)
                    .args([name, &reps.to_string(), "--child"])
                    .status()
                    .expect("spawn benchmark child");
                assert!(status.success(), "{name} benchmark failed");
            }
        }
        "pmj" => run::<PmjImpl>("pmj", reps),
        "pmjbn" => run::<PmjBnImpl>("pmjbn", reps),
        "sobol" => run::<SobolImpl>("sobol", reps),
        "sobolbn" => run::<SobolBnImpl>("sobolbn", reps),
        "lattice" => run::<LatticeImpl>("lattice", reps),
        "latticebn" => run::<LatticeBnImpl>("latticebn", reps),
        other => panic!("unknown sampler {other:?}; expected one of {SAMPLERS:?} or all"),
    }
}
