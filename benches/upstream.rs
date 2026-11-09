//! Rust half of the C++ comparison benchmark (`tools/bench_ref.cpp`).
//!
//! Same workload as upstream's `src/tools/lib/benchmark.cpp`: for each of 2^15
//! sample indices, build `Sampler::new(0, 0, 0, i)`, then 64 times chain
//! `new_domain(0)` and draw 4 floats (256 dimensions). Prints one CSV line per
//! sampler, `name,init_us,samples_us,ns_per_draw`, with the draw time the
//! median over `reps` runs.
//!
//! Caches are process-global and lazily built here, so `init_us` is the single
//! first-touch `warm_cache()` cost rather than a median.
//!
//! ```text
//! cargo bench --bench upstream -- [sampler|all] [reps]
//! ```

use std::hint::black_box;
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
    let reps = args.get(1).and_then(|r| r.parse().ok()).unwrap_or(9);
    let want = |n: &str| which == "all" || which == n;

    println!("sampler,init_us,samples_us,ns_per_draw");
    if want("pmj") {
        run::<PmjImpl>("pmj", reps);
    }
    if want("pmjbn") {
        run::<PmjBnImpl>("pmjbn", reps);
    }
    if want("sobol") {
        run::<SobolImpl>("sobol", reps);
    }
    if want("sobolbn") {
        run::<SobolBnImpl>("sobolbn", reps);
    }
    if want("lattice") {
        run::<LatticeImpl>("lattice", reps);
    }
    if want("latticebn") {
        run::<LatticeBnImpl>("latticebn", reps);
    }
}
