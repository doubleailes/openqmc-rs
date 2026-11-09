// Reference benchmark: times the upstream C++ samplers with the same workload as
// upstream's own `src/tools/lib/benchmark.cpp` (and as `../benches/upstream.rs`),
// so the two implementations can be compared number-for-number.
//
// Workload per sampler: for i in [0, 2^15): Sampler(0, 0, 0, i), then 64 times
// chain newDomain(0) and drawSample<4> floats (256 dims). The cache is allocated
// and initialised outside the timed draw loop; init is timed separately.
//
// Output: one CSV line per sampler, `name,init_us,samples_us,ns_per_draw`
// (median over REPS runs; a "draw" is one 4D drawSample call).
#include <oqmc/lattice.h>
#include <oqmc/latticebn.h>
#include <oqmc/pmj.h>
#include <oqmc/pmjbn.h>
#include <oqmc/sobol.h>
#include <oqmc/sobolbn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kSamples = 1 << 15;
constexpr int kDims = 256;

template <typename Sampler>
void loop(int nsamples, int ndims, const void* cache) {
    for (int i = 0; i < nsamples; ++i) {
        auto domain = Sampler(0, 0, 0, i, cache);
        for (int j = 0; j < ndims; j += 4) {
            domain = domain.newDomain(0);
            float sample[4];
            domain.template drawSample<4>(sample);
            volatile float save[4];
            save[0] = sample[0];
            save[1] = sample[1];
            save[2] = sample[2];
            save[3] = sample[3];
            (void)save;
        }
    }
}

template <typename F>
double time_us(F f) {
    const auto start = std::chrono::steady_clock::now();
    f();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename Sampler>
void run(const char* name, int reps) {
    std::vector<double> init, samples;
    for (int r = 0; r < reps; ++r) {
        void* cache = std::malloc(Sampler::cacheSize > 0 ? Sampler::cacheSize : 1);
        init.push_back(time_us([cache] { Sampler::initialiseCache(cache); }));
        loop<Sampler>(1024, kDims, cache); // warm-up
        samples.push_back(time_us([cache] { loop<Sampler>(kSamples, kDims, cache); }));
        std::free(cache);
    }
    const double s = median(samples);
    const double draws = double(kSamples) * (kDims / 4);
    std::printf("%s,%.1f,%.1f,%.3f\n", name, median(init), s, s * 1000.0 / draws);
    std::fflush(stdout);
}

} // namespace

// USAGE: bench_ref [sampler|all] [reps]
int main(int argc, char* argv[]) {
    const char* which = argc > 1 ? argv[1] : "all";
    const int reps = argc > 2 ? std::atoi(argv[2]) : 9;
    const auto want = [which](const char* n) {
        return !std::strcmp(which, "all") || !std::strcmp(which, n);
    };
    std::printf("sampler,init_us,samples_us,ns_per_draw\n");
    if (want("pmj")) run<oqmc::PmjSampler>("pmj", reps);
    if (want("pmjbn")) run<oqmc::PmjBnSampler>("pmjbn", reps);
    if (want("sobol")) run<oqmc::SobolSampler>("sobol", reps);
    if (want("sobolbn")) run<oqmc::SobolBnSampler>("sobolbn", reps);
    if (want("lattice")) run<oqmc::LatticeSampler>("lattice", reps);
    if (want("latticebn")) run<oqmc::LatticeBnSampler>("latticebn", reps);
    return 0;
}
