// Copyright Contributors to the Cellarium project.
// SPDX-License-Identifier: BSD-3-Clause

// Times BPCells' own BP-128 decoders on one matrix stored in the BPCells directory format, bypassing its Python bindings.
//
//   bpcells_decode_harness MATRIX_DIR EXPECTED_DIR [REPEATS] [RANGES]
//
// MATRIX_DIR   a BPCells matrix directory written cell-major (e.g. by bpcells.experimental.DirMatrix.from_scipy_sparse(csr))
// EXPECTED_DIR contains expected_val.bin and expected_idx.bin (raw little-endian uint32: the CSR data and indices)
//
// It reads the compressed streams (val_data, val_idx, index_data, index_idx, index_starts, the *_idx_offsets and idxptr) into
// memory, decodes the value stream (BP-128 frame-of-reference) and the index stream (BP-128 delta) into flat uint32 arrays with
// BPCells' BP128_FOR_UIntReader / BP128_D1Z_UIntReader, checks them against the expected arrays, and prints one line:
//
//   RESULT nnz=... compressed_bytes=... read_ms=... decode_ms_1=... decode_ms_N=... threads_N=... correct=1
//
// decode_ms_1: both streams, one thread. decode_ms_N: each stream split into RANGES 128-aligned ranges decoded by separate threads
// (2 * RANGES threads in total; this parallelization is ours, BPCells' readers are single-threaded). Output arrays are preallocated
// and reused, so page-fault costs are excluded. BPCells itself is not modified; see build.sh.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "arrayIO/bp128.h"
#include "arrayIO/vector.h"

using namespace BPCells;

static const uint64_t LOAD_SIZE = 4096;  // elements per load() call and per reader buffer

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// BPCells files start with an 8-byte type header ("UINT32v1" / "UINT64v1") followed by the raw array
template <class T> static std::vector<T> read_bpcells_array(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    size_t n = (size_t(f.tellg()) - 8) / sizeof(T);
    std::vector<T> v(n);
    f.seekg(8);
    f.read(reinterpret_cast<char *>(v.data()), n * sizeof(T));
    return v;
}

static std::vector<uint32_t> read_raw_u32(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    std::vector<uint32_t> v(size_t(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), v.size() * 4);
    return v;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MATRIX_DIR EXPECTED_DIR [REPEATS] [RANGES]\n", argv[0]); return 2; }
    const std::string dir = argv[1], expdir = argv[2];
    const int reps = argc > 3 ? atoi(argv[3]) : 15;
    const int ranges = argc > 4 ? atoi(argv[4]) : 4;

    // read the compressed streams (timed: this is the "file -> memory" part of a read)
    VecReaderWriterBuilder vb(LOAD_SIZE);
    std::vector<double> t_read;
    uint64_t compressed_bytes = 0, count = 0;
    for (int r = 0; r < std::max(3, reps / 3); r++) {
        double t0 = now_s();
        compressed_bytes = 0;
        for (auto name : {"val_data", "val_idx", "index_data", "index_idx", "index_starts"}) {
            vb.getIntVecs()[name] = read_bpcells_array<uint32_t>(dir + "/" + name);
            compressed_bytes += vb.getIntVecs()[name].size() * 4;
        }
        for (auto name : {"val_idx_offsets", "index_idx_offsets"}) {
            vb.getLongVecs()[name] = read_bpcells_array<uint64_t>(dir + "/" + name);
            compressed_bytes += vb.getLongVecs()[name].size() * 8;
        }
        auto idxptr = read_bpcells_array<uint64_t>(dir + "/idxptr");
        compressed_bytes += idxptr.size() * 8;
        count = idxptr.back();
        t_read.push_back(now_s() - t0);
    }

    auto make_val = [&]() {
        return std::make_unique<BP128_FOR_UIntReader>(vb.openUIntReader("val_data"), vb.openUIntReader("val_idx"), vb.openULongReader("val_idx_offsets"), count);
    };
    auto make_idx = [&]() {
        return std::make_unique<BP128_D1Z_UIntReader>(vb.openUIntReader("index_data"), vb.openUIntReader("index_idx"), vb.openULongReader("index_idx_offsets"),
                                                      vb.openUIntReader("index_starts"), count);
    };
    std::vector<uint32_t> val_out(count), idx_out(count);
    auto decode_range = [&](BP128UIntReader &r, uint32_t *out, uint64_t lo, uint64_t hi) {  // lo must be a multiple of 128
        r.seek(lo);
        uint64_t pos = lo;
        while (pos < hi) pos += r.load(out + pos, std::min<uint64_t>(hi - pos, LOAD_SIZE));
    };

    // correctness against the expected CSR arrays
    {
        auto v = make_val();
        auto i = make_idx();
        decode_range(*v, val_out.data(), 0, count);
        decode_range(*i, idx_out.data(), 0, count);
    }
    auto ev = read_raw_u32(expdir + "/expected_val.bin"), ei = read_raw_u32(expdir + "/expected_idx.bin");
    const bool correct = ev.size() == count && ei.size() == count && memcmp(ev.data(), val_out.data(), count * 4) == 0 && memcmp(ei.data(), idx_out.data(), count * 4) == 0;

    std::vector<double> t1, tn;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        {
            auto v = make_val();
            decode_range(*v, val_out.data(), 0, count);
            auto i = make_idx();
            decode_range(*i, idx_out.data(), 0, count);
        }
        t1.push_back(now_s() - t0);
        t0 = now_s();
        {
            std::vector<std::thread> th;
            uint64_t chunk = ((count / ranges + 127) / 128) * 128;
            for (int s = 0; s < ranges; s++) {
                uint64_t lo = s * chunk, hi = std::min<uint64_t>(count, lo + chunk);
                if (lo >= hi) continue;
                th.emplace_back([&, lo, hi] { auto v = make_val(); decode_range(*v, val_out.data(), lo, hi); });
                th.emplace_back([&, lo, hi] { auto i = make_idx(); decode_range(*i, idx_out.data(), lo, hi); });
            }
            for (auto &t : th) t.join();
        }
        tn.push_back(now_s() - t0);
    }
    printf("RESULT nnz=%llu compressed_bytes=%llu read_ms=%.3f decode_ms_1=%.3f decode_ms_N=%.3f threads_N=%d correct=%d\n", (unsigned long long)count,
           (unsigned long long)compressed_bytes, median(t_read) * 1e3, median(t1) * 1e3, median(tn) * 1e3, 2 * ranges, correct ? 1 : 0);
    return correct ? 0 : 1;
}
