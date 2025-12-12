/**
 * Utility functions: entropy, compressibility, etc.
 */

#include "../include/hartonomous_native.h"
#include <zlib.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

extern "C" {

void compute_entropy_batch(
    const uint8_t** data,
    const int* lengths,
    int count,
    double* out_entropy)
{
    if (!data || !lengths || !out_entropy || count <= 0) {
        return;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < count; idx++) {
        const uint8_t* bytes = data[idx];
        int len = lengths[idx];

        if (len <= 0) {
            out_entropy[idx] = 0.0;
            continue;
        }

        // Count byte frequencies
        int freq[256] = {0};
        for (int i = 0; i < len; i++) {
            freq[bytes[i]]++;
        }

        // Compute Shannon entropy: H = -Σ p(x) log₂ p(x)
        double entropy = 0.0;
        for (int i = 0; i < 256; i++) {
            if (freq[i] > 0) {
                double p = static_cast<double>(freq[i]) / len;
                entropy -= p * std::log2(p);
            }
        }

        out_entropy[idx] = entropy;
    }
}

void compute_compressibility_batch(
    const uint8_t** data,
    const int* lengths,
    int count,
    double* out_ratio)
{
    if (!data || !lengths || !out_ratio || count <= 0) {
        return;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < count; idx++) {
        const uint8_t* bytes = data[idx];
        int len = lengths[idx];

        if (len <= 0) {
            out_ratio[idx] = 1.0;
            continue;
        }

        // Compress with gzip
        uLongf compressed_size = compressBound(len);
        std::vector<uint8_t> compressed(compressed_size);

        int result = compress(
            compressed.data(),
            &compressed_size,
            bytes,
            len
        );

        if (result != Z_OK) {
            out_ratio[idx] = 1.0;  // Incompressible
        } else {
            out_ratio[idx] = static_cast<double>(compressed_size) / len;
        }
    }
}

void create_float_atoms_batch(
    const float* values,
    int count,
    int64_t* out_hilbert,
    double* out_coords)
{
    if (!values || !out_hilbert || !out_coords || count <= 0) {
        return;
    }

    // For each float, compute coordinates:
    // X = value itself
    // Y = entropy of binary representation
    // Z = compressibility
    // M = 0 (will be updated with ref_count later)

    std::vector<const uint8_t*> data_ptrs(count);
    std::vector<int> lengths(count, 4);  // float32 = 4 bytes
    std::vector<double> entropies(count);
    std::vector<double> compressibilities(count);

    // Prepare pointers
    for (int i = 0; i < count; i++) {
        data_ptrs[i] = reinterpret_cast<const uint8_t*>(&values[i]);
    }

    // Compute entropy and compressibility
    compute_entropy_batch(data_ptrs.data(), lengths.data(), count, entropies.data());
    compute_compressibility_batch(data_ptrs.data(), lengths.data(), count, compressibilities.data());

    // Build coordinates
    std::vector<double> coords(count * 4);
    for (int i = 0; i < count; i++) {
        coords[i * 4 + 0] = values[i];  // X
        coords[i * 4 + 1] = entropies[i];  // Y
        coords[i * 4 + 2] = compressibilities[i];  // Z
        coords[i * 4 + 3] = 0.0;  // M
    }

    // Copy coords to output
    std::memcpy(out_coords, coords.data(), coords.size() * sizeof(double));

    // Encode to Hilbert (this is defined in hilbert.cpp)
    extern void hilbert_encode_4d_batch(const double*, int, int64_t*);
    hilbert_encode_4d_batch(coords.data(), count, out_hilbert);
}

} // extern "C"
