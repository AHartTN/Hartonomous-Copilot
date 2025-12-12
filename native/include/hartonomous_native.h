/**
 * Hartonomous Native Library
 * Production-grade C++ core for performance-critical operations
 *
 * Shared across:
 * - Python (via ctypes/CFFI)
 * - PostgreSQL (via C extension)
 * - C# (via P/Invoke)
 */

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef _WIN32
    #ifdef HARTONOMOUS_EXPORTS
        #define EXPORT __declspec(dllexport)
    #else
        #define EXPORT __declspec(dllimport)
    #endif
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// ====================
// HILBERT ENCODING
// ====================

/**
 * Encode 4D coordinates to 4-manifold Hilbert representation.
 *
 * @param coords Array of [x, y, z, m] coordinates
 * @param count Number of 4D points
 * @param out_hilbert Output array [h_xy, h_yz, h_zm, h_my, ...] (4*count int64s)
 *
 * Parallelized with OpenMP. Zero RBAR.
 */
EXPORT void hilbert_encode_4d_batch(
    const double* coords,
    int count,
    int64_t* out_hilbert
);

/**
 * Decode 4-manifold Hilbert back to 4D coordinates.
 * Includes integrity checking (Y appears in h_xy, h_yz, h_my; etc.)
 *
 * @return 0 on success, error code on integrity check failure
 */
EXPORT int hilbert_decode_4d_batch(
    const int64_t* hilbert,
    int count,
    double* out_coords
);

/**
 * Pack 4-manifold (168 bits) to 64-bit for spatial indexing.
 * Takes top 16 bits of each manifold (lossy but sufficient for k-NN).
 */
EXPORT void hilbert_pack_64bit(
    const int64_t* hilbert_full,  // (count * 4) int64s
    int count,
    uint64_t* out_packed          // count uint64s
);

// ====================
// DIMENSIONALITY REDUCTION
// ====================

/**
 * PCA projection for embedding dimensionality reduction.
 *
 * @param embeddings Input matrix (n × d_in), row-major
 * @param n Number of samples
 * @param d_in Input dimensions
 * @param d_out Output dimensions (typically 4)
 * @param out_coords Output matrix (n × d_out)
 *
 * Uses Eigen for fast eigendecomposition.
 */
EXPORT void pca_project(
    const float* embeddings,
    int n,
    int d_in,
    int d_out,
    float* out_coords
);

/**
 * Laplacian Eigenmap with Neumann extension (Feb 2025 algorithm).
 * Preserves manifold topology better than PCA.
 *
 * @param n_landmarks Number of landmark points for Nyström extension
 * @param k_neighbors K for k-NN graph construction
 */
EXPORT void laplacian_eigenmap(
    const float* embeddings,
    int n,
    int d_in,
    int d_out,
    int n_landmarks,
    int k_neighbors,
    float* out_coords
);

// ====================
// COORDINATE COMPUTATION
// ====================

/**
 * Compute Shannon entropy of byte sequences.
 * Used for Y-coordinate of primitive atoms.
 */
EXPORT void compute_entropy_batch(
    const uint8_t** data,     // Array of pointers to byte sequences
    const int* lengths,
    int count,
    double* out_entropy
);

/**
 * Compute compressibility (gzip ratio).
 * Used for Z-coordinate of primitive atoms.
 */
EXPORT void compute_compressibility_batch(
    const uint8_t** data,
    const int* lengths,
    int count,
    double* out_ratio
);

// ====================
// BATCH UTILITIES
// ====================

/**
 * Bulk float atom creation.
 * Computes coordinates (value, entropy, compressibility, 0) and Hilbert IDs.
 *
 * @param values Array of float32 values
 * @param count Number of floats
 * @param out_hilbert Output Hilbert IDs (count * 4)
 * @param out_coords Output coordinates (count * 4)
 */
EXPORT void create_float_atoms_batch(
    const float* values,
    int count,
    int64_t* out_hilbert,
    double* out_coords
);

// ====================
// STATISTICS (Meta-Learning)
// ====================

/**
 * Update running statistics using Welford's algorithm.
 * Numerically stable variance computation.
 *
 * @param old_mean Previous mean
 * @param old_m2 Previous M2 (sum of squared deviations)
 * @param old_count Previous sample count
 * @param new_value New observation
 * @param out_mean Updated mean
 * @param out_m2 Updated M2
 */
EXPORT void welford_update(
    double old_mean,
    double old_m2,
    int old_count,
    double new_value,
    double* out_mean,
    double* out_m2
);

// ====================
// VERSION INFO
// ====================

EXPORT const char* hartonomous_version();
EXPORT int hartonomous_build_number();

} // extern "C"
