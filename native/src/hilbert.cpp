/**
 * 4-Manifold Hilbert Curve Encoding
 * Based on Skilling's algorithm (2004)
 */

#include "../include/hartonomous_native.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int BITS_PER_DIM = 21;  // 21 bits × 2 dims = 42-bit manifold
constexpr uint32_t MAX_21BIT = 0x1FFFFF;  // 2^21 - 1

/**
 * Skilling's AxestoTranspose transformation.
 * Transforms coordinates to Hilbert-curve transposed format.
 *
 * This is THE core algorithm. Direct from Skilling (2004).
 */
inline void axes_to_transpose(uint32_t* X, int n, int b) {
    uint32_t M = 1u << (b - 1);
    uint32_t P, Q, t;
    int i;

    // Inverse undo
    for (Q = M; Q > 1; Q >>= 1) {
        P = Q - 1;
        for (i = 0; i < n; i++) {
            if (X[i] & Q) {
                X[0] ^= P;  // invert
            } else {
                t = (X[0] ^ X[i]) & P;
                X[0] ^= t;
                X[i] ^= t;
            }
        }
    }

    // Gray encode
    for (i = 1; i < n; i++) {
        X[i] ^= X[i-1];
    }

    t = 0;
    for (Q = M; Q > 1; Q >>= 1) {
        if (X[n-1] & Q) {
            t ^= Q - 1;
        }
    }

    for (i = 0; i < n; i++) {
        X[i] ^= t;
    }
}

/**
 * Skilling's TransposetoAxes transformation.
 * Inverse of above.
 */
inline void transpose_to_axes(uint32_t* X, int n, int b) {
    uint32_t M = 1u << (b - 1);
    uint32_t P, Q, t;
    int i;

    // Undo Gray encode
    t = X[n-1] >> 1;
    for (i = n-1; i > 0; i--) {
        X[i] ^= X[i-1];
    }
    X[0] ^= t;

    // Undo invert
    for (Q = 2; Q != M << 1; Q <<= 1) {
        P = Q - 1;
        for (i = n-1; i >= 0; i--) {
            if (X[i] & Q) {
                X[0] ^= P;
            } else {
                t = (X[0] ^ X[i]) & P;
                X[0] ^= t;
                X[i] ^= t;
            }
        }
    }
}

/**
 * Encode 2D coordinates to Hilbert index.
 */
inline uint64_t hilbert_encode_2d(uint32_t x, uint32_t y, int bits) {
    uint32_t coords[2] = {x, y};
    axes_to_transpose(coords, 2, bits);

    // Interleave bits
    uint64_t result = 0;
    for (int i = 0; i < bits; i++) {
        result |= (static_cast<uint64_t>((coords[0] >> i) & 1)) << (2 * i);
        result |= (static_cast<uint64_t>((coords[1] >> i) & 1)) << (2 * i + 1);
    }
    return result;
}

/**
 * Decode Hilbert index to 2D coordinates.
 */
inline void hilbert_decode_2d(uint64_t h, uint32_t* out_x, uint32_t* out_y, int bits) {
    uint32_t coords[2] = {0, 0};

    // De-interleave bits
    for (int i = 0; i < bits; i++) {
        coords[0] |= ((h >> (2 * i)) & 1) << i;
        coords[1] |= ((h >> (2 * i + 1)) & 1) << i;
    }

    transpose_to_axes(coords, 2, bits);
    *out_x = coords[0];
    *out_y = coords[1];
}

/**
 * Quantize double [-∞, ∞] to uint32 [0, 2^21-1]
 */
inline uint32_t quantize(double value) {
    // Apply tanh to bound to [-1, 1]
    double bounded = std::tanh(value);
    // Map to [0, 1]
    double normalized = (bounded + 1.0) * 0.5;
    // Scale to [0, 2^21-1]
    return static_cast<uint32_t>(normalized * MAX_21BIT);
}

/**
 * Dequantize uint32 [0, 2^21-1] to double
 */
inline double dequantize(uint32_t quantized) {
    // Map to [0, 1]
    double normalized = static_cast<double>(quantized) / MAX_21BIT;
    // Map to [-1, 1]
    double bounded = normalized * 2.0 - 1.0;
    // Apply inverse tanh (atanh)
    return std::atanh(std::clamp(bounded, -0.999999, 0.999999));
}

} // anonymous namespace

extern "C" {

void hilbert_encode_4d_batch(
    const double* coords,
    int count,
    int64_t* out_hilbert)
{
    if (!coords || !out_hilbert || count <= 0) {
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; i++) {
        const double* coord = &coords[i * 4];
        int64_t* hilbert = &out_hilbert[i * 4];

        // Quantize to 21-bit integers
        uint32_t qx = quantize(coord[0]);
        uint32_t qy = quantize(coord[1]);
        uint32_t qz = quantize(coord[2]);
        uint32_t qm = quantize(coord[3]);

        // Encode 4 overlapping 2D manifolds
        hilbert[0] = static_cast<int64_t>(hilbert_encode_2d(qx, qy, BITS_PER_DIM));  // h_xy
        hilbert[1] = static_cast<int64_t>(hilbert_encode_2d(qy, qz, BITS_PER_DIM));  // h_yz
        hilbert[2] = static_cast<int64_t>(hilbert_encode_2d(qz, qm, BITS_PER_DIM));  // h_zm
        hilbert[3] = static_cast<int64_t>(hilbert_encode_2d(qm, qy, BITS_PER_DIM));  // h_my (cycle)
    }
}

int hilbert_decode_4d_batch(
    const int64_t* hilbert,
    int count,
    double* out_coords)
{
    if (!hilbert || !out_coords || count <= 0) {
        return -1;
    }

    int error_code = 0;

    #pragma omp parallel for schedule(static) shared(error_code)
    for (int i = 0; i < count; i++) {
        // Skip if error already occurred
        if (error_code != 0) continue;

        const int64_t* h = &hilbert[i * 4];
        double* coord = &out_coords[i * 4];

        uint32_t qx1, qy1, qy2, qy3, qz1, qz2, qm1, qm2;

        // Decode each manifold
        hilbert_decode_2d(h[0], &qx1, &qy1, BITS_PER_DIM);  // XY
        hilbert_decode_2d(h[1], &qy2, &qz1, BITS_PER_DIM);  // YZ
        hilbert_decode_2d(h[2], &qz2, &qm1, BITS_PER_DIM);  // ZM
        hilbert_decode_2d(h[3], &qm2, &qy3, BITS_PER_DIM);  // MY

        // Integrity check: Y appears in 3 manifolds, should be consistent
        uint32_t y_diff1 = (qy1 > qy2) ? (qy1 - qy2) : (qy2 - qy1);
        uint32_t y_diff2 = (qy2 > qy3) ? (qy2 - qy3) : (qy3 - qy2);
        uint32_t z_diff = (qz1 > qz2) ? (qz1 - qz2) : (qz2 - qz1);
        uint32_t m_diff = (qm1 > qm2) ? (qm1 - qm2) : (qm2 - qm1);

        // Allow small quantization error
        constexpr uint32_t TOLERANCE = 2;
        if (y_diff1 > TOLERANCE || y_diff2 > TOLERANCE ||
            z_diff > TOLERANCE || m_diff > TOLERANCE) {
            #pragma omp atomic write
            error_code = -2;  // Integrity check failed
            continue;
        }

        // Average overlapping dimensions
        uint32_t qx = qx1;
        uint32_t qy = (qy1 + qy2 + qy3) / 3;
        uint32_t qz = (qz1 + qz2) / 2;
        uint32_t qm = (qm1 + qm2) / 2;

        // Dequantize
        coord[0] = dequantize(qx);
        coord[1] = dequantize(qy);
        coord[2] = dequantize(qz);
        coord[3] = dequantize(qm);
    }

    return error_code;  // Success or error code
}

void hilbert_pack_64bit(
    const int64_t* hilbert_full,
    int count,
    uint64_t* out_packed)
{
    if (!hilbert_full || !out_packed || count <= 0) {
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; i++) {
        const int64_t* h = &hilbert_full[i * 4];

        // Take top 16 bits of each 42-bit manifold
        uint64_t packed = 0;
        packed |= (static_cast<uint64_t>(h[0] >> 26) & 0xFFFF);
        packed |= (static_cast<uint64_t>(h[1] >> 26) & 0xFFFF) << 16;
        packed |= (static_cast<uint64_t>(h[2] >> 26) & 0xFFFF) << 32;
        packed |= (static_cast<uint64_t>(h[3] >> 26) & 0xFFFF) << 48;

        out_packed[i] = packed;
    }
}

} // extern "C"
