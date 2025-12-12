/**
 * Manifold Learning: PCA and Laplacian Eigenmaps
 * The embedding vectors ARE the manifold - we just project to 4D
 */

#include "../include/hartonomous_native.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <algorithm>
#include <random>

using namespace Eigen;

extern "C" {

void pca_project(
    const float* embeddings,
    int n,
    int d_in,
    int d_out,
    float* out_coords)
{
    if (!embeddings || !out_coords || n <= 0 || d_in <= 0 || d_out <= 0) {
        return;
    }

    // Map input data (n × d_in), row-major
    Map<const Matrix<float, Dynamic, Dynamic, RowMajor>> data(embeddings, n, d_in);

    // Center data
    VectorXf mean = data.colwise().mean();
    MatrixXf centered = data.rowwise() - mean.transpose();

    // Compute covariance matrix (d_in × d_in)
    // For large d_in, this is expensive but Eigen is optimized
    MatrixXf cov = (centered.adjoint() * centered) / static_cast<float>(n - 1);

    // Eigendecomposition
    SelfAdjointEigenSolver<MatrixXf> eig(cov);

    // Take top d_out eigenvectors (largest eigenvalues)
    MatrixXf components = eig.eigenvectors().rightCols(d_out);

    // Project data to lower dimension
    MatrixXf projected = centered * components;

    // Copy to output (row-major)
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(out_coords, n, d_out) = projected;
}

void laplacian_eigenmap(
    const float* embeddings,
    int n,
    int d_in,
    int d_out,
    int n_landmarks,
    int k_neighbors,
    float* out_coords)
{
    if (!embeddings || !out_coords || n <= 0 || d_in <= 0 || d_out <= 0) {
        return;
    }

    // Map input data
    Map<const Matrix<float, Dynamic, Dynamic, RowMajor>> X(embeddings, n, d_in);

    // ===== STEP 1: Select landmarks (k-means++) =====
    std::vector<int> landmark_indices;
    landmark_indices.reserve(n_landmarks);

    std::mt19937 rng(42);  // Deterministic for reproducibility
    std::uniform_int_distribution<int> dist(0, n - 1);

    // First landmark: random
    landmark_indices.push_back(dist(rng));

    // Remaining landmarks: k-means++ initialization
    VectorXf min_distances = VectorXf::Constant(n, std::numeric_limits<float>::max());

    for (int l = 1; l < n_landmarks; l++) {
        int last_landmark = landmark_indices.back();
        VectorXf landmark_vec = X.row(last_landmark);

        // Update minimum distances
        for (int i = 0; i < n; i++) {
            float dist = (X.row(i) - landmark_vec.transpose()).squaredNorm();
            min_distances[i] = std::min(min_distances[i], dist);
        }

        // Choose next landmark proportional to min_distance^2
        std::discrete_distribution<int> weighted_dist(
            min_distances.data(),
            min_distances.data() + min_distances.size()
        );
        landmark_indices.push_back(weighted_dist(rng));
    }

    // ===== STEP 2: Construct k-NN graph =====
    // For efficiency, only compute graph on/from landmarks
    // Full Nyström extension later

    SparseMatrix<float> W(n_landmarks, n_landmarks);
    W.reserve(VectorXi::Constant(n_landmarks, k_neighbors));

    for (int i = 0; i < n_landmarks; i++) {
        VectorXf landmark_i = X.row(landmark_indices[i]);

        // Find k nearest neighbors among landmarks
        std::vector<std::pair<float, int>> distances;
        distances.reserve(n_landmarks);

        for (int j = 0; j < n_landmarks; j++) {
            if (i == j) continue;
            VectorXf landmark_j = X.row(landmark_indices[j]);
            float dist = (landmark_i - landmark_j).squaredNorm();
            distances.push_back({dist, j});
        }

        std::partial_sort(
            distances.begin(),
            distances.begin() + std::min(k_neighbors, static_cast<int>(distances.size())),
            distances.end()
        );

        // Add edges with Gaussian kernel weights
        float sigma = distances[k_neighbors / 2].first;  // Adaptive bandwidth
        for (int k = 0; k < std::min(k_neighbors, static_cast<int>(distances.size())); k++) {
            float dist = distances[k].first;
            int j = distances[k].second;
            float weight = std::exp(-dist / (2.0f * sigma));
            W.insert(i, j) = weight;
            W.insert(j, i) = weight;  // Symmetric
        }
    }

    W.makeCompressed();

    // ===== STEP 3: Compute Graph Laplacian =====
    // L = D - W (unnormalized Laplacian)
    VectorXf degrees = VectorXf::Zero(n_landmarks);
    for (int k = 0; k < W.outerSize(); ++k) {
        for (SparseMatrix<float>::InnerIterator it(W, k); it; ++it) {
            degrees[it.row()] += it.value();
        }
    }

    SparseMatrix<float> D(n_landmarks, n_landmarks);
    D.reserve(VectorXi::Constant(n_landmarks, 1));
    for (int i = 0; i < n_landmarks; i++) {
        D.insert(i, i) = degrees[i];
    }
    D.makeCompressed();

    SparseMatrix<float> L = D - W;

    // ===== STEP 4: Solve generalized eigenvalue problem =====
    // L v = λ D v
    // We want smallest non-zero eigenvalues

    // Convert to dense for eigendecomposition (landmarks should be manageable size)
    MatrixXf L_dense = MatrixXf(L);
    MatrixXf D_dense = MatrixXf(D);

    GeneralizedSelfAdjointEigenSolver<MatrixXf> eig(L_dense, D_dense);

    // Take eigenvectors for smallest non-zero eigenvalues (skip first - trivial)
    MatrixXf landmark_coords = eig.eigenvectors().leftCols(d_out + 1).rightCols(d_out);

    // ===== STEP 5: Nyström extension to all points =====
    // Compute adaptive bandwidth from landmark distances
    std::vector<float> landmark_distances;
    landmark_distances.reserve(n_landmarks * k_neighbors);
    for (int i = 0; i < n_landmarks; i++) {
        VectorXf landmark_i = X.row(landmark_indices[i]);
        for (int j = i + 1; j < std::min(i + k_neighbors, n_landmarks); j++) {
            VectorXf landmark_j = X.row(landmark_indices[j]);
            float dist = (landmark_i - landmark_j).norm();
            landmark_distances.push_back(dist);
        }
    }

    std::nth_element(
        landmark_distances.begin(),
        landmark_distances.begin() + landmark_distances.size() / 2,
        landmark_distances.end()
    );
    float median_dist = landmark_distances[landmark_distances.size() / 2];
    float sigma = median_dist * median_dist;  // Adaptive bandwidth

    MatrixXf all_coords(n, d_out);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        VectorXf point = X.row(i);
        VectorXf weights(n_landmarks);
        float total_weight = 0.0f;

        // Compute weights to all landmarks
        for (int l = 0; l < n_landmarks; l++) {
            VectorXf landmark = X.row(landmark_indices[l]);
            float dist_sq = (point - landmark).squaredNorm();
            float weight = std::exp(-dist_sq / (2.0f * sigma));
            weights[l] = weight;
            total_weight += weight;
        }

        // Normalize weights
        if (total_weight > 1e-8f) {
            weights /= total_weight;
        }

        // Interpolate coordinates
        all_coords.row(i) = weights.transpose() * landmark_coords;
    }

    // Normalize coordinates to have unit variance per dimension
    for (int d = 0; d < d_out; d++) {
        float mean = all_coords.col(d).mean();
        all_coords.col(d).array() -= mean;
        float std = std::sqrt(all_coords.col(d).array().square().sum() / n);
        if (std > 1e-8f) {
            all_coords.col(d) /= std;
        }
    }

    // Copy to output
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(out_coords, n, d_out) = all_coords;
}

} // extern "C"
