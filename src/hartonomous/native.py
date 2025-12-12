"""
Python bindings to hartonomous_native C++ library
Uses ctypes for zero-overhead FFI
"""

import ctypes
import numpy as np
from pathlib import Path
import platform

# Locate shared library
def find_native_lib():
    """Find the native library in various possible locations"""
    lib_name = {
        'Windows': 'hartonomous_native.dll',
        'Darwin': 'libhartonomous_native.dylib',
        'Linux': 'libhartonomous_native.so'
    }[platform.system()]

    # Try several locations
    search_paths = [
        Path(__file__).parent / lib_name,
        Path(__file__).parent.parent.parent / 'native' / 'build' / lib_name,
        Path('/usr/local/lib') / lib_name,
        Path('/usr/lib') / lib_name,
    ]

    for path in search_paths:
        if path.exists():
            return str(path)

    raise RuntimeError(f"Cannot find {lib_name}. Did you build the native library?")

_lib = ctypes.CDLL(find_native_lib())

# ==================== HILBERT ENCODING ====================

_lib.hilbert_encode_4d_batch.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # coords
    ctypes.c_int,                     # count
    ctypes.POINTER(ctypes.c_int64),   # out_hilbert
]
_lib.hilbert_encode_4d_batch.restype = None

def hilbert_encode_batch(coords: np.ndarray) -> np.ndarray:
    """
    Encode 4D coordinates to 4-manifold Hilbert representation.

    Args:
        coords: numpy array of shape (N, 4) with dtype float64

    Returns:
        numpy array of shape (N, 4) with dtype int64
        Each row is (h_xy, h_yz, h_zm, h_my)
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected shape (N, 4), got {coords.shape}")

    n = coords.shape[0]
    out = np.empty((n, 4), dtype=np.int64)

    _lib.hilbert_encode_4d_batch(
        coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    )

    return out

_lib.hilbert_decode_4d_batch.argtypes = [
    ctypes.POINTER(ctypes.c_int64),   # hilbert
    ctypes.c_int,                     # count
    ctypes.POINTER(ctypes.c_double),  # out_coords
]
_lib.hilbert_decode_4d_batch.restype = ctypes.c_int

def hilbert_decode_batch(hilbert: np.ndarray) -> np.ndarray:
    """
    Decode 4-manifold Hilbert back to 4D coordinates.
    Includes integrity checking.

    Args:
        hilbert: numpy array of shape (N, 4) with dtype int64

    Returns:
        numpy array of shape (N, 4) with dtype float64

    Raises:
        RuntimeError: If integrity check fails
    """
    hilbert = np.ascontiguousarray(hilbert, dtype=np.int64)
    if hilbert.ndim != 2 or hilbert.shape[1] != 4:
        raise ValueError(f"Expected shape (N, 4), got {hilbert.shape}")

    n = hilbert.shape[0]
    out = np.empty((n, 4), dtype=np.float64)

    ret = _lib.hilbert_decode_4d_batch(
        hilbert.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    if ret != 0:
        raise RuntimeError(f"Hilbert decode failed with code {ret} (integrity check)")

    return out

_lib.hilbert_pack_64bit.argtypes = [
    ctypes.POINTER(ctypes.c_int64),   # hilbert_full
    ctypes.c_int,                     # count
    ctypes.POINTER(ctypes.c_uint64),  # out_packed
]
_lib.hilbert_pack_64bit.restype = None

def hilbert_pack(hilbert: np.ndarray) -> np.ndarray:
    """
    Pack 4-manifold (168 bits) to 64 bits for spatial indexing.
    Lossy but sufficient for k-NN queries.
    """
    hilbert = np.ascontiguousarray(hilbert, dtype=np.int64)
    n = hilbert.shape[0]
    out = np.empty(n, dtype=np.uint64)

    _lib.hilbert_pack_64bit(
        hilbert.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    )

    return out

# ==================== DIMENSIONALITY REDUCTION ====================

_lib.pca_project.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # embeddings
    ctypes.c_int,                     # n
    ctypes.c_int,                     # d_in
    ctypes.c_int,                     # d_out
    ctypes.POINTER(ctypes.c_float),   # out_coords
]
_lib.pca_project.restype = None

def pca_project(embeddings: np.ndarray, n_components: int = 4) -> np.ndarray:
    """
    PCA projection for dimensionality reduction.

    Args:
        embeddings: numpy array of shape (N, D) with dtype float32
        n_components: Target dimensions (default 4)

    Returns:
        numpy array of shape (N, n_components) with dtype float32
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

    n, d_in = embeddings.shape
    out = np.empty((n, n_components), dtype=np.float32)

    _lib.pca_project(
        embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
        d_in,
        n_components,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )

    return out

_lib.laplacian_eigenmap.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # embeddings
    ctypes.c_int,                     # n
    ctypes.c_int,                     # d_in
    ctypes.c_int,                     # d_out
    ctypes.c_int,                     # n_landmarks
    ctypes.c_int,                     # k_neighbors
    ctypes.POINTER(ctypes.c_float),   # out_coords
]
_lib.laplacian_eigenmap.restype = None

def laplacian_eigenmap(
    embeddings: np.ndarray,
    n_components: int = 4,
    n_landmarks: int = 1000,
    k_neighbors: int = 10
) -> np.ndarray:
    """
    Laplacian Eigenmap with Neumann extension.
    Preserves manifold topology better than PCA.

    Args:
        embeddings: numpy array of shape (N, D)
        n_components: Target dimensions
        n_landmarks: Number of landmark points for Nyström extension
        k_neighbors: K for k-NN graph

    Returns:
        numpy array of shape (N, n_components)
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

    n, d_in = embeddings.shape
    n_landmarks = min(n_landmarks, n)  # Can't have more landmarks than points

    out = np.empty((n, n_components), dtype=np.float32)

    _lib.laplacian_eigenmap(
        embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
        d_in,
        n_components,
        n_landmarks,
        k_neighbors,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )

    return out

# ==================== VERSION ====================

_lib.hartonomous_version.argtypes = []
_lib.hartonomous_version.restype = ctypes.c_char_p

def version() -> str:
    """Get native library version string"""
    return _lib.hartonomous_version().decode('utf-8')

# Test on import
if __name__ == "__main__":
    print(f"Hartonomous Native Library: {version()}")

    # Quick test
    coords = np.array([[0.5, -0.3, 0.8, 0.0]], dtype=np.float64)
    hilbert = hilbert_encode_batch(coords)
    decoded = hilbert_decode_batch(hilbert)

    print(f"Original:  {coords[0]}")
    print(f"Hilbert:   {hilbert[0]}")
    print(f"Decoded:   {decoded[0]}")
    print(f"Error:     {np.abs(coords - decoded).max():.9f}")
