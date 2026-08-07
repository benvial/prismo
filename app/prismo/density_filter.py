import numpy as np
from scipy.sparse import csr_matrix


def assemble_filter_matrix(mesh_coords: np.ndarray, r_min: float = 50e-9) -> csr_matrix:
    """Build sparse convolution filter matrix from node coordinates.

    H_ij = max(0, r_min - dist(i,j))
    Returns csr_matrix.
    """
    coords = np.asarray(mesh_coords, dtype=float)
    n = coords.shape[0]

    # Build sparse matrix using pairwise distances
    # Only compute for pairs within r_min
    rows, cols, data = [], [], []
    for i in range(n):
        dist_i = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
        mask = dist_i <= r_min
        j_indices = np.where(mask)[0]
        for j in j_indices:
            rows.append(i)
            cols.append(j)
            data.append(max(0.0, r_min - dist_i[j]))

    return csr_matrix((data, (rows, cols)), shape=(n, n))


def apply_filter(rho: np.ndarray, H: csr_matrix) -> np.ndarray:
    """Apply density filter: rho_tilde = H @ rho / H_sum."""
    rho = np.asarray(rho, dtype=float)
    H_sum = np.asarray(H.sum(axis=1)).flatten()
    rho_tilde = H.dot(rho) / H_sum
    return rho_tilde


def vjp_filter(cotangent: np.ndarray, H: csr_matrix) -> np.ndarray:
    """Adjoint of density filter: dJ/drho = H @ (dJ/drho_tilde / H_sum).

    The cotangent is normalized by H_sum *before* the matmul: the forward
    pass normalizes each output row by H_sum_i, so the adjoint divides the
    incoming cotangent by H_sum_i before propagating through H.
    """
    cotangent = np.asarray(cotangent, dtype=float)
    H_sum = np.asarray(H.sum(axis=1)).flatten()
    return H.dot(cotangent / H_sum)
