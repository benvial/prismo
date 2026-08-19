"""Carrier-field -> gyptis-design-cell mesh-transfer operator (ticket 04).

The shared charge-transport mesh and the gyptis design mesh do not coincide
(spike 01 decision 5), so the nodal permittivity perturbation is carried onto
the gyptis design cells by *point-location interpolation*: each design-cell
centroid is located in the shared mesh's silicon triangulation and the nodal
field is interpolated there with the triangle's linear (barycentric) shape
functions.

The result is a static, sparse operator ``T`` of shape
``(n_design_cells, n_nodes)``. Two properties hold by construction:

- **Silicon-only support.** Only nodes of the silicon triangulation carry
  weight; oxide-only nodes have zero columns.
- **Partition of unity.** Each row sums to one, so a uniform nodal field maps
  to the same uniform field on the design cells -- the invariant the pipeline
  relies on to keep the background solve rho-independent (ticket 05).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["MeshTransferOperator", "build_mesh_transfer_operator"]


@dataclass(frozen=True)
class MeshTransferOperator:
    """A static nodal-field -> design-cell interpolation, as a sparse matrix.

    Attributes:
        matrix: Sparse ``(n_design_cells, n_nodes)`` operator. Calling the
            instance applies it to a nodal field.
    """

    matrix: csr_matrix

    def __call__(self, nodal_field: np.ndarray) -> np.ndarray:
        """Interpolate a nodal field onto the design cells."""
        return self.matrix @ np.asarray(nodal_field)

    def dense(self) -> np.ndarray:
        """The operator as a dense ``(n_design_cells, n_nodes)`` array.

        The pipeline needs a dense, JAX-traceable matrix to apply the transfer
        inside an autodiff pass; pass ``operator.dense()`` as ``design_transfer``.
        """
        return self.matrix.toarray()

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_design_cells, n_nodes)``."""
        return self.matrix.shape


def _barycentric(
    points: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Barycentric coordinates of ``points`` in each triangle ``(v0, v1, v2)``.

    Args:
        points: ``(n_points, 2)`` query coordinates.
        v0: ``(n_tri, 2)`` first vertex of each triangle.
        v1: ``(n_tri, 2)`` second vertex of each triangle.
        v2: ``(n_tri, 2)`` third vertex of each triangle.

    Returns:
        ``(n_points, n_tri, 3)`` barycentric coordinates; they sum to one along
        the last axis and are all non-negative exactly inside a triangle.
    """
    # Solve [ (v0-v2) (v1-v2) ] [l0 l1]^T = (p - v2) per triangle, per point.
    e0 = v0 - v2  # (n_tri, 2)
    e1 = v1 - v2
    det = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]  # (n_tri,)
    det = np.where(det == 0.0, np.finfo(float).tiny, det)

    rel = points[:, None, :] - v2[None, :, :]  # (n_points, n_tri, 2)
    l0 = (rel[..., 0] * e1[None, :, 1] - rel[..., 1] * e1[None, :, 0]) / det[None, :]
    l1 = (e0[None, :, 0] * rel[..., 1] - e0[None, :, 1] * rel[..., 0]) / det[None, :]
    l2 = 1.0 - l0 - l1
    return np.stack([l0, l1, l2], axis=-1)


def _weights_for_point(bary_point: np.ndarray, tol: float) -> tuple[int, np.ndarray]:
    """Pick a triangle for one point and return (triangle index, 3 weights).

    Prefers a triangle that contains the point (all barycentric coords >= -tol).
    Otherwise falls back to the triangle closest to containing it and clamps the
    barycentric coordinates onto the simplex, preserving the partition of unity.

    Args:
        bary_point: ``(n_tri, 3)`` barycentric coords of the point per triangle.
        tol: Inclusion tolerance for the containment test.
    """
    inside = np.all(bary_point >= -tol, axis=1)
    if np.any(inside):
        tri = int(np.argmax(inside))
        weights = np.clip(bary_point[tri], 0.0, None)
    else:
        # Closest to the simplex: smallest total negative barycentric mass.
        outside_mass = np.sum(np.clip(-bary_point, 0.0, None), axis=1)
        tri = int(np.argmin(outside_mass))
        weights = np.clip(bary_point[tri], 0.0, None)

    total = weights.sum()
    if total == 0.0:  # degenerate; fall back to the nearest single vertex.
        weights = np.zeros(3)
        weights[int(np.argmax(bary_point[tri]))] = 1.0
        total = 1.0
    return tri, weights / total


def build_mesh_transfer_operator(
    node_coords: np.ndarray,
    silicon_triangles: np.ndarray,
    design_centroids: np.ndarray,
    tol: float = 1e-9,
) -> MeshTransferOperator:
    """Build the nodal-field -> design-cell interpolation operator.

    Args:
        node_coords: ``(n_nodes, 2)`` shared-mesh node coordinates.
        silicon_triangles: ``(n_tri, 3)`` node indices of the silicon
            triangulation only -- oxide elements are excluded so oxide nodes
            never carry weight.
        design_centroids: ``(n_design_cells, 2)`` gyptis design-cell centroids
            (from ``gyptis.tesseract_api.design_cell_centroids``); the operator's
            row order follows this array.
        tol: Barycentric containment tolerance for point location.

    Returns:
        A :class:`MeshTransferOperator` wrapping the sparse ``(n_design_cells,
        n_nodes)`` matrix.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    tris = np.asarray(silicon_triangles)
    centroids = np.asarray(design_centroids, dtype=float)

    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("silicon_triangles must have shape (n_tri, 3)")
    if node_coords.ndim != 2 or node_coords.shape[1] != 2:
        raise ValueError("node_coords must have shape (n_nodes, 2)")
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError("design_centroids must have shape (n_design_cells, 2)")

    n_nodes = node_coords.shape[0]
    n_design = centroids.shape[0]

    v0 = node_coords[tris[:, 0]]
    v1 = node_coords[tris[:, 1]]
    v2 = node_coords[tris[:, 2]]
    bary = _barycentric(centroids, v0, v1, v2)  # (n_design, n_tri, 3)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row in range(n_design):
        tri, weights = _weights_for_point(bary[row], tol)
        for corner in range(3):
            rows.append(row)
            cols.append(int(tris[tri, corner]))
            data.append(float(weights[corner]))

    matrix = csr_matrix(
        (data, (rows, cols)), shape=(n_design, n_nodes), dtype=float
    )
    return MeshTransferOperator(matrix=matrix)
