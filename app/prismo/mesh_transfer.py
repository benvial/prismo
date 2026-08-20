"""Carrier-field -> gyptis-design-cell mesh-transfer operator (ticket 05).

Both solvers now share one gmsh geometry (ticket 05), so the transfer is an
*exact local restriction* rather than a cross-mesh interpolation: every gyptis
design cell is a triangle of the shared mesh whose three vertices are shared-mesh
nodes. The design-cell value is the mean of its three vertex nodal values -- the
value a linear nodal field takes at the cell centroid -- so the operator has
weight ``1/3`` on each vertex node and nothing else.

The result is a static, sparse operator ``T`` of shape
``(n_design_cells, n_nodes)``. Two properties hold by construction:

- **Silicon-only support.** A design cell is a silicon rib cell, so only silicon
  nodes carry weight; oxide-only nodes have zero columns.
- **Partition of unity.** Each row sums to one (three weights of ``1/3``), so a
  uniform nodal field maps to the same uniform field on the design cells -- the
  invariant the pipeline relies on to keep the background solve rho-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

__all__ = ["MeshTransferOperator", "build_mesh_transfer_operator"]


@dataclass(frozen=True)
class MeshTransferOperator:
    """A static nodal-field -> design-cell restriction, as a sparse matrix.

    Attributes:
        matrix: Sparse ``(n_design_cells, n_nodes)`` operator. Calling the
            instance applies it to a nodal field.
    """

    matrix: csr_matrix

    def __call__(self, nodal_field: np.ndarray) -> np.ndarray:
        """Restrict a nodal field onto the design cells."""
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


def build_mesh_transfer_operator(
    node_coords: np.ndarray,
    design_cell_vertices: np.ndarray,
    tol: float = 1e-6,
) -> MeshTransferOperator:
    """Build the exact nodal-field -> design-cell restriction operator.

    Args:
        node_coords: ``(n_nodes, 2)`` shared-mesh node coordinates, in the same
            frame and units as ``design_cell_vertices``.
        design_cell_vertices: ``(n_design_cells, 3, 2)`` vertex coordinates of
            each gyptis design cell (from ``write_mesh`` /
            ``gyptis.tesseract_api.design_cell_vertices``); the operator's row
            order follows this array. Each vertex is a shared-mesh node.
        tol: Maximum distance for matching a design-cell vertex to a shared-mesh
            node. The two are the same mesh, so a match is essentially exact;
            the tolerance only absorbs gmsh/dolfin float round-trip.

    Returns:
        A :class:`MeshTransferOperator` wrapping the sparse ``(n_design_cells,
        n_nodes)`` matrix with weight ``1/3`` on each of a cell's three vertices.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    vertices = np.asarray(design_cell_vertices, dtype=float)

    if node_coords.ndim != 2 or node_coords.shape[1] != 2:
        raise ValueError("node_coords must have shape (n_nodes, 2)")
    if vertices.ndim != 3 or vertices.shape[1:] != (3, 2):
        raise ValueError("design_cell_vertices must have shape (n_design_cells, 3, 2)")

    n_nodes = node_coords.shape[0]
    n_design = vertices.shape[0]

    # Same mesh: a design-cell vertex coincides with exactly one shared-mesh node.
    tree = cKDTree(node_coords)
    dist, node_idx = tree.query(vertices.reshape(-1, 2))
    if n_design and dist.max() > tol:
        raise ValueError(
            "a design-cell vertex did not coincide with any shared-mesh node "
            f"(max distance {dist.max():.3e} > tol {tol:.3e}); the design cells "
            "and node_coords must come from the same unified mesh"
        )
    node_idx = node_idx.reshape(n_design, 3)

    rows = np.repeat(np.arange(n_design), 3)
    cols = node_idx.reshape(-1)
    data = np.full(rows.size, 1.0 / 3.0)
    matrix = csr_matrix((data, (rows, cols)), shape=(n_design, n_nodes), dtype=float)
    return MeshTransferOperator(matrix=matrix)
