"""Tests for the carrier-field -> gyptis-design-cell mesh-transfer operator.

Ticket 05: both solvers share one gmsh geometry, so the operator is an exact
local restriction -- each design cell is a shared-mesh triangle and its value is
the mean of its three vertex nodal values (weight 1/3 each). Only silicon
(design-cell) nodes contribute and each row is a partition of unity, so a uniform
input maps to a uniform output.
"""

import numpy as np
import pytest
from prismo.mesh_transfer import (
    MeshTransferOperator,
    build_mesh_transfer_operator,
)

# A [0,2] x [0,1] strip: left square is silicon, right square is oxide.
#   3(0,1) 4(1,1) 5(2,1)
#   0(0,0) 1(1,0) 2(2,0)
NODES = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
)
SILICON_TRIS = np.array([[0, 1, 4], [0, 4, 3]])  # left square only
OXIDE_ONLY_NODES = (2, 5)
# Each design cell is a shared-mesh triangle; its vertices are shared-mesh nodes.
DESIGN_VERTS = NODES[SILICON_TRIS]  # (2, 3, 2)
CENTROIDS = DESIGN_VERTS.mean(axis=1)


def test_build_returns_operator() -> None:
    op = build_mesh_transfer_operator(NODES, DESIGN_VERTS)
    assert isinstance(op, MeshTransferOperator)


def test_output_shape_matches_design_cells() -> None:
    op = build_mesh_transfer_operator(NODES, DESIGN_VERTS)
    assert op.matrix.shape == (len(DESIGN_VERTS), len(NODES))
    out = op(np.ones(len(NODES)))
    assert out.shape == (len(DESIGN_VERTS),)


def test_weights_are_one_third_on_each_vertex() -> None:
    op = build_mesh_transfer_operator(NODES, DESIGN_VERTS)
    dense = op.matrix.toarray()
    for cell, tri in enumerate(SILICON_TRIS):
        for node in tri:
            np.testing.assert_allclose(dense[cell, node], 1.0 / 3.0)


def test_rows_are_a_partition_of_unity() -> None:
    op = build_mesh_transfer_operator(NODES, DESIGN_VERTS)
    row_sums = np.asarray(op.matrix.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0)


def test_uniform_input_maps_to_uniform_output() -> None:
    op = build_mesh_transfer_operator(NODES, DESIGN_VERTS)
    out = op(np.full(len(NODES), 7.3))
    np.testing.assert_allclose(out, 7.3)


def test_only_silicon_nodes_contribute() -> None:
    op = build_mesh_transfer_operator(NODES, DESIGN_VERTS)
    dense = op.matrix.toarray()
    for node in OXIDE_ONLY_NODES:
        assert np.all(dense[:, node] == 0.0)

    # Perturbing an oxide-only node leaves the transferred field unchanged.
    field = np.ones(len(NODES))
    perturbed = field.copy()
    perturbed[OXIDE_ONLY_NODES[0]] += 100.0
    np.testing.assert_allclose(op(field), op(perturbed))


def test_reproduces_linear_field_exactly() -> None:
    """The mean of a linear field's vertex values is its value at the centroid."""
    op = build_mesh_transfer_operator(NODES, DESIGN_VERTS)
    a, b, c = 2.0, -3.0, 0.5
    nodal = a * NODES[:, 0] + b * NODES[:, 1] + c
    expected = a * CENTROIDS[:, 0] + b * CENTROIDS[:, 1] + c
    np.testing.assert_allclose(op(nodal), expected, rtol=1e-12)


def test_rejects_malformed_vertices() -> None:
    with pytest.raises(ValueError):
        build_mesh_transfer_operator(NODES, np.array([[0.0, 1.0]]))


def test_rejects_vertex_off_the_mesh() -> None:
    off_mesh = np.array([[[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]]])
    with pytest.raises(ValueError):
        build_mesh_transfer_operator(NODES, off_mesh)
