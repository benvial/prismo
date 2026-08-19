"""Tests for the carrier-field -> gyptis-design-cell mesh-transfer operator.

Ticket 04: a static operator maps a nodal permittivity field on the shared
mesh onto the gyptis design-region cells via point-location interpolation
(non-matching meshes; spike 01 decision 5). Only silicon nodes contribute and
each row is a partition of unity, so a uniform input maps to a uniform output.
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
# Interior points of each silicon triangle.
CENTROIDS = np.array([[0.6, 0.3], [0.3, 0.6]])


def test_build_returns_operator() -> None:
    op = build_mesh_transfer_operator(NODES, SILICON_TRIS, CENTROIDS)
    assert isinstance(op, MeshTransferOperator)


def test_output_shape_matches_design_cells() -> None:
    op = build_mesh_transfer_operator(NODES, SILICON_TRIS, CENTROIDS)
    assert op.matrix.shape == (len(CENTROIDS), len(NODES))
    out = op(np.ones(len(NODES)))
    assert out.shape == (len(CENTROIDS),)


def test_rows_are_a_partition_of_unity() -> None:
    op = build_mesh_transfer_operator(NODES, SILICON_TRIS, CENTROIDS)
    row_sums = np.asarray(op.matrix.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0)


def test_uniform_input_maps_to_uniform_output() -> None:
    op = build_mesh_transfer_operator(NODES, SILICON_TRIS, CENTROIDS)
    out = op(np.full(len(NODES), 7.3))
    np.testing.assert_allclose(out, 7.3)


def test_only_silicon_nodes_contribute() -> None:
    op = build_mesh_transfer_operator(NODES, SILICON_TRIS, CENTROIDS)
    dense = op.matrix.toarray()
    for node in OXIDE_ONLY_NODES:
        assert np.all(dense[:, node] == 0.0)

    # Perturbing an oxide-only node leaves the transferred field unchanged.
    field = np.ones(len(NODES))
    perturbed = field.copy()
    perturbed[OXIDE_ONLY_NODES[0]] += 100.0
    np.testing.assert_allclose(op(field), op(perturbed))


def test_reproduces_linear_field_exactly() -> None:
    """Barycentric interpolation is exact for a linear nodal field."""
    op = build_mesh_transfer_operator(NODES, SILICON_TRIS, CENTROIDS)
    a, b, c = 2.0, -3.0, 0.5
    nodal = a * NODES[:, 0] + b * NODES[:, 1] + c
    expected = a * CENTROIDS[:, 0] + b * CENTROIDS[:, 1] + c
    np.testing.assert_allclose(op(nodal), expected, rtol=1e-12)


def test_point_outside_silicon_still_partition_of_unity() -> None:
    """A centroid just outside the silicon region falls back to silicon nodes."""
    outside = np.array([[1.4, 0.5]])  # inside the oxide square
    op = build_mesh_transfer_operator(NODES, SILICON_TRIS, outside)
    row_sums = np.asarray(op.matrix.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0)
    dense = op.matrix.toarray()
    for node in OXIDE_ONLY_NODES:
        assert np.all(dense[:, node] == 0.0)


def test_rejects_malformed_triangles() -> None:
    with pytest.raises(ValueError):
        build_mesh_transfer_operator(NODES, np.array([[0, 1]]), CENTROIDS)
