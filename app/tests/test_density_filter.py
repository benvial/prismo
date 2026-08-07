"""Tests for the density filter implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import issparse
from prismo.density_filter import (
    apply_filter,
    assemble_filter_matrix,
    vjp_filter,
)

N = 32
RNG = np.random.default_rng(0)


def _grid_coords(n=N, spacing=20e-9):
    """Build a regular n x n grid of node coordinates."""
    xs, ys = np.meshgrid(
        np.arange(n) * spacing, np.arange(n) * spacing, indexing="xy"
    )
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


@pytest.fixture
def coords():
    return _grid_coords()


@pytest.fixture
def H(coords):
    return assemble_filter_matrix(coords)


def test_assemble_filter_matrix_shapes(H, coords):
    n = coords.shape[0]
    assert issparse(H)
    assert H.shape == (n, n)
    assert H.getformat() == "csr"


def test_filter_matrix_symmetric(H):
    dense = H.toarray()
    np.testing.assert_allclose(dense, dense.T)


def test_filter_matrix_diagonal_positive(H):
    diag = H.diagonal()
    assert np.all(diag > 0)
    np.testing.assert_allclose(diag, 50e-9 * np.ones_like(diag))


def test_uniform_preserved(H):
    n = H.shape[0]
    rho = np.ones(n)
    rho_tilde = apply_filter(rho, H)
    np.testing.assert_allclose(rho_tilde, rho)


def test_box_preservation(H):
    n = H.shape[0]
    rho = RNG.random(n)
    rho_tilde = apply_filter(rho, H)
    assert np.all(rho_tilde >= 0.0)
    assert np.all(rho_tilde <= 1.0)


def test_apply_filter_shape(H):
    n = H.shape[0]
    rho = RNG.random(n)
    rho_tilde = apply_filter(rho, H)
    assert rho_tilde.shape == (n,)


def test_vjp_filter_shape(H):
    n = H.shape[0]
    cotangent = RNG.random(n)
    out = vjp_filter(cotangent, H)
    assert out.shape == (n,)


def test_vjp_filter_linear(H):
    n = H.shape[0]
    c1 = RNG.random(n)
    c2 = RNG.random(n)
    a, b = 0.3, 0.7
    lhs = vjp_filter(a * c1 + b * c2, H)
    rhs = a * vjp_filter(c1, H) + b * vjp_filter(c2, H)
    np.testing.assert_allclose(lhs, rhs)


def test_vjp_filter_matches_finite_difference(H):
    n = H.shape[0]
    rho = RNG.random(n)
    cotangent = RNG.random(n)

    def loss(r):
        return float(np.dot(cotangent, apply_filter(r, H)))

    eps = 1e-6
    fd = np.zeros(n)
    for i in range(n):
        r_plus = rho.copy()
        r_plus[i] += eps
        fd[i] = (loss(r_plus) - loss(rho)) / eps

    grad = vjp_filter(cotangent, H)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-8)


def test_gradient_consistency_jax():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    coords = _grid_coords()
    H = assemble_filter_matrix(coords)
    H_dense = jnp.asarray(H.toarray())
    H_sum = jnp.asarray(np.asarray(H.sum(axis=1)).flatten())

    n = coords.shape[0]
    rho = jnp.asarray(RNG.random(n))
    ct = jnp.asarray(RNG.random(n))

    def loss(r):
        rho_tilde = H_dense @ r / H_sum
        return jnp.sum(ct * rho_tilde)

    jax_grad = np.asarray(jax.grad(loss)(rho))
    expected = vjp_filter(np.asarray(ct), H)
    np.testing.assert_allclose(jax_grad, expected)


class TestDensityFilterWithMesh:
    """Integration: real mesh coordinates → filter pipeline."""

    @pytest.fixture
    def mesh_coords(self):
        pytest.importorskip("gmsh")
        import gmsh  # type: ignore[import-untyped]
        from prismo.waveguide_mesh import (
            RibWaveguideGeometry,
            build_rib_waveguide_mesh_via_gmsh,
            read_mesh_node_coordinates,
        )

        gmsh.initialize()
        try:
            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
                mesh_path = f.name
            try:
                build_rib_waveguide_mesh_via_gmsh(
                    mesh_path, RibWaveguideGeometry()
                )
                coords = read_mesh_node_coordinates(mesh_path)
                assert coords.shape[0] > 0
                assert coords.shape[1] == 2
                yield coords
            finally:
                Path(mesh_path).unlink(missing_ok=True)
        finally:
            gmsh.finalize()

    @pytest.fixture
    def H(self, mesh_coords):
        return assemble_filter_matrix(mesh_coords)

    def test_filter_matrix_from_mesh_coords(self, H, mesh_coords):
        n = mesh_coords.shape[0]
        assert issparse(H)
        assert H.shape == (n, n)
        assert H.getformat() == "csr"

    def test_filter_matrix_symmetric_from_mesh(self, H):
        dense = H.toarray()
        np.testing.assert_allclose(dense, dense.T)

    def test_uniform_preserved_from_mesh(self, H):
        n = H.shape[0]
        rho = np.ones(n)
        rho_tilde = apply_filter(rho, H)
        np.testing.assert_allclose(rho_tilde, rho)

    def test_box_preservation_from_mesh(self, H):
        n = H.shape[0]
        rho = np.random.default_rng(42).random(n)
        rho_tilde = apply_filter(rho, H)
        assert np.all(rho_tilde >= 0.0)
        assert np.all(rho_tilde <= 1.0)

    def test_vjp_adjoint_consistency_from_mesh(self, H):
        n = H.shape[0]
        rng = np.random.default_rng(7)
        rho = rng.random(n)
        cotangent = rng.random(n)

        def loss(r):
            return float(np.dot(cotangent, apply_filter(r, H)))

        eps = 1e-6
        fd = np.zeros(n)
        for i in range(n):
            r_plus = rho.copy()
            r_plus[i] += eps
            fd[i] = (loss(r_plus) - loss(rho)) / eps

        grad = vjp_filter(cotangent, H)
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-8)

    def test_silicon_node_indices_valid(self, mesh_coords):
        expected_tags = set(range(mesh_coords.shape[0]))
        assert len(expected_tags) == mesh_coords.shape[0]
        assert np.all(np.isfinite(mesh_coords))
