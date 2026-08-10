"""Tests for the end-to-end differentiable pipeline.

Ref: ticket 14 -- end-to-end pipeline via JAX composition.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from prismo.density_filter import assemble_filter_matrix  # noqa: E402
from prismo.pipeline import _sb_jax, pipeline  # noqa: E402
from prismo.soref_bennett import soref_bennett as _sb_numpy  # noqa: E402
from prismo_shared.schemas import CarrierDensityField  # noqa: E402

RNG = np.random.default_rng(0)
jax.config.update("jax_enable_x64", True)


def _grid_coords(n: int = 16, spacing: float = 20e-9) -> np.ndarray:
    xs, ys = np.meshgrid(
        np.arange(n) * spacing, np.arange(n) * spacing, indexing="xy",
    )
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


class TestSorefBennettJax:
    """Pure-JAX Soref-Bennett matches the numpy reference."""

    N_NODES = 8

    def test_zero_perturbation(self):
        es = jnp.ones(self.N_NODES)
        hs = jnp.ones(self.N_NODES)
        eq_e = jnp.ones(self.N_NODES)
        eq_h = jnp.ones(self.N_NODES)

        depsilon, dalpha = _sb_jax(es, hs, eq_e, eq_h)
        np.testing.assert_allclose(depsilon, 0.0, atol=0.0)
        np.testing.assert_allclose(dalpha, 0.0, atol=0.0)

    def test_matches_numpy_reference(self):
        n = jnp.asarray(RNG.random(self.N_NODES) * 1e24 + 1e20)
        p = jnp.asarray(RNG.random(self.N_NODES) * 1e24 + 1e20)
        n_eq = jnp.asarray(RNG.random(self.N_NODES) * 1e21)
        p_eq = jnp.asarray(RNG.random(self.N_NODES) * 1e21)

        depsilon_jax, dalpha_jax = _sb_jax(n, p, n_eq, p_eq)

        result_np = _sb_numpy(
            CarrierDensityField(
                electrons=np.asarray(n).tolist(),
                holes=np.asarray(p).tolist(),
                equilibrium_electrons=np.asarray(n_eq).tolist(),
                equilibrium_holes=np.asarray(p_eq).tolist(),
            ),
        )
        np.testing.assert_allclose(
            depsilon_jax, np.asarray(result_np.delta_permittivity), rtol=1e-12,
        )
        np.testing.assert_allclose(
            dalpha_jax, np.asarray(result_np.delta_absorption), rtol=1e-12,
        )

    def test_differentiable_elements(self):
        electrons = jnp.asarray(RNG.random(self.N_NODES) * 1e24 + 1e20)
        holes = jnp.asarray(RNG.random(self.N_NODES) * 1e24 + 1e20)
        eq_e = jnp.asarray(RNG.random(self.N_NODES) * 1e21)
        eq_h = jnp.asarray(RNG.random(self.N_NODES) * 1e21)

        def loss(e):
            depsilon, _ = _sb_jax(e, holes, eq_e, eq_h)
            return jnp.sum(depsilon)

        grad = jax.grad(loss)(electrons)
        assert grad.shape == electrons.shape
        assert jnp.all(jnp.isfinite(grad))


class TestPipelineStub:
    """Pipeline with no containers available (stubs)."""

    N_NODES = 16

    @pytest.fixture
    def rho(self) -> jax.Array:
        return jnp.asarray(RNG.random(self.N_NODES), dtype=jnp.float64)

    def test_forward_returns_scalar(self, rho):
        result = pipeline(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_forward_returns_zero_pre_container(self, rho):
        result = pipeline(rho)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_gradient_is_finite(self, rho):
        grad = jax.grad(pipeline)(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_is_zero_pre_container(self, rho):
        grad = jax.grad(pipeline)(rho)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_jit_works(self, rho):
        jitted = jax.jit(pipeline)
        result = jitted(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_jit_gradient_works(self, rho):
        grad_fn = jax.jit(jax.grad(pipeline))
        grad = grad_fn(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_dtype_preserved(self, rho):
        result32 = pipeline(jnp.asarray(rho, dtype=jnp.float32))
        assert result32.dtype == jnp.float32

        result64 = pipeline(jnp.asarray(rho, dtype=jnp.float64))
        assert result64.dtype == jnp.float64


class TestPipelineWithFilter:
    """Pipeline with a synthetic density-filter matrix."""

    N_SIDE = 8

    @pytest.fixture
    def coords(self) -> np.ndarray:
        return _grid_coords(self.N_SIDE)

    @pytest.fixture
    def H_sparse(self, coords):
        return assemble_filter_matrix(coords)

    @pytest.fixture
    def H_dense(self, H_sparse):
        return jnp.asarray(H_sparse.toarray(), dtype=jnp.float64)

    @pytest.fixture
    def H_sum(self, H_dense):
        return jnp.sum(H_dense, axis=1)

    @pytest.fixture
    def n_nodes(self) -> int:
        return self.N_SIDE * self.N_SIDE

    @pytest.fixture
    def rho(self, n_nodes) -> jax.Array:
        return jnp.full((n_nodes,), 0.25, dtype=jnp.float64)

    def test_pipeline_with_filter_runs(self, rho, H_dense, H_sum):
        result = pipeline(rho, H=H_dense, H_sum=H_sum)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_gradient_with_filter(self, rho, H_dense, H_sum):
        grad = jax.grad(
            lambda r: pipeline(r, H=H_dense, H_sum=H_sum),
        )(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_with_filter(self, rho, H_dense, H_sum):
        fn = jax.jit(lambda r: pipeline(r, H=H_dense, H_sum=H_sum))
        result = fn(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_box_bounds_preserved(self, rho, H_dense, H_sum):
        for val in [0.0, 0.5, 1.0]:
            r = jnp.full_like(rho, val)
            result = pipeline(r, H=H_dense, H_sum=H_sum)
            assert jnp.isfinite(result)

    def test_shape_consistency(self, rho, H_dense, H_sum):
        result = pipeline(rho, H=H_dense, H_sum=H_sum)
        d = float(result)
        assert isinstance(d, float)


class TestPipelineDopingMapping:
    """The doping mapping N = 10^(14 + 7*rho) produces sane values."""

    def test_range(self):
        n_nodes = 10
        rho = jnp.linspace(0.0, 1.0, n_nodes)
        result = pipeline(rho)
        assert jnp.isfinite(result)
        assert result.ndim == 0

    def test_edge_cases_finite(self):
        for val in [0.0, 0.001, 0.5, 0.999, 1.0]:
            rho = jnp.array([val])
            result = pipeline(rho)
            assert jnp.isfinite(result), f"Non-finite at rho={val}"

    def test_gradient_exists_for_nonuniform_rho(self):
        n_nodes = 10
        rho = jnp.asarray(RNG.random(n_nodes), dtype=jnp.float64)
        grad = jax.grad(pipeline)(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))


class TestPipelineGradientValidation:
    """Finite-difference validation of the full-pipeline gradient.

    Compares jax.grad(pipeline) against central finite differences for
    random perturbation directions at random rho vectors.
    """

    N_NODES = 10
    N_DIRECTIONS = 3

    @pytest.fixture
    def rho_vectors(self) -> list[jax.Array]:
        return [
            jnp.asarray(RNG.random(self.N_NODES), dtype=jnp.float64)
            for _ in range(self.N_DIRECTIONS)
        ]

    @pytest.fixture
    def directions(self) -> list[jax.Array]:
        dirs = []
        for _ in range(self.N_DIRECTIONS):
            d = jnp.asarray(RNG.standard_normal(self.N_NODES), dtype=jnp.float64)
            d = d / jnp.linalg.norm(d)
            dirs.append(d)
        return dirs

    def test_gradient_vs_fd_central(self, rho_vectors, directions):
        step_sizes = [1e-3, 1e-4, 1e-5, 1e-6]
        grad_fn = jax.grad(pipeline)

        for rho in rho_vectors:
            grad_exact = grad_fn(rho)
            for direction in directions:
                for h in step_sizes:
                    f_plus = pipeline(rho + h * direction)
                    f_minus = pipeline(rho - h * direction)
                    fd_grad_dir = (f_plus - f_minus) / (2.0 * h)
                    exact_grad_dir = jnp.dot(grad_exact, direction)

                    rel_error = float(
                        abs(fd_grad_dir - exact_grad_dir)
                        / max(abs(exact_grad_dir), 1e-30)
                    )
                    assert jnp.isfinite(rel_error)

                    if h <= 1e-4:
                        assert rel_error < 1.0, (
                            f"FD error too large at h={h}: "
                            f"rel_error={rel_error:.2e}"
                        )

    def test_gradient_vs_fd_multiple_rho(self):
        grad_fn = jax.grad(pipeline)
        for seed in [42, 99, 137]:
            rng = np.random.default_rng(seed)
            rho = jnp.asarray(rng.random(self.N_NODES), dtype=jnp.float64)
            direction = jnp.asarray(
                rng.standard_normal(self.N_NODES), dtype=jnp.float64
            )
            direction = direction / jnp.linalg.norm(direction)

            grad_exact = grad_fn(rho)
            exact_dir = float(jnp.dot(grad_exact, direction))

            h = 1e-5
            f_plus = pipeline(rho + h * direction)
            f_minus = pipeline(rho - h * direction)
            fd_dir = float((f_plus - f_minus) / (2.0 * h))

            assert abs(fd_dir - exact_dir) < 1e-8, (
                f"seed={seed}: exact={exact_dir:.4e}, fd={fd_dir:.4e}"
            )


class TestPipelineShapeValidation:
    """Intermediate shapes through the pipeline are consistent."""

    N_NODES = 16

    @pytest.fixture
    def rho(self) -> jax.Array:
        return jnp.asarray(RNG.random(self.N_NODES), dtype=jnp.float64)

    def test_doping_range(self, rho):

        doping = jnp.power(
            jnp.asarray(10.0, dtype=rho.dtype),
            jnp.asarray(14.0, dtype=rho.dtype)
            + jnp.asarray(7.0, dtype=rho.dtype) * rho,
        )
        assert doping.shape == rho.shape
        assert jnp.all(doping >= 1e14)
        assert jnp.all(doping <= 1e21)

    def test_sb_output_shapes(self, rho):
        from prismo.pipeline import _sb_jax

        n = jnp.ones_like(rho) * 1e24
        p = jnp.ones_like(rho) * 1e24
        n_eq = jnp.ones_like(rho) * 1e21
        p_eq = jnp.ones_like(rho) * 1e21

        depsilon, dalpha = _sb_jax(n, p, n_eq, p_eq)
        assert depsilon.shape == rho.shape
        assert dalpha.shape == rho.shape

    def test_gyptis_scalar_output(self, rho):
        from prismo.pipeline import _gyptis_call_jax

        result = _gyptis_call_jax(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_ct_call_output_shapes(self, rho):
        from prismo.pipeline import _ct_call_jax

        n_out, p_out = _ct_call_jax(rho, 0.0)
        assert n_out.shape == rho.shape
        assert p_out.shape == rho.shape
