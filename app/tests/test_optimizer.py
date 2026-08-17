"""Tests for the NLopt MMA optimization loop.

Ref: ticket 15.
"""

from unittest import mock

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from prismo.density_filter import assemble_filter_matrix  # noqa: E402
from prismo.optimizer import optimize_doping  # noqa: E402

RNG = np.random.default_rng(0)
jax.config.update("jax_enable_x64", True)

N_NODES = 16


def _grid_coords(n: int = 4, spacing: float = 20e-9) -> np.ndarray:
    xs, ys = np.meshgrid(
        np.arange(n) * spacing, np.arange(n) * spacing, indexing="xy",
    )
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


class TestOptimizeDopingStub:
    """Optimizer with stub pipeline (zero gradient)."""

    N_NODES = 8

    @pytest.fixture
    def rho0(self) -> np.ndarray:
        return np.full(self.N_NODES, 0.25, dtype=float)

    def test_runs_and_returns_valid(self, rho0):
        rho_opt, history = optimize_doping(rho0, max_iter=3)
        assert isinstance(rho_opt, np.ndarray)
        assert rho_opt.shape == (self.N_NODES,)
        assert np.all(rho_opt >= 0.0)
        assert np.all(rho_opt <= 1.0)
        assert isinstance(history, list)
        assert len(history) > 0

    def test_history_format(self, rho0):
        _, history = optimize_doping(rho0, max_iter=3)
        for entry in history:
            assert "iteration" in entry
            assert "delta_n_eff" in entry
            assert "delta_rho" in entry
            assert "grad_norm" in entry
            assert "wall_time" in entry
            assert isinstance(entry["iteration"], int)
            assert isinstance(entry["delta_n_eff"], float)
            assert isinstance(entry["delta_rho"], float)
            assert isinstance(entry["grad_norm"], float)
            assert isinstance(entry["wall_time"], float)

    def test_default_initial_rho(self):
        rho_opt, _ = optimize_doping(n_nodes=self.N_NODES, max_iter=3)
        assert rho_opt.shape == (self.N_NODES,)
        assert np.all(rho_opt >= 0.0)
        assert np.all(rho_opt <= 1.0)

    def test_missing_n_nodes_raises(self):
        with pytest.raises(ValueError):
            optimize_doping()

    def test_jit_flag_works(self, rho0):
        rho_opt, _ = optimize_doping(rho0, max_iter=3, use_jit=False)
        assert rho_opt.shape == (self.N_NODES,)

    def test_max_iter_respected(self, rho0):
        _, history = optimize_doping(rho0, max_iter=5)
        assert len(history) <= 5


class TestOptimizeDopingAnalytical:
    """Optimizer with a simple concave analytical pipeline.

    Uses `-sum((rho - target)^2)` so the maximiser is `rho = target`.
    """

    N_NODES = 4

    @pytest.fixture
    def rho_initial(self) -> np.ndarray:
        return np.full(self.N_NODES, 0.25, dtype=float)

    def test_converges_concave(self, rho_initial):
        target = jnp.full(self.N_NODES, 0.8, dtype=jnp.float64)

        def mock_pipeline(rho, **kwargs):
            diff = rho - target
            return -jnp.sum(diff**2)

        with mock.patch(
            "prismo.optimizer.pipeline", side_effect=mock_pipeline,
        ) as mock_pipe:
            rho_opt, _ = optimize_doping(
                initial_rho=rho_initial, max_iter=100, ftol_rel=1e-8,
            )
            assert mock_pipe.called

        np.testing.assert_allclose(rho_opt, np.asarray(target), rtol=0.05)

    def test_evaluates_pipeline_once_per_optimizer_callback(self, rho_initial):
        """One NLopt callback obtains value and gradient from one pipeline run."""
        target = jnp.full(self.N_NODES, 0.8, dtype=jnp.float64)

        def mock_pipeline(rho, **kwargs):
            return -jnp.sum((rho - target) ** 2)

        with mock.patch(
            "prismo.optimizer.pipeline", side_effect=mock_pipeline,
        ) as mock_pipe:
            _, history = optimize_doping(
                initial_rho=rho_initial,
                max_iter=3,
                use_jit=False,
            )

        assert mock_pipe.call_count == len(history)


class TestOptimizeDopingWithFilter:
    """Optimizer integrated with the density filter."""

    N_SIDE = 4

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
    def n_nodes(self) -> int:
        return self.N_SIDE * self.N_SIDE

    @pytest.fixture
    def rho0(self, n_nodes) -> np.ndarray:
        return np.full(n_nodes, 0.25, dtype=float)

    def test_runs_with_filter(self, rho0, H_dense):
        rho_opt, history = optimize_doping(
            initial_rho=rho0, H=H_dense, max_iter=3,
        )
        assert rho_opt.shape == rho0.shape
        assert len(history) > 0

    def test_box_bounds_preserved_with_filter(self, rho0, H_dense):
        rho_opt, _ = optimize_doping(
            initial_rho=rho0, H=H_dense, max_iter=5,
        )
        assert np.all(rho_opt >= 0.0)
        assert np.all(rho_opt <= 1.0)

    def test_builds_filter_from_coords(self, rho0, coords):
        rho_opt, _ = optimize_doping(
            initial_rho=rho0, mesh_coords=coords, r_min=50e-9, max_iter=2,
        )
        assert rho_opt.shape == rho0.shape
