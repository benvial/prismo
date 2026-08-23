"""Tests for the NLopt MMA optimization loop.

Ref: ticket 15.
"""

import json
from itertools import pairwise
from unittest import mock

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from _doubles import stub_components as _stub_components  # noqa: E402
from prismo.density_filter import assemble_filter_matrix  # noqa: E402
from prismo.optimizer import optimize_doping  # noqa: E402
from prismo.pipeline import PipelineTerms  # noqa: E402

RNG = np.random.default_rng(0)
jax.config.update("jax_enable_x64", True)

N_NODES = 16


def _with_terms(objective_fn):
    """Wrap a scalar objective mock into the ``(objective, terms)`` seam (ticket 25).

    The mock's value doubles as Δneff.
    """

    def mock_pipeline_with_terms(rho, **kwargs):
        value = objective_fn(rho, **kwargs)
        return value, PipelineTerms(delta_neff=value, modal_loss_db_cm=jnp.nan)

    return mock_pipeline_with_terms


def _grid_coords(n: int = 4, spacing: float = 20e-9) -> np.ndarray:
    xs, ys = np.meshgrid(
        np.arange(n) * spacing,
        np.arange(n) * spacing,
        indexing="xy",
    )
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


class TestOptimizeDopingStub:
    """Optimizer with stub pipeline (zero gradient)."""

    N_NODES = 8

    @pytest.fixture
    def rho0(self) -> np.ndarray:
        return np.full(self.N_NODES, 0.25, dtype=float)

    def test_runs_and_returns_valid(self, rho0):
        rho_opt, history = optimize_doping(
            rho0, max_iter=3, components=_stub_components()
        )
        assert isinstance(rho_opt, np.ndarray)
        assert rho_opt.shape == (self.N_NODES,)
        assert np.all(rho_opt >= 0.0)
        assert np.all(rho_opt <= 1.0)
        assert isinstance(history, list)
        assert len(history) > 0

    def test_history_format(self, rho0):
        _, history = optimize_doping(rho0, max_iter=3, components=_stub_components())
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
        rho_opt, _ = optimize_doping(
            n_nodes=self.N_NODES, max_iter=3, components=_stub_components()
        )
        assert rho_opt.shape == (self.N_NODES,)
        assert np.all(rho_opt >= 0.0)
        assert np.all(rho_opt <= 1.0)

    def test_missing_n_nodes_raises(self):
        with pytest.raises(ValueError):
            optimize_doping()

    def test_jit_flag_works(self, rho0):
        rho_opt, _ = optimize_doping(
            rho0, max_iter=3, use_jit=False, components=_stub_components()
        )
        assert rho_opt.shape == (self.N_NODES,)

    def test_max_iter_respected(self, rho0):
        _, history = optimize_doping(rho0, max_iter=5, components=_stub_components())
        assert len(history) <= 5

    def test_iteration_callback_receives_each_solver_candidate(self, rho0):
        received: list[tuple[int, np.ndarray]] = []

        _, history = optimize_doping(
            rho0,
            max_iter=3,
            on_iteration=lambda iteration, rho: received.append((iteration, rho)),
            components=_stub_components(),
        )

        assert [iteration for iteration, _ in received] == list(
            range(1, len(history) + 1)
        )
        assert all(rho.shape == rho0.shape for _, rho in received)


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
            "prismo.optimizer.pipeline_with_terms",
            side_effect=_with_terms(mock_pipeline),
        ) as mock_pipe:
            rho_opt, _ = optimize_doping(
                initial_rho=rho_initial,
                max_iter=100,
                ftol_rel=1e-8,
            )
            assert mock_pipe.called

        np.testing.assert_allclose(rho_opt, np.asarray(target), rtol=0.05)

    def test_evaluates_pipeline_once_per_optimizer_callback(self, rho_initial):
        """One NLopt callback obtains value and gradient from one pipeline run."""
        target = jnp.full(self.N_NODES, 0.8, dtype=jnp.float64)

        def mock_pipeline(rho, **kwargs):
            return -jnp.sum((rho - target) ** 2)

        with mock.patch(
            "prismo.optimizer.pipeline_with_terms",
            side_effect=_with_terms(mock_pipeline),
        ) as mock_pipe:
            _, history = optimize_doping(
                initial_rho=rho_initial,
                max_iter=3,
                use_jit=False,
            )

        assert mock_pipe.call_count == len(history)

    def test_records_combined_callback_timing(self, rho_initial):
        """History exposes the wall time and component phases of each callback."""
        target = jnp.full(self.N_NODES, 0.8, dtype=jnp.float64)

        def mock_pipeline(rho, **kwargs):
            return -jnp.sum((rho - target) ** 2)

        with mock.patch(
            "prismo.optimizer.pipeline_with_terms",
            side_effect=_with_terms(mock_pipeline),
        ):
            _, history = optimize_doping(
                initial_rho=rho_initial,
                max_iter=3,
                use_jit=False,
            )

        for entry in history:
            assert entry["callback_time"] >= 0.0

    def test_combined_callback_uses_public_component_calls_once(self, rho_initial):
        """One callback makes two bias forwards and their matching VJPs."""
        from prismo.pipeline import (
            PipelineComponents,
            build_chargetransport_component,
            build_gyptis_components,
        )

        class FakeChargeTransport:
            def __init__(self):
                self.forward_biases: list[float] = []
                self.vjp_biases: list[float] = []

            def apply(self, inputs):
                self.forward_biases.append(inputs["bias_voltage"])
                doping = np.asarray(inputs["doping"], dtype=float)
                carrier = np.full_like(
                    doping,
                    1e18 if inputs["bias_voltage"] == 0.0 else 0.0,
                )
                return {"electrons": carrier, "holes": carrier}

            def vector_jacobian_product(
                self,
                inputs,
                input_names,
                output_names,
                cotangent,
            ):
                self.vjp_biases.append(inputs["bias_voltage"])
                return {"doping": np.zeros_like(inputs["doping"], dtype=float)}

        class FakeGyptis:
            def __init__(self):
                self.vjp_calls = 0

            def apply(self, inputs):
                return {"neff_sq": float(np.mean(inputs["design_epsilon"]))}

            def vector_jacobian_product(
                self,
                inputs,
                input_names,
                output_names,
                cotangent,
            ):
                self.vjp_calls += 1
                design_epsilon = np.asarray(inputs["design_epsilon"], dtype=float)
                return {
                    "design_epsilon": np.full(
                        design_epsilon.shape,
                        cotangent["neff_sq"] / len(design_epsilon),
                    )
                }

        ct = FakeChargeTransport()
        gyptis = FakeGyptis()
        perturbed, background = build_gyptis_components(container=gyptis)
        components = PipelineComponents(
            chargetransport=build_chargetransport_component(container=ct),
            gyptis=perturbed,
            gyptis_background=background,
        )

        _, history = optimize_doping(
            initial_rho=rho_initial,
            max_iter=3,
            use_jit=False,
            components=components,
        )

        assert ct.forward_biases.count(0.0) == len(history)
        assert ct.forward_biases.count(-5.0) == len(history)
        assert ct.vjp_biases.count(0.0) == len(history)
        assert ct.vjp_biases.count(-5.0) == len(history)
        assert gyptis.vjp_calls == len(history)


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
            initial_rho=rho0,
            H=H_dense,
            max_iter=3,
            components=_stub_components(),
        )
        assert rho_opt.shape == rho0.shape
        assert len(history) > 0

    def test_box_bounds_preserved_with_filter(self, rho0, H_dense):
        rho_opt, _ = optimize_doping(
            initial_rho=rho0,
            H=H_dense,
            max_iter=5,
            components=_stub_components(),
        )
        assert np.all(rho_opt >= 0.0)
        assert np.all(rho_opt <= 1.0)

    def test_builds_filter_from_coords(self, rho0, coords):
        rho_opt, _ = optimize_doping(
            initial_rho=rho0,
            mesh_coords=coords,
            r_min=50e-9,
            max_iter=2,
            components=_stub_components(),
        )
        assert rho_opt.shape == rho0.shape


class TestOptimizeDopingThreadsMeshRef:
    """The optimizer must forward the shared mesh to ChargeTransport.

    Without it CT never receives the real geometry and solves on its 1D
    fallback -- the crash the whole feature exists to fix. Ref:
    .scratch/chargetransport-mesh-node-ordering/issues/02.
    """

    def test_mesh_ref_reaches_chargetransport_component(self):
        from prismo.pipeline import (
            _CT_MESH_MOUNT,
            PipelineComponents,
            build_chargetransport_component,
            build_gyptis_components,
        )
        from prismo_shared.schemas import MeshRef

        class RecordingChargeTransport:
            def __init__(self):
                self.mesh_paths: list[str | None] = []

            def apply(self, inputs):
                ref = inputs.get("mesh_ref")
                self.mesh_paths.append(None if ref is None else ref["path"])
                doping = np.asarray(inputs["doping"], dtype=float)
                return {"electrons": doping, "holes": doping}

            def vector_jacobian_product(self, inputs, *_args):
                return {"doping": np.zeros_like(inputs["doping"], dtype=float)}

        class FakeGyptis:
            def apply(self, inputs):
                return {"neff_sq": float(np.mean(inputs["design_epsilon"]))}

            def vector_jacobian_product(self, inputs, *_args, cotangent=None):
                design_epsilon = np.asarray(inputs["design_epsilon"], dtype=float)
                return {"design_epsilon": np.zeros_like(design_epsilon)}

        ct = RecordingChargeTransport()
        perturbed, background = build_gyptis_components(container=FakeGyptis())
        components = PipelineComponents(
            chargetransport=build_chargetransport_component(container=ct),
            gyptis=perturbed,
            gyptis_background=background,
        )

        mesh_ref = MeshRef(path="/host/outputs/waveguide.msh", n_nodes=4)
        optimize_doping(
            initial_rho=np.full(4, 0.25, dtype=float),
            max_iter=1,
            use_jit=False,
            mesh_ref=mesh_ref,
            components=components,
        )

        # Every CT forward received the mesh (rewritten to the container mount),
        # never None.
        assert ct.mesh_paths, "ChargeTransport was never called"
        assert all(p == f"{_CT_MESH_MOUNT}/waveguide.msh" for p in ct.mesh_paths)


class TestOptimizeDopingSurvivesSolverFailure:
    """A failed physics solve is "step too large": halve the move limit, retry.

    ChargeTransport's Newton solve diverges on some designs MMA proposes. NLopt
    cannot be told to reject a trial point, and feeding MMA a fabricated penalty
    poisons its asymptote update, so the optimizer abandons that MMA instance,
    halves the move limit and re-proposes from the same iterate (ticket 19). It
    stops only after a bounded number of halvings, keeping the best feasible
    design.
    """

    N_NODES = 4

    def test_retries_with_a_smaller_step_after_a_failed_solve(self):
        target = np.full(self.N_NODES, 0.8, dtype=float)
        target_j = jnp.asarray(target)
        calls = {"n": 0}
        # ``on_iteration`` sees the concrete candidate; the pipeline mock only
        # ever sees a JAX tracer, so candidates must be captured here.
        candidates: list[np.ndarray] = []

        def mock_pipeline(rho, **kwargs):
            calls["n"] += 1
            if calls["n"] == 4:
                raise RuntimeError("Julia forward solve failed: ConvergenceError()")
            return -jnp.sum((rho - target_j) ** 2)

        with mock.patch("prismo.optimizer.pipeline_with_terms", side_effect=_with_terms(mock_pipeline)):
            rho_opt, history = optimize_doping(
                initial_rho=np.full(self.N_NODES, 0.25, dtype=float),
                max_iter=60,
                ftol_rel=1e-8,
                # Count real evaluations: under JIT the mock is traced once and
                # the raise would never reach the optimizer at run time.
                use_jit=False,
                on_iteration=lambda i, r: candidates.append(r),
            )

        assert calls["n"] > 4, "run did not continue past the failed solve"
        # The failed trial is not a physics evaluation, so it stays out of
        # history -- main.py audits every entry there for a valid signal.
        assert len(history) == calls["n"] - 1
        # The retry from the same iterate took a smaller step than the failure.
        failed = candidates[3]
        before = candidates[2]
        retry = candidates[4]
        assert np.max(np.abs(retry - before)) < np.max(np.abs(failed - before))
        # And the run still reaches the optimum.
        np.testing.assert_allclose(rho_opt, target, rtol=0.05)

    def test_stops_after_bounded_halvings_keeping_the_best_design(self):
        """Every step after the seed fails: bounded retries, seed returned."""
        calls = {"n": 0}

        def mock_pipeline(rho, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("Julia forward solve failed: ConvergenceError()")
            return jnp.asarray(1.0, dtype=jnp.float64) - jnp.sum(rho**2) * 0.0

        seed = np.full(self.N_NODES, 0.25, dtype=float)
        with mock.patch("prismo.optimizer.pipeline_with_terms", side_effect=_with_terms(mock_pipeline)):
            rho_opt, history = optimize_doping(
                initial_rho=seed,
                max_iter=100,
                use_jit=False,
                max_move_halvings=4,
            )

        # Seed + at most one failed trial per halving level (4 halvings + the
        # first attempt), then stop.
        assert calls["n"] <= 1 + 4 + 1
        assert len(history) == 1
        np.testing.assert_allclose(rho_opt, seed)

    def test_failure_on_the_seed_still_raises(self):
        def mock_pipeline(rho, **kwargs):
            raise RuntimeError("Julia forward solve failed: ConvergenceError()")

        with (
            mock.patch("prismo.optimizer.pipeline_with_terms", side_effect=_with_terms(mock_pipeline)),
            pytest.raises(RuntimeError, match="ConvergenceError"),
        ):
            optimize_doping(
                initial_rho=np.full(self.N_NODES, 0.25, dtype=float),
                max_iter=10,
                use_jit=False,
            )

    def test_stalled_design_stops_instead_of_burning_evaluations(self):
        """A design MMA re-proposes unchanged must not cost solves forever."""
        calls = {"n": 0}

        def mock_pipeline(rho, **kwargs):
            calls["n"] += 1
            # Flat objective: every trial point is identical to the last, which
            # is the degenerate loop observed against the real containers.
            return jnp.asarray(1.0, dtype=jnp.float64)

        with mock.patch("prismo.optimizer.pipeline_with_terms", side_effect=_with_terms(mock_pipeline)):
            rho_opt, _ = optimize_doping(
                initial_rho=np.full(self.N_NODES, 0.25, dtype=float),
                max_iter=500,
                ftol_rel=0.0,
                use_jit=False,
            )

        assert calls["n"] < 20, f"stall guard did not fire ({calls['n']} solves)"
        assert rho_opt.shape == (self.N_NODES,)


class TestMoveLimit:
    """No iteration moves any design variable by more than the move limit."""

    N_NODES = 6

    @staticmethod
    def _quadratic(target: np.ndarray):
        target_j = jnp.asarray(target)

        def mock_pipeline(rho, **kwargs):
            return -jnp.sum((rho - target_j) ** 2)

        return mock_pipeline

    def test_every_step_respects_the_move_limit(self):
        target = np.array([0.9, -0.9, 0.5, -0.5, 0.0, 0.25])
        candidates: list[np.ndarray] = []
        move_limit = 0.07

        with mock.patch(
            "prismo.optimizer.pipeline_with_terms", side_effect=_with_terms(self._quadratic(target))
        ):
            rho_opt, history = optimize_doping(
                initial_rho=np.zeros(self.N_NODES),
                max_iter=80,
                ftol_rel=1e-10,
                move_limit=move_limit,
                use_jit=False,
                on_iteration=lambda i, r: candidates.append(r),
            )

        # Every evaluated candidate is within the box of the iterate it was
        # proposed from; ``max_step`` records exactly that distance.
        assert all(h["max_step"] <= move_limit + 1e-12 for h in history)
        assert all(h["move_limit"] <= move_limit for h in history)
        # Consecutive candidates can differ by at most two boxes (a rejected
        # trial and the next proposal from the same iterate).
        steps = [np.max(np.abs(b - a)) for a, b in pairwise(candidates)]
        assert max(steps) <= 2 * move_limit + 1e-12
        # The limit slows but does not stop progress to the optimum.
        np.testing.assert_allclose(rho_opt, target, atol=0.03)
        # It actually binds: the seed is > 0.5 from most targets, so the first
        # accepted step is a full box for the steepest variables.
        assert history[1]["max_step"] > 0.5 * move_limit

    def test_move_limit_must_be_positive(self):
        with pytest.raises(ValueError, match="move_limit"):
            optimize_doping(
                initial_rho=np.zeros(self.N_NODES),
                move_limit=0.0,
                components=_stub_components(),
            )

    def test_bounds_still_hold_at_the_rails(self):
        target = np.full(self.N_NODES, 3.0)
        with mock.patch(
            "prismo.optimizer.pipeline_with_terms", side_effect=_with_terms(self._quadratic(target))
        ):
            rho_opt, _ = optimize_doping(
                initial_rho=np.full(self.N_NODES, 0.9),
                max_iter=20,
                move_limit=0.1,
                use_jit=False,
            )
        assert np.all(rho_opt <= 1.0)
        np.testing.assert_allclose(rho_opt, 1.0, atol=1e-6)


class TestCheckpoint:
    """``checkpoint.json`` (theta + history) is written after every evaluation."""

    N_NODES = 4

    def test_checkpoint_written_after_every_evaluation(self, tmp_path):
        target = jnp.full(self.N_NODES, 0.8, dtype=jnp.float64)
        snapshots: list[int] = []
        path = tmp_path / "outputs" / "checkpoint.json"

        def mock_pipeline(rho, **kwargs):
            return -jnp.sum((rho - target) ** 2)

        def on_iteration(i, rho):
            # Before evaluation ``i`` the checkpoint holds ``i - 1`` entries.
            snapshots.append(
                len(json.loads(path.read_text())["history"]) if path.exists() else 0
            )

        with mock.patch("prismo.optimizer.pipeline_with_terms", side_effect=_with_terms(mock_pipeline)):
            rho_opt, history = optimize_doping(
                initial_rho=np.full(self.N_NODES, 0.25, dtype=float),
                max_iter=6,
                use_jit=False,
                checkpoint_path=path,
                on_iteration=on_iteration,
            )

        assert snapshots == list(range(len(history)))
        saved = json.loads(path.read_text())
        assert len(saved["history"]) == len(history)
        np.testing.assert_allclose(saved["rho_opt"], rho_opt)
        assert saved["move_limit"] > 0.0

    def test_no_checkpoint_without_a_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        optimize_doping(
            initial_rho=np.full(self.N_NODES, 0.25, dtype=float),
            max_iter=2,
            components=_stub_components(),
        )
        assert not (tmp_path / "outputs").exists()


class TestLossAwareObjective:
    """Ticket 25: the optimizer drives ``delta_neff - w * loss`` and records both."""

    N = 6

    @staticmethod
    def _components():
        def ct(doping, bias_voltage, mesh_ref=None):
            # Carriers follow |doping|; reverse bias depletes a fixed fraction.
            n = jnp.abs(doping) + 1e15
            p = 0.1 * n
            if bias_voltage != 0.0:
                return 0.5 * n, 0.5 * p
            return n, p

        return _stub_components(chargetransport=ct)

    def test_history_records_objective_delta_neff_and_loss(self, tmp_path):
        rho0 = np.full(self.N, 0.3)
        overlap = np.full(self.N, 1.0 / self.N)
        w = 1e-6
        _, history = optimize_doping(
            rho0,
            max_iter=3,
            components=self._components(),
            loss_weight=w,
            mode_overlap=overlap,
            checkpoint_path=tmp_path / "checkpoint.json",
        )
        for entry in history:
            assert np.isfinite(entry["modal_loss_db_cm"])
            assert entry["modal_loss_db_cm"] > 0.0
            assert entry["objective"] == pytest.approx(
                entry["delta_n_eff"] - w * entry["modal_loss_db_cm"]
            )
        # The checkpoint carries the same keys and stays valid JSON.
        saved = json.loads((tmp_path / "checkpoint.json").read_text())
        assert saved["history"][0]["modal_loss_db_cm"] == pytest.approx(
            history[0]["modal_loss_db_cm"]
        )

    def test_without_overlap_the_loss_is_unreported_and_objective_is_delta_neff(
        self, tmp_path
    ):
        rho0 = np.full(self.N, 0.3)
        _, history = optimize_doping(
            rho0,
            max_iter=2,
            components=self._components(),
            checkpoint_path=tmp_path / "checkpoint.json",
        )
        for entry in history:
            assert entry["modal_loss_db_cm"] is None
            assert entry["objective"] == entry["delta_n_eff"]
        saved = json.loads((tmp_path / "checkpoint.json").read_text())
        assert saved["history"][0]["modal_loss_db_cm"] is None

    def test_loss_penalty_changes_the_accepted_steps(self):
        """A large loss weight must pull the design away from the pure-Δneff path."""
        rho0 = np.full(self.N, 0.3)
        overlap = np.full(self.N, 1.0 / self.N)
        rho_free, _ = optimize_doping(
            rho0, max_iter=4, components=self._components(), move_limit=0.1
        )
        rho_pen, history = optimize_doping(
            rho0,
            max_iter=4,
            components=self._components(),
            move_limit=0.1,
            loss_weight=1.0,  # dominant: every step is judged on loss
            mode_overlap=overlap,
        )
        assert not np.allclose(rho_free, rho_pen)
        # Accepted steps improve the *objective*, i.e. lower the loss here.
        objectives = [h["objective"] for h in history]
        assert max(objectives) >= objectives[0]


class TestDesignHistoryCheckpoint:
    """Every history record carries the design it evaluated (replayable run)."""

    N = 4

    def test_each_record_carries_its_design(self, tmp_path):
        target = jnp.full(self.N, 0.8)

        def mock_pipeline(rho, **kwargs):
            return -jnp.sum((rho - target) ** 2)

        with mock.patch(
            "prismo.optimizer.pipeline_with_terms", side_effect=_with_terms(mock_pipeline)
        ):
            _, history = optimize_doping(
                initial_rho=np.full(self.N, 0.25),
                max_iter=4,
                use_jit=False,
                checkpoint_path=tmp_path / "checkpoint.json",
            )
        np.testing.assert_allclose(history[0]["design"], np.full(self.N, 0.25))
        saved = json.loads((tmp_path / "checkpoint.json").read_text())
        for entry, saved_entry in zip(history, saved["history"], strict=True):
            assert len(saved_entry["design"]) == self.N
            np.testing.assert_allclose(saved_entry["design"], entry["design"])

    def test_failed_evaluations_write_no_record(self, tmp_path):
        target = jnp.full(self.N, 0.8)
        calls = {"n": 0}

        def mock_pipeline(rho, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("Julia forward solve failed")
            return -jnp.sum((rho - target) ** 2)

        with mock.patch(
            "prismo.optimizer.pipeline_with_terms", side_effect=_with_terms(mock_pipeline)
        ):
            _, history = optimize_doping(
                initial_rho=np.full(self.N, 0.25),
                max_iter=5,
                use_jit=False,
                checkpoint_path=tmp_path / "checkpoint.json",
            )
        assert calls["n"] > 3
        assert len(history) == calls["n"] - 1
        assert all("design" in entry for entry in history)
