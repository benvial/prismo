"""Tests for paper-ready output plotting.

Ref: ticket 16.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from _doubles import stub_components
from prismo.outputs import (
    GradientValidationResult,
    ModeField,
    generate_outputs,
    plot_convergence,
    plot_doping_field,
    plot_gradient_validation,
    plot_live_doping_field,
    plot_mode_field,
    validate_gradient,
    vpi_axis_is_symlog,
)
from prismo.pipeline import pipeline

RNG = np.random.default_rng(42)

N_NODES = 16


def _stub_pipeline(theta):
    """The pipeline driven by physics-free component doubles (ticket 04).

    The implicit no-backend stubs were deleted, so plotting tests that drive the
    real ``pipeline`` supply explicit doubles through the ``components=`` seam.
    """
    return pipeline(theta, components=stub_components())


def _make_history() -> list[dict]:
    return [
        {
            "iteration": i + 1,
            "delta_n_eff": 0.001 * (1.0 - 0.5 ** i),
            "delta_rho": 0.01 / (i + 1),
            "grad_norm": 0.001 / (i + 1),
            "wall_time": i * 0.5,
        }
        for i in range(10)
    ]


def _make_coords(n_nodes: int = N_NODES) -> np.ndarray:
    xs, ys = np.meshgrid(
        np.arange(int(np.sqrt(n_nodes))) * 20e-9,
        np.arange(int(np.sqrt(n_nodes))) * 20e-9,
        indexing="xy",
    )
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


class TestConvergencePlot:
    def test_generates_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = _make_history()
            path = plot_convergence(history, output_dir=tmp)
            assert Path(path).exists()
            assert Path(path).suffix == ".pdf"

    def test_empty_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_convergence([], output_dir=tmp)
            assert Path(path).exists()

    def test_survives_a_zero_objective_iteration(self):
        """VpiLpi diverges at delta_n_eff = 0; the curve must still render."""
        history = _make_history()
        history[0]["delta_n_eff"] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            assert plot_convergence(history, output_dir=tmp).exists()

    def test_survives_a_near_zero_objective_iteration(self):
        """A near-zero start puts VpiLpi orders of magnitude off the converged value."""
        history = _make_history()
        history[0]["delta_n_eff"] = 1e-12
        with tempfile.TemporaryDirectory() as tmp:
            assert plot_convergence(history, output_dir=tmp).exists()

    def test_vpi_axis_is_linear_over_a_narrow_span(self):
        """A run that starts converged spans little; symlog would label no ticks."""
        assert not vpi_axis_is_symlog([3.57, 3.59])

    def test_vpi_axis_is_symlog_over_a_wide_span(self):
        """A run starting near zero spans decades; linear would flatten it."""
        assert vpi_axis_is_symlog([3.57, 3.9e5])

    def test_survives_a_sign_change_in_the_objective(self):
        """VpiLpi carries the objective's sign, so the axis must span both."""
        history = _make_history()
        history[0]["delta_n_eff"] = -1e-4
        with tempfile.TemporaryDirectory() as tmp:
            assert plot_convergence(history, output_dir=tmp).exists()


class TestDopingFieldPlot:
    def test_generates_pdf(self):
        coords = _make_coords()
        rho_initial = np.full(N_NODES, 0.25)
        rho_opt = RNG.random(N_NODES)
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_doping_field(rho_initial, rho_opt, coords, output_dir=tmp)
            assert Path(path).exists()
            assert Path(path).suffix == ".pdf"

    def test_live_plot_replaces_previous_image(self):
        coords = _make_coords()
        doping = np.linspace(-1e18, 1e18, N_NODES)
        with tempfile.TemporaryDirectory() as tmp:
            first = plot_live_doping_field(doping, coords, iteration=1, output_dir=tmp)
            second = plot_live_doping_field(doping * 0.5, coords, iteration=2, output_dir=tmp)
            assert first == second
            assert first.name == "doping_field_live.png"
            assert first.exists()

    def test_live_plot_named_per_iteration(self):
        coords = _make_coords()
        doping = np.linspace(-1e18, 1e18, N_NODES)
        with tempfile.TemporaryDirectory() as tmp:
            first = plot_live_doping_field(
                doping, coords, iteration=1, output_dir=tmp, name="doping_field_1"
            )
            second = plot_live_doping_field(
                doping * 0.5, coords, iteration=2, output_dir=tmp, name="doping_field_2"
            )
            assert first != second
            assert first.exists() and second.exists()
            # No leftover temp files: each snapshot is written straight to its
            # final path.
            assert sorted(q.name for q in Path(tmp).iterdir()) == [
                "doping_field_1.png",
                "doping_field_2.png",
            ]

    def test_with_geometry(self):
        from prismo.waveguide_mesh import RibWaveguideGeometry

        coords = _make_coords()
        rho_initial = np.full(N_NODES, 0.25)
        rho_opt = RNG.random(N_NODES)
        geometry = RibWaveguideGeometry()
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_doping_field(
                rho_initial, rho_opt, coords, geometry=geometry, output_dir=tmp,
            )
            assert Path(path).exists()


class TestGradientValidationPlot:
    def test_generates_pdf(self):
        import jax.numpy as jnp

        rho = jnp.asarray(RNG.random(N_NODES), dtype=jnp.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_gradient_validation(
                _stub_pipeline, rho, n_directions=2, output_dir=tmp
            )
            assert Path(path).exists()
            assert Path(path).suffix == ".pdf"

    def test_keeps_finite_differences_inside_design_bounds(self):
        import jax
        import jax.numpy as jnp

        received: list[np.ndarray] = []

        def bounded_pipeline(theta):
            if not isinstance(theta, jax.core.Tracer):
                received.append(np.asarray(theta))
            return jnp.sum(theta**2)

        # A node near the signed lower bound -1 must not be stepped below it.
        theta = jnp.asarray([-0.999, 0.5])
        direction = jnp.asarray([-1.0, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            plot_gradient_validation(
                bounded_pipeline,
                theta,
                directions=[direction],
                step_sizes=np.asarray([1e-4, 1e-3, 1e-2]),
                output_dir=tmp,
            )

        assert all(
            np.all(values >= -1.0) and np.all(values <= 1.0) for values in received
        )

    def test_keeps_both_stencil_sides_inside_bounds(self):
        import jax
        import jax.numpy as jnp

        received: list[np.ndarray] = []

        def bounded_pipeline(theta):
            if not isinstance(theta, jax.core.Tracer):
                received.append(np.asarray(theta))
            return jnp.sum(theta**2)

        # A node near the signed LOWER bound stepped along a POSITIVE direction:
        # the +h sample stays well inside, but the -h sample is the binding one
        # (rho - h < -1). The feasibility bound must clamp both stencil sides.
        theta = jnp.asarray([-0.9, 0.0])
        direction = jnp.asarray([1.0, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            plot_gradient_validation(
                bounded_pipeline,
                theta,
                directions=[direction],
                # 0.5 would push rho - h*d = -1.4 out of the box under a
                # one-sided (upper-only) bound.
                step_sizes=np.asarray([0.05, 0.5]),
                output_dir=tmp,
            )

        assert received, "expected at least one feasible finite-difference sample"
        assert all(
            np.all(values >= -1.0) and np.all(values <= 1.0) for values in received
        )

    def test_projects_directions_off_railed_nodes(self):
        import jax.numpy as jnp

        # An optimized design rails nodes at the ±1 bounds, so a dense random
        # direction leaves the central stencil no feasible step at all. The
        # validation must fall back to the interior subspace (the unpinned
        # node here) rather than raising after a finished optimization.
        theta = jnp.asarray([-1.0, 0.2])
        direction = jnp.asarray([np.sqrt(0.5), np.sqrt(0.5)])
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_gradient_validation(
                lambda theta: jnp.sum(theta**2),
                theta,
                directions=[direction],
                step_sizes=np.asarray([1e-3, 1e-2]),
                output_dir=tmp,
            )
            assert Path(path).exists()

    def test_rejects_gradient_validation_without_feasible_steps(self):
        import jax.numpy as jnp

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="no feasible"):
                plot_gradient_validation(
                    lambda theta: jnp.sum(theta**2),
                    # A node pinned at the signed lower bound -1 with a
                    # downward direction leaves no feasible step.
                    jnp.asarray([-1.0, 0.5]),
                    directions=[jnp.asarray([-1.0, 0.0])],
                    output_dir=tmp,
                )


class TestValidateGradient:
    def test_passes_tolerance_for_real_gradient(self):
        import jax.numpy as jnp

        rho = jnp.asarray(RNG.uniform(-0.5, 0.5, N_NODES), dtype=jnp.float64)
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_gradient(
                _stub_pipeline, rho, n_directions=3, tolerance=1e-2, output_dir=tmp
            )
            assert isinstance(result, GradientValidationResult)
            # A real adjoint vs. central FD must agree to well under the tolerance.
            assert result.passed is True
            assert result.worst_rel_error <= 1e-2
            assert result.n_directions == 3
            assert len(result.best_rel_errors) == 3
            assert result.worst_rel_error == max(result.best_rel_errors)
            assert result.tolerance == 1e-2
            assert Path(result.figure_path).exists()
            assert Path(result.figure_path).name == "gradient_validation.pdf"

    def test_fails_when_tolerance_unreachable(self):
        import jax.numpy as jnp

        # A genuinely nonlinear scalar so central FD carries a nonzero O(h^2)
        # truncation error at every step -- no step reaches a relative error
        # this small, so the gate must report a failure without raising (the
        # figure is still the deliverable).
        def nonlinear(theta):
            return jnp.sum(jnp.sin(7.0 * theta))

        rho = jnp.asarray(RNG.uniform(-0.5, 0.5, N_NODES), dtype=jnp.float64)
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_gradient(
                nonlinear, rho, n_directions=1, tolerance=1e-30, output_dir=tmp
            )
            assert result.passed is False
            assert result.worst_rel_error > 1e-30
            assert Path(result.figure_path).exists()

    def test_reports_best_error_per_direction(self):
        import jax.numpy as jnp

        rho = jnp.asarray(RNG.uniform(-0.5, 0.5, N_NODES), dtype=jnp.float64)
        d1 = jnp.asarray(RNG.standard_normal(N_NODES), dtype=jnp.float64)
        d1 = d1 / jnp.linalg.norm(d1)
        d2 = jnp.asarray(RNG.standard_normal(N_NODES), dtype=jnp.float64)
        d2 = d2 / jnp.linalg.norm(d2)
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_gradient(
                _stub_pipeline,
                rho,
                directions=[d1, d2],
                tolerance=1e-2,
                output_dir=tmp,
            )
        assert result.n_directions == 2
        assert len(result.best_rel_errors) == 2
        assert all(e >= 0.0 for e in result.best_rel_errors)


class TestModeFieldPlot:
    @staticmethod
    def _mode(**overrides) -> ModeField:
        xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 12), np.linspace(-0.5, 0.5, 9))
        coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
        abs_e = np.exp(-(coords[:, 0] ** 2 + coords[:, 1] ** 2) / 0.05)
        fields = dict(abs_e=abs_e / abs_e.max(), coords_um=coords)
        fields.update(overrides)
        return ModeField(**fields)

    def test_generates_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_mode_field(self._mode(), output_dir=tmp)
            assert Path(path).exists()
            assert path.name == "mode_field.pdf"

    def test_draws_rib_outline(self):
        mode = self._mode(rib_bounds=(-0.25, 0.25, -0.06, 0.16))
        with tempfile.TemporaryDirectory() as tmp:
            assert plot_mode_field(mode, output_dir=tmp).exists()

    def test_labels_a_higher_order_mode(self):
        mode = self._mode(mode_index=1)
        with tempfile.TemporaryDirectory() as tmp:
            assert plot_mode_field(mode, output_dir=tmp).exists()

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="one magnitude per vertex"):
            ModeField(abs_e=np.ones(4), coords_um=np.zeros((3, 2)))


class TestGenerateOutputs:
    def test_emits_the_headline_figures(self):
        """The four headline figures, and no retired panel (ticket 07)."""
        import jax.numpy as jnp

        coords = _make_coords()
        mode = ModeField(
            abs_e=np.linspace(0.0, 1.0, N_NODES), coords_um=coords * 1e6,
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_outputs(
                np.full(N_NODES, 0.25), RNG.random(N_NODES), _make_history(),
                coords,
                pipeline_fn=_stub_pipeline,
                gradient_validation_rho=jnp.full(N_NODES, 0.25),
                mode_field=mode,
                output_dir=tmp,
            )
            assert {path.name for path in paths} == {
                "convergence.pdf",
                "doping_field.pdf",
                "gradient_validation.pdf",
                "mode_field.pdf",
            }

    def test_skips_mode_figure_without_a_mode(self):
        coords = _make_coords()
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_outputs(
                np.full(N_NODES, 0.25), RNG.random(N_NODES), _make_history(),
                coords, output_dir=tmp,
            )
            assert {path.name for path in paths} == {
                "convergence.pdf",
                "doping_field.pdf",
            }

    def test_custom_directions(self):
        import jax.numpy as jnp

        rho = jnp.asarray(RNG.random(N_NODES), dtype=jnp.float64)
        d1 = jnp.asarray(RNG.standard_normal(N_NODES), dtype=jnp.float64)
        d1 = d1 / jnp.linalg.norm(d1)
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_gradient_validation(
                _stub_pipeline, rho, directions=[d1], output_dir=tmp,
            )
            assert Path(path).exists()


class TestColdReevaluation:
    """Warm/cold Δneff comparison and its convergence-figure annotation (ticket 20)."""

    def test_discrepancy_and_verdict(self):
        from prismo.outputs import ColdReevaluation

        same = ColdReevaluation(warm_delta_neff=3e-4, cold_delta_neff=3e-4 * (1 + 1e-9))
        assert same.passed
        assert same.rel_discrepancy < 1e-8
        off = ColdReevaluation(warm_delta_neff=2.99e-4, cold_delta_neff=6.5e-5)
        assert not off.passed
        assert off.rel_discrepancy == pytest.approx(abs(6.5e-5 - 2.99e-4) / 2.99e-4)
        zero = ColdReevaluation(warm_delta_neff=0.0, cold_delta_neff=0.0)
        assert zero.rel_discrepancy == 0.0

    def test_convergence_figure_accepts_the_annotation(self, tmp_path):
        from prismo.outputs import ColdReevaluation, plot_convergence

        history = [
            {"iteration": i, "delta_n_eff": 1e-4 * i, "delta_rho": 0.0,
             "grad_norm": 1.0, "wall_time": float(i)}
            for i in range(1, 4)
        ]
        for cold in (
            ColdReevaluation(warm_delta_neff=3e-4, cold_delta_neff=3e-4),
            ColdReevaluation(warm_delta_neff=3e-4, cold_delta_neff=6.5e-5),
        ):
            path = plot_convergence(history, output_dir=tmp_path, cold_reevaluation=cold)
            assert path.exists()


import jax.numpy as jnp  # noqa: E402


class TestGradientValidationColdHook:
    """``before_evaluation`` runs before the adjoint and every FD sample."""

    def test_hook_called_before_each_evaluation(self, tmp_path):
        from prismo.outputs import validate_gradient

        calls: list[str] = []

        def pipeline_fn(rho):
            calls.append("eval")
            return jnp.sum(rho**2)

        def before():
            calls.append("reset")

        steps = np.asarray([1e-3, 1e-2, 1e-1])
        validate_gradient(
            pipeline_fn,
            jnp.asarray([0.1, 0.2, 0.3]),
            n_directions=2,
            step_sizes=steps,
            output_dir=tmp_path,
            before_evaluation=before,
        )
        # Every evaluation is preceded by a reset, and nothing else happens.
        assert calls[0] == "reset"
        assert calls.count("reset") == calls.count("eval")
        assert all(a == "reset" for a, b in zip(calls[::2], calls[1::2], strict=True))


class TestObjectiveLineScan:
    """Ticket 23: f(rho + t*d) along one direction at uniform spacing."""

    @staticmethod
    def _offsets(n: int = 11, spacing: float = 1e-3) -> np.ndarray:
        half = n // 2
        return spacing * np.arange(-half, half + 1, dtype=float)

    def test_exact_quadratic_has_zero_residual_and_matching_slope(self):
        import jax.numpy as jnp
        from prismo.outputs import scan_objective_line

        rho = jnp.linspace(-0.5, 0.5, N_NODES)
        d = jnp.ones(N_NODES) / np.sqrt(N_NODES)
        a = jnp.arange(N_NODES, dtype=float) / N_NODES

        def f(x):
            return jnp.sum(x**2) + jnp.dot(a, x)

        scan = scan_objective_line(f, rho, d, self._offsets())
        assert scan.values.shape == (11,)
        assert scan.rms_rel_residual < 1e-10
        assert scan.max_rel_residual < 1e-10
        assert scan.noise_estimate < 1e-10
        assert scan.fitted_slope == pytest.approx(scan.adjoint_slope, rel=1e-8)
        assert scan.figure_path is None
        assert scan.json_path is None

    def test_noisy_objective_reports_the_floor(self):
        import jax.numpy as jnp
        from prismo.outputs import scan_objective_line

        rho = jnp.zeros(N_NODES)
        d = jnp.ones(N_NODES) / np.sqrt(N_NODES)
        amplitude = 1e-3
        rng = np.random.default_rng(7)
        noise = iter(amplitude * rng.choice([-1.0, 1.0], size=64))

        def f(x):
            # Deterministic per call but uncorrelated between neighbours.
            return 1.0 + jnp.sum(x) + next(noise)

        scan = scan_objective_line(f, rho, d, self._offsets(n=21))
        assert 0.2 * amplitude < scan.rms_rel_residual < 2.0 * amplitude
        assert scan.max_rel_residual >= scan.rms_rel_residual
        assert 0.2 * amplitude < scan.noise_estimate < 2.0 * amplitude

    def test_writes_figure_and_json_when_output_dir_given(self, tmp_path):
        import json

        import jax.numpy as jnp
        from prismo.outputs import scan_objective_line

        rho = jnp.zeros(4)
        d = jnp.asarray([1.0, 0.0, 0.0, 0.0])
        scan = scan_objective_line(
            lambda x: jnp.sum(x**2), rho, d, self._offsets(n=5), output_dir=tmp_path
        )
        assert scan.figure_path == tmp_path / "objective_line_scan.pdf"
        assert scan.json_path == tmp_path / "objective_line_scan.json"
        assert scan.figure_path.exists()
        payload = json.loads(scan.json_path.read_text())
        assert len(payload["offsets"]) == 5
        assert len(payload["values"]) == 5
        assert payload["rms_rel_residual"] == pytest.approx(scan.rms_rel_residual)

    def test_before_evaluation_runs_before_gradient_and_every_sample(self):
        import jax.numpy as jnp
        from prismo.outputs import scan_objective_line

        calls: list[str] = []
        rho = jnp.zeros(3)
        d = jnp.asarray([0.0, 1.0, 0.0])
        offsets = self._offsets(n=7)
        scan_objective_line(
            lambda x: jnp.sum(x),
            rho,
            d,
            offsets,
            before_evaluation=lambda: calls.append("reset"),
        )
        assert len(calls) == 1 + len(offsets)

    def test_feasible_offsets_keep_the_box(self):
        import jax.numpy as jnp
        from prismo.outputs import feasible_offsets

        rho = jnp.asarray([0.999, 0.0, -0.5])
        d = jnp.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        offsets = self._offsets(n=11, spacing=1e-3)
        kept = feasible_offsets(rho, d, offsets)
        # +t*d_0 <= 1 - 0.999 -> t <= 1e-3*sqrt(3) ~ 1.73e-3; -t side is free.
        assert kept.min() == offsets.min()
        assert kept.max() == pytest.approx(1e-3)
        assert np.all(np.abs(np.asarray(rho)[None, :] + kept[:, None] * np.asarray(d)) <= 1.0)

    def test_probe_direction_returns_the_gradient_it_used(self):
        import jax.numpy as jnp
        from prismo.outputs import probe_direction, scan_objective_line

        rho = jnp.asarray([0.2, -0.1, 0.0, 0.999])
        a = jnp.asarray([1.0, 2.0, 3.0, 4.0])
        calls: list[int] = []

        def f(x):
            calls.append(1)
            return jnp.dot(a, x)

        d, grad = probe_direction(f, rho, "gradient")
        assert grad is not None
        # Projected off the rail-pinned last variable and unit length.
        assert float(d[3]) == 0.0
        assert float(jnp.linalg.norm(d)) == pytest.approx(1.0)
        n_calls_after_direction = len(calls)
        scan = scan_objective_line(f, rho, d, self._offsets(n=5), gradient=grad)
        # Only the 5 samples ran: no second adjoint.
        assert len(calls) == n_calls_after_direction + 5
        assert scan.adjoint_slope == pytest.approx(float(jnp.dot(a, d)))
        d_random, grad_random = probe_direction(f, rho, "random")
        assert grad_random is None
        assert float(jnp.linalg.norm(d_random)) == pytest.approx(1.0)
        with pytest.raises(ValueError, match="gradient"):
            probe_direction(f, rho, "sideways")
