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
    generate_outputs,
    plot_convergence,
    plot_delta_neff_breakdown,
    plot_doping_field,
    plot_gradient_validation,
    plot_live_doping_field,
    validate_gradient,
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


class TestBreakdownPlot:
    def test_generates_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_delta_neff_breakdown(0.001, 0.0001, output_dir=tmp)
            assert Path(path).exists()
            assert Path(path).suffix == ".pdf"

    def test_zero_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_delta_neff_breakdown(0.0, 0.0, output_dir=tmp)
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


class TestGenerateOutputs:
    def test_includes_breakdown_plot(self):
        coords = _make_coords()
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_outputs(
                np.full(N_NODES, 0.25), RNG.random(N_NODES), _make_history(),
                coords, output_dir=tmp,
            )
            assert {"convergence.pdf", "doping_field.pdf", "breakdown.pdf"} <= {
                path.name for path in paths
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
