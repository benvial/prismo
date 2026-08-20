"""Tests for the main application module."""

import types
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

runner = CliRunner()


def _container_components(centroids: np.ndarray):
    """A minimal bundle exposing the design-cell geometry seam.

    The container pipeline reads design-cell vertices through
    ``PipelineComponents.design_cell_vertices`` to build the mesh-transfer
    operator; the solve components are unused when ``optimize_doping`` is
    stubbed, so they can be ``None``. Vertices are faked (the transfer build is
    stubbed in these tests), one degenerate triangle per centroid.
    """
    from prismo.pipeline import PipelineComponents

    centroids = np.asarray(centroids, dtype=float)
    vertices = np.repeat(centroids[:, None, :], 3, axis=1)
    return PipelineComponents(
        chargetransport=None,
        gyptis=None,
        gyptis_background=None,
        design_cell_centroids=lambda: centroids,
        design_cell_vertices=lambda: vertices,
    )


def _stub_mesh_transfer(
    monkeypatch: pytest.MonkeyPatch, n_nodes: int, n_design: int
) -> None:
    """Stub the silicon-triangulation read and transfer build for a fake mesh.

    The container tests drive degenerate coordinate sets (colinear or a single
    edge) that cannot triangulate, so the transfer machinery is replaced with a
    fixed ``(n_design, n_nodes)`` operator; the assembly wiring is covered end
    to end in ``test_pipeline``.
    """
    import prismo.mesh_transfer as mesh_transfer_module
    import prismo.waveguide_mesh as mesh_module

    monkeypatch.setattr(
        mesh_module,
        "read_mesh_silicon_triangulation",
        lambda _: np.empty((0, 3), dtype=np.intp),
    )
    monkeypatch.setattr(
        mesh_transfer_module,
        "build_mesh_transfer_operator",
        lambda *args, **kwargs: types.SimpleNamespace(
            dense=lambda: np.zeros((n_design, n_nodes)),
            shape=(n_design, n_nodes),
        ),
    )


def test_import() -> None:
    """Placeholder test that ensures package import works as expected."""
    import_module("prismo")


def test_cli_help() -> None:
    """Test the CLI help command."""
    main_module = import_module("prismo.main")
    result = runner.invoke(main_module.app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.stdout


def test_cli_run_help() -> None:
    """Test the CLI run subcommand help."""
    main_module = import_module("prismo.main")
    result = runner.invoke(main_module.app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "Run the PRISMO" in result.stdout


@pytest.mark.slow
def test_cli_run_without_backend_errors() -> None:
    """Ticket 04: a no-backend `prismo run` fails loudly, never fabricating a result.

    Without a container (no Julia/gyptis solver), the deleted physics-free
    stubs used to let the synthetic run "succeed" with fabricated carriers and
    an effective-medium neff. It must now surface the missing solver backend.
    """
    main_module = import_module("prismo.main")
    result = runner.invoke(
        main_module.app,
        ["--max-iter", "3", "--no-jit"],
    )
    assert result.exit_code != 0
    assert result.exception is not None
    assert "backend" in str(result.exception)


def test_container_run_seeds_signed_junction_for_optimization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The run seeds a signed lateral P/N junction as the initial design field."""
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    captured: dict[str, object] = {}
    output_captured: dict[str, object] = {}
    history = [
        {
            "iteration": index,
            "delta_n_eff": 1e-3,
            "delta_rho": 1e-3,
            "grad_norm": 1e-4,
            "wall_time": 0.1,
        }
        for index in range(1, 6)
    ]

    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return np.asarray([0.2, 0.3, 0.4]), history

    monkeypatch.setattr(optimizer_module, "optimize_doping", fake_optimize)
    monkeypatch.setattr(
        outputs_module,
        "generate_outputs",
        lambda **kwargs: (
            output_captured.update(kwargs)
            or [
                tmp_path / name
                for name in (
                    "convergence.pdf",
                    "doping_field.pdf",
                    "gradient_validation.pdf",
                )
            ]
        ),
    )
    _stub_mesh_transfer(monkeypatch, n_nodes=coords.shape[0], n_design=2)

    main_module._run_pipeline(
        r_min=50e-9,
        max_iter=5,
        ftol_rel=1e-5,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        no_jit=True,
        use_containers=True,
        components=_container_components([[0.5, 0.0], [1.5, 0.0]]),
    )

    # median x is 1.0; nodes at x<=1 seed p-type (-0.3), x>1 seed n-type (+0.3).
    np.testing.assert_allclose(captured["initial_rho"], [-0.3, -0.3, 0.3])
    assert captured["min_mma_evaluations"] == 5
    assert callable(captured["on_iteration"])
    # The container setup feeds the assembled mesh-transfer matrix to the solve.
    assert captured["design_transfer"] is not None
    assert np.asarray(captured["design_transfer"]).shape == (2, coords.shape[0])
    assert output_captured["gradient_validation_directions"] == 1
    np.testing.assert_allclose(
        output_captured["gradient_validation_steps"],
        [1e-4, 1e-3, 1e-2],
    )
    np.testing.assert_allclose(
        output_captured["gradient_validation_rho"],
        [-0.3, -0.3, 0.3],
    )


def test_container_run_rejects_near_zero_objective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A negligible (near-zero) Δneff fails the container validity gate."""
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.waveguide_mesh as mesh_module

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    history = [
        {
            "iteration": index,
            "delta_n_eff": 1e-15,
            "delta_rho": 0.0,
            "grad_norm": 1e-4,
            "wall_time": 0.1,
        }
        for index in range(1, 6)
    ]
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        optimizer_module,
        "optimize_doping",
        lambda **_: (np.full(2, 0.25), history),
    )
    _stub_mesh_transfer(monkeypatch, n_nodes=coords.shape[0], n_design=1)

    with pytest.raises(RuntimeError, match="invalid optimization signal"):
        main_module._run_pipeline(
            r_min=50e-9,
            max_iter=5,
            ftol_rel=1e-5,
            mesh_path=str(tmp_path / "mesh.msh"),
            output_dir=str(tmp_path),
            no_jit=True,
            use_containers=True,
            components=_container_components([[0.5, 0.0]]),
        )
