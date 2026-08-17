"""Tests for the main application module."""

from importlib import import_module
from pathlib import Path

import numpy as np

import pytest
from typer.testing import CliRunner

runner = CliRunner()


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
def test_cli_run_synthetic() -> None:
    """Test `prismo run` on a synthetic mesh (no gmsh)."""
    main_module = import_module("prismo.main")
    result = runner.invoke(
        main_module.app,
        ["--max-iter", "3", "--no-jit"],
    )
    assert result.exit_code == 0, result.output
    assert "Done" in result.stdout


def test_container_run_passes_fixed_pn_polarity_to_optimization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Container execution keeps the same mixed-sign PN field throughout MMA."""
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    captured: dict[str, object] = {}
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

    monkeypatch.setattr(mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh")
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return np.asarray([0.2, 0.3, 0.4]), history

    monkeypatch.setattr(optimizer_module, "optimize_doping", fake_optimize)
    monkeypatch.setattr(
        outputs_module,
        "generate_outputs",
        lambda **kwargs: [tmp_path / name for name in (
            "convergence.pdf", "doping_field.pdf", "gradient_validation.pdf",
        )],
    )

    main_module._run_pipeline(
        r_min=50e-9,
        max_iter=5,
        ftol_rel=1e-5,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        no_jit=True,
        use_containers=True,
    )

    np.testing.assert_array_equal(captured["polarity"], [-1.0, -1.0, 1.0])
    assert captured["min_mma_evaluations"] == 5
