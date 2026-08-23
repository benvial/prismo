"""Tests for the main application module."""

import types
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import typer
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

    def write_mesh(path: str | Path) -> np.ndarray:
        Path(path).write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        return vertices

    return PipelineComponents(
        chargetransport=None,
        gyptis=None,
        gyptis_background=None,
        design_cell_centroids=lambda: centroids,
        design_cell_vertices=lambda: vertices,
        write_mesh=write_mesh,
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


def test_cli_validate_gradient_help() -> None:
    """The ticket 06 deliverable is exposed as its own subcommand."""
    main_module = import_module("prismo.main")
    result = runner.invoke(main_module.app, ["validate-gradient", "--help"])
    assert result.exit_code == 0, result.output
    assert "--use-containers" in result.stdout
    assert "--tolerance" in result.stdout


def test_run_gradient_validation_passes_for_real_gradient(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The composed adjoint agrees with central FD under the stated tolerance.

    The shared identity-carrier / mean-neff doubles are smooth in θ, so the
    gradient is real (not a stub short-circuit) and central FD tracks it.
    """
    import prismo.main as main_module
    import prismo.waveguide_mesh as mesh_module
    from _doubles import stub_components

    coords = np.asarray(
        [[0.0, 0.0], [1e-6, 0.0], [2e-6, 0.0], [3e-6, 0.0]], dtype=float
    )
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    # The stubbed mesh file is never written, so the silicon-group read that
    # picks the design nodes has nothing to open: report no silicon groups, and
    # every node of this four-node double stays a design variable.
    monkeypatch.setattr(
        mesh_module,
        "read_mesh_silicon_triangulation",
        lambda _: np.empty((0, 3), dtype=np.intp),
    )

    result = main_module._run_gradient_validation(
        r_min=0.05,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        tolerance=1e-2,
        n_directions=2,
        use_containers=False,
        components=stub_components(),
    )
    assert result.passed is True
    assert result.worst_rel_error <= 1e-2
    assert (tmp_path / "gradient_validation.pdf").exists()


@pytest.mark.slow
def test_cli_run_without_backend_errors(tmp_path: Path) -> None:
    """Ticket 04: a no-backend `prismo run` fails loudly, never fabricating a result.

    Without a container (no Julia/gyptis solver), the deleted physics-free
    stubs used to let the synthetic run "succeed" with fabricated carriers and
    an effective-medium neff. It must now surface the missing solver backend.

    Mesh and output paths are pinned to ``tmp_path``: on the defaults this run
    authors the *local* rib mesh over ``outputs/waveguide.msh``, the shared mesh
    a container run hands ChargeTransport, so running the suite would leave the
    next ``make run-containers`` reading a stale, differently-sized mesh.
    """
    main_module = import_module("prismo.main")
    result = runner.invoke(
        main_module.app,
        [
            "run",
            "--max-iter", "3",
            "--no-jit",
            "--mesh-path", str(tmp_path / "waveguide.msh"),
            "--output-dir", str(tmp_path),
        ],
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

    # Reverse bias is applied to the right cathode: left nodes seed n-type and
    # right nodes p-type so -5 V widens, rather than forward-biases, junction.
    np.testing.assert_allclose(captured["initial_rho"], [0.3, 0.3, -0.3])
    # The move limit and the per-evaluation checkpoint reach the optimizer
    # (ticket 19); the checkpoint lives next to the figures.
    assert captured["move_limit"] == pytest.approx(0.05)
    assert Path(captured["checkpoint_path"]) == tmp_path / "checkpoint.json"
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
        [0.3, 0.3, -0.3],
    )


def test_container_run_figures_probe_the_optimized_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The gradient figure binds the optimized function; the mode figure lands.

    The finite differences behind ``gradient_validation.pdf`` must run the same
    pipeline the optimizer drove -- filter, ``mesh_ref`` and transfer bound --
    not an unfiltered one on ChargeTransport's 1D fallback device (ticket 15).
    """
    import jax.numpy as jnp
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module
    from prismo.pipeline import PipelineComponents

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
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
    mode_calls: list[np.ndarray] = []

    def mode_field(design_epsilon, core_epsilon):
        mode_calls.append((np.asarray(design_epsilon), core_epsilon))
        return np.asarray([0.25, 1.0]), np.asarray([[-0.1, 0.0], [0.1, 0.0]])

    base = _container_components([[0.5, 0.0], [1.5, 0.0]])
    components = PipelineComponents(
        chargetransport=lambda doping, bias, mesh_ref=None: (doping, doping),
        gyptis=lambda design_epsilon, core_epsilon=None: jnp.mean(design_epsilon),
        gyptis_background=lambda design_epsilon, core_epsilon=None: jnp.mean(
            design_epsilon
        ),
        design_cell_centroids=base.design_cell_centroids,
        design_cell_vertices=base.design_cell_vertices,
        write_mesh=base.write_mesh,
        mode_field=mode_field,
    )

    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        optimizer_module,
        "optimize_doping",
        lambda **_: (np.asarray([0.2, 0.3, 0.4]), history),
    )
    monkeypatch.setattr(
        outputs_module,
        "generate_outputs",
        lambda **kwargs: (output_captured.update(kwargs) or []),
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
        components=components,
    )

    bound = output_captured["pipeline_fn"].keywords
    assert bound["H"] is not None
    assert bound["H_sum"] is not None
    assert bound["mesh_ref"] is not None
    assert bound["design_transfer"] is not None

    # The mode is queried once, on the design-cell permittivity field, against
    # the same background the solve components were given.
    from prismo.pipeline import DEFAULT_BACKGROUND_EPSILON

    assert len(mode_calls) == 1
    queried_design, queried_background = mode_calls[0]
    assert queried_design.shape == (2,)
    assert queried_background == DEFAULT_BACKGROUND_EPSILON
    mode = output_captured["mode_field"]
    np.testing.assert_allclose(mode.abs_e, [0.25, 1.0])
    # The rib outline is the bounding box of the design cells.
    assert mode.rib_bounds == (0.5, 1.5, 0.0, 0.0)


def test_container_run_survives_a_failing_mode_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A backend that cannot answer the mode query costs one figure, not all four.

    The query runs after a multi-minute optimization has already succeeded -- an
    image predating the ``mode_field`` operation, say, answers it with an error
    -- so it must degrade to a skipped figure rather than sink the run.
    """
    import jax.numpy as jnp
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module
    from prismo.pipeline import PipelineComponents

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    output_captured: dict[str, object] = {}
    history = [
        {
            "iteration": 1,
            "delta_n_eff": 1e-3,
            "delta_rho": 1e-3,
            "grad_norm": 1e-4,
            "wall_time": 0.1,
        }
    ]

    def failing_mode_field(design_epsilon, core_epsilon):
        raise RuntimeError("422 unknown operation 'mode_field'")

    base = _container_components([[0.5, 0.0], [1.5, 0.0]])
    components = PipelineComponents(
        chargetransport=lambda doping, bias, mesh_ref=None: (doping, doping),
        gyptis=lambda design_epsilon, core_epsilon=None: jnp.mean(design_epsilon),
        gyptis_background=lambda design_epsilon, core_epsilon=None: jnp.mean(
            design_epsilon
        ),
        design_cell_centroids=base.design_cell_centroids,
        design_cell_vertices=base.design_cell_vertices,
        write_mesh=base.write_mesh,
        mode_field=failing_mode_field,
    )

    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        optimizer_module,
        "optimize_doping",
        lambda **_: (np.asarray([0.2, 0.3, 0.4]), history),
    )
    monkeypatch.setattr(
        outputs_module,
        "generate_outputs",
        lambda **kwargs: (output_captured.update(kwargs) or []),
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
        components=components,
    )

    # The other figures were still requested, with no mode field.
    assert output_captured["mode_field"] is None
    assert output_captured["pipeline_fn"] is not None


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


def _solve_components_with_reset(base, resets: list[str]):
    """Live-looking doubles: identity carriers, mean neff, and a reset seam."""
    import jax.numpy as jnp
    from prismo.pipeline import PipelineComponents

    return PipelineComponents(
        chargetransport=lambda doping, bias, mesh_ref=None: (doping, doping),
        gyptis=lambda design_epsilon, core_epsilon=None: jnp.mean(design_epsilon),
        gyptis_background=lambda design_epsilon, core_epsilon=None: jnp.mean(
            design_epsilon
        ),
        design_cell_centroids=base.design_cell_centroids,
        design_cell_vertices=base.design_cell_vertices,
        write_mesh=base.write_mesh,
        reset_chargetransport=lambda: resets.append("reset"),
    )


def test_container_run_reports_warm_and_cold_objective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The reported optimum is re-solved cold; VπLπ comes from the cold value.

    The ChargeTransport worker is reset before the best design is evaluated
    once more, warm and cold Δneff are both printed, and a discrepancy beyond
    the tolerance is surfaced as a warning and handed to the convergence figure
    (ticket 20).
    """
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module
    from prismo.pipeline import vpi_lpi_v_cm

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    output_captured: dict[str, object] = {}
    calls: list[str] = []
    # The optimizer "saw" 1e-3; the doubles re-solve to something else, so the
    # warm/cold discrepancy is far above the tolerance.
    history = [
        {"iteration": 1, "delta_n_eff": 5e-4, "delta_rho": 0.0, "grad_norm": 1e-4,
         "wall_time": 0.1},
        {"iteration": 2, "delta_n_eff": 1e-3, "delta_rho": 1e-3, "grad_norm": 1e-4,
         "wall_time": 0.2},
        # A rejected trial after the best: the reported warm value is the max.
        {"iteration": 3, "delta_n_eff": 8e-4, "delta_rho": 1e-3, "grad_norm": 1e-4,
         "wall_time": 0.3},
    ]
    base = _container_components([[0.5, 0.0], [1.5, 0.0]])
    components = _solve_components_with_reset(base, calls)

    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)

    def fake_optimize(**kwargs):
        calls.append("optimize")
        return np.asarray([0.2, 0.3, 0.4]), history

    monkeypatch.setattr(optimizer_module, "optimize_doping", fake_optimize)
    monkeypatch.setattr(
        outputs_module,
        "generate_outputs",
        lambda **kwargs: (output_captured.update(kwargs) or []),
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
        components=components,
    )

    # Reset happens after the optimization and before the cold solve.
    assert calls == ["optimize", "reset"]
    cold = output_captured["cold_reevaluation"]
    assert cold is not None
    assert cold.warm_delta_neff == pytest.approx(1e-3)
    # The cold value is the bound pipeline at the reported design.
    expected_cold = float(output_captured["pipeline_fn"](np.asarray([0.2, 0.3, 0.4])))
    assert cold.cold_delta_neff == pytest.approx(expected_cold)
    assert not cold.passed
    out = capsys.readouterr().out
    assert "Delta_n_eff (warm, optimizer) = +1.000000e-03" in out
    assert "Delta_n_eff (cold re-solve)" in out
    assert "WARNING: warm/cold Delta_n_eff disagree" in out
    assert f"VpiLpi = {vpi_lpi_v_cm(expected_cold):+.4e}" in out


def test_container_run_without_reset_seam_skips_cold_reevaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    output_captured: dict[str, object] = {}
    history = [
        {"iteration": 1, "delta_n_eff": 1e-3, "delta_rho": 0.0, "grad_norm": 1e-4,
         "wall_time": 0.1},
    ]
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        optimizer_module,
        "optimize_doping",
        lambda **_: (np.asarray([0.2, 0.3, 0.4]), history),
    )
    monkeypatch.setattr(
        outputs_module,
        "generate_outputs",
        lambda **kwargs: (output_captured.update(kwargs) or []),
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

    assert output_captured["cold_reevaluation"] is None
    assert "Cold re-evaluation skipped" in capsys.readouterr().out


def test_run_clears_stale_live_doping_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A run starts with no ``doping_field_<n>.png`` from a previous run (ticket 21)."""
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module

    stale = [tmp_path / f"doping_field_{n}.png" for n in (1, 2, 17)]
    for frame in stale:
        frame.write_bytes(b"stale")
    keep = tmp_path / "doping_field.pdf"
    keep.write_bytes(b"headline")

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    history = [
        {"iteration": 1, "delta_n_eff": 1e-3, "delta_rho": 0.0, "grad_norm": 1e-4,
         "wall_time": 0.1},
    ]
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        optimizer_module,
        "optimize_doping",
        lambda **_: (np.asarray([0.2, 0.3, 0.4]), history),
    )
    monkeypatch.setattr(outputs_module, "generate_outputs", lambda **kwargs: [])
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

    assert not any(frame.exists() for frame in stale)
    assert keep.exists()


def test_gradient_validation_cold_resets_before_every_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--cold`` resets the worker before each FD sample (and the adjoint)."""
    import prismo.main as main_module
    import prismo.waveguide_mesh as mesh_module
    from _doubles import stub_components

    resets: list[str] = []
    components = stub_components(
        reset_chargetransport=lambda: resets.append("reset")
    )
    coords = np.asarray(
        [[0.0, 0.0], [1e-6, 0.0], [2e-6, 0.0], [3e-6, 0.0]], dtype=float
    )
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        mesh_module,
        "read_mesh_silicon_triangulation",
        lambda _: np.empty((0, 3), dtype=np.intp),
    )
    step_sizes = np.asarray([1e-3, 1e-2])

    result = main_module._run_gradient_validation(
        r_min=0.05,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        tolerance=1e-2,
        n_directions=2,
        step_sizes=step_sizes,
        use_containers=False,
        components=components,
        cold=True,
    )
    assert result.passed is True
    # One reset to prove the seam, one before the adjoint, one before each of
    # the 2 directions x 2 steps x 2 sides central-difference evaluations.
    assert len(resets) == 1 + 1 + 2 * len(step_sizes) * 2


def test_gradient_validation_cold_requires_a_reset_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import prismo.main as main_module
    import prismo.waveguide_mesh as mesh_module
    from _doubles import stub_components

    coords = np.asarray([[0.0, 0.0], [1e-6, 0.0]], dtype=float)
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        mesh_module,
        "read_mesh_silicon_triangulation",
        lambda _: np.empty((0, 3), dtype=np.intp),
    )
    with pytest.raises(RuntimeError, match="--cold requires"):
        main_module._run_gradient_validation(
            r_min=0.05,
            mesh_path=str(tmp_path / "mesh.msh"),
            output_dir=str(tmp_path),
            tolerance=1e-2,
            n_directions=1,
            use_containers=False,
            components=stub_components(),
            cold=True,
        )


def test_gradient_validation_default_is_warm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import prismo.main as main_module
    import prismo.waveguide_mesh as mesh_module
    from _doubles import stub_components

    resets: list[str] = []
    components = stub_components(
        reset_chargetransport=lambda: resets.append("reset")
    )
    coords = np.asarray(
        [[0.0, 0.0], [1e-6, 0.0], [2e-6, 0.0], [3e-6, 0.0]], dtype=float
    )
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        mesh_module,
        "read_mesh_silicon_triangulation",
        lambda _: np.empty((0, 3), dtype=np.intp),
    )
    main_module._run_gradient_validation(
        r_min=0.05,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        tolerance=1e-2,
        n_directions=1,
        step_sizes=np.asarray([1e-2]),
        use_containers=False,
        components=components,
    )
    assert resets == []


def test_local_pipeline_inputs_keep_the_seeded_junction_under_the_default_filter(
    tmp_path: Path,
) -> None:
    """The µm-scaled ``--r-min`` default must be a local filter on the local mesh.

    ``build_pipeline_inputs`` authors the rib mesh itself when no containers
    are running. That mesh used to be written in metres while ``--r-min``
    defaulted to ``0.05`` micrometres, so the filter radius was 0.05 m over a
    3 µm device: ``H`` became all-pairs and ``theta_tilde`` collapsed to the
    global mean, erasing the seeded P/N junction the optimizer starts from
    (ticket 15). Both are micrometres now, so the filtered field must still
    change sign across the junction.
    """
    pytest.importorskip("gmsh")
    import jax.numpy as jnp

    main_module = import_module("prismo.main")

    inputs = main_module.build_pipeline_inputs(
        r_min=0.05,
        mesh_path=str(tmp_path / "waveguide.msh"),
        use_containers=False,
        components=None,
    )

    assert inputs.real_mesh
    theta_tilde = np.asarray(
        (inputs.H_dense @ jnp.asarray(inputs.theta_init)) / inputs.H_sum
    )
    assert theta_tilde.min() < 0.0 < theta_tilde.max()

    # A local filter couples a node to a handful of neighbours, not to the
    # whole mesh; all-pairs coupling is exactly the mean-collapse failure.
    n_design = len(inputs.design_nodes)
    density = float(np.count_nonzero(np.asarray(inputs.H_dense))) / (n_design**2)
    assert density < 0.5


def test_mesh_size_refines_the_locally_authored_mesh(tmp_path: Path) -> None:
    """``--mesh-size`` is one knob on the silicon resolution, in micrometres.

    The container path hands it to the gyptis mesh author through
    ``PRISMO_GYPTIS_MESH_SIZE``; the local path sizes its own rib mesh with it,
    junction and bulk following at the class defaults' ratios. Halving it must
    therefore give a mesh with strictly more nodes.
    """
    pytest.importorskip("gmsh")

    main_module = import_module("prismo.main")
    from prismo.waveguide_mesh import RibWaveguideGeometry

    geometry = main_module._local_geometry(RibWaveguideGeometry, 0.02)
    assert (
        geometry.mesh_res_junction,
        geometry.mesh_res_core,
        geometry.mesh_res_bulk,
    ) == (0.01, 0.02, 0.05)

    coarse = main_module.build_pipeline_inputs(
        r_min=0.05,
        mesh_path=str(tmp_path / "coarse.msh"),
        use_containers=False,
        components=None,
        mesh_size=0.04,
    )
    fine = main_module.build_pipeline_inputs(
        r_min=0.05,
        mesh_path=str(tmp_path / "fine.msh"),
        use_containers=False,
        components=None,
        mesh_size=0.02,
    )

    assert coarse.real_mesh and fine.real_mesh
    assert fine.n_nodes > coarse.n_nodes
    assert len(fine.design_nodes) > len(coarse.design_nodes)


def test_mesh_size_must_be_positive() -> None:
    """A non-positive element size is a CLI mistake, not a degenerate mesh."""
    import typer

    main_module = import_module("prismo.main")
    from prismo.waveguide_mesh import RibWaveguideGeometry

    with pytest.raises(typer.BadParameter):
        main_module._local_geometry(RibWaveguideGeometry, 0.0)


def test_design_variables_live_on_the_silicon_nodes_only(tmp_path: Path) -> None:
    """The MMA design set is the silicon subdomain, not the whole mesh.

    Oxide, substrate, clad and PML nodes have no physics attached -- CT gathers
    doping on the silicon subgrid and every gyptis design cell is a rib
    triangle with silicon vertices -- so a variable there is either dead or
    dopes the device from outside it. The filter and the seed are sized by the
    design set, while the mesh contracts downstream stay full-length.
    """
    pytest.importorskip("gmsh")

    main_module = import_module("prismo.main")
    from prismo.waveguide_mesh import RibWaveguideGeometry

    inputs = main_module.build_pipeline_inputs(
        r_min=0.05,
        mesh_path=str(tmp_path / "waveguide.msh"),
        use_containers=False,
        components=None,
    )
    geometry = RibWaveguideGeometry()
    n_design = len(inputs.design_nodes)

    assert 0 < n_design < inputs.n_nodes
    assert inputs.theta_init.shape == (n_design,)
    assert inputs.H_dense.shape == (n_design, n_design)

    # Every design node sits in the silicon band (slab bottom to rib top);
    # nothing in the oxide carries a variable.
    design_coords = inputs.coords[inputs.design_nodes.indices]
    assert design_coords[:, 1].min() >= geometry.substrate_thickness - 1e-9
    assert design_coords[:, 1].max() <= geometry.rib_top + 1e-9

    # Scattering back to full node order is what keeps the solver contracts
    # (mesh_ref node ordering, the mesh-transfer operator) unchanged.
    full = inputs.design_nodes.scatter_numpy(np.asarray(inputs.theta_init))
    assert full.shape == (inputs.n_nodes,)
    np.testing.assert_allclose(
        full[inputs.design_nodes.indices], np.asarray(inputs.theta_init)
    )


def test_container_overlay_geometry_follows_the_gyptis_frame() -> None:
    """Container figures draw the overlay in the gyptis mesh's own frame.

    The gyptis author centres its layer stack on y = 0 with a 0.35 um
    substrate; RibWaveguideGeometry describes the local author's frame (y from
    0, 0.5 um substrate), so drawing it over container node coordinates put
    the rib outline off the device (ticket 16). The overlay must instead be
    derived from the design-cell vertices and node coordinates.
    """
    from prismo.main import PipelineInputs, _container_overlay_geometry
    from prismo.waveguide_mesh import RibWaveguideGeometry

    # gyptis-like frame: 500 nm x 220 nm rib sitting on the slab top at
    # y = -0.06, domain (incl. PML) spanning x in [-1.5, 1.5].
    vertices = np.asarray(
        [
            [[-0.25, -0.06], [0.25, -0.06], [0.25, 0.16]],
            [[-0.25, -0.06], [0.25, 0.16], [-0.25, 0.16]],
        ]
    )
    coords = np.asarray([[-1.5, -0.51], [1.5, 0.51], [0.0, 0.0]])
    inputs = PipelineInputs(
        geometry=RibWaveguideGeometry(),
        coords=coords,
        n_nodes=coords.shape[0],
        real_mesh=True,
        actual_mesh=Path("unused.msh"),
        mesh_ref=None,
        H_dense=None,
        H_sum=None,
        theta_init=None,
        design_transfer=None,
        design_vertices=vertices,
    )

    overlay = _container_overlay_geometry(inputs)
    assert overlay.rib_left == pytest.approx(-0.25)
    assert overlay.rib_right == pytest.approx(0.25)
    assert overlay.slab_top == pytest.approx(-0.06)
    assert overlay.rib_top == pytest.approx(0.16)
    # substrate/slab interface: slab_top minus the shared 100 nm slab.
    assert overlay.substrate_thickness == pytest.approx(-0.16)
    assert overlay.half_width == pytest.approx(1.5)

    # Without vertices there is nothing to derive from: keep the local frame.
    inputs_no_vertices = PipelineInputs(
        geometry=inputs.geometry,
        coords=coords,
        n_nodes=coords.shape[0],
        real_mesh=True,
        actual_mesh=Path("unused.msh"),
        mesh_ref=None,
        H_dense=None,
        H_sum=None,
        theta_init=None,
        design_transfer=None,
        design_vertices=None,
    )
    assert _container_overlay_geometry(inputs_no_vertices) is inputs.geometry


def _probe_mesh_stubs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, n_nodes: int = 4
) -> None:
    import prismo.waveguide_mesh as mesh_module

    coords = np.asarray([[i * 1e-6, 0.0] for i in range(n_nodes)], dtype=float)
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        mesh_module,
        "read_mesh_silicon_triangulation",
        lambda _: np.empty((0, 3), dtype=np.intp),
    )


def test_objective_probe_loads_the_checkpoint_design(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--design checkpoint.json`` probes around ``rho_opt``, not the seed."""
    import json

    import prismo.main as main_module
    from _doubles import stub_components

    _probe_mesh_stubs(monkeypatch, tmp_path)
    rho_opt = [0.25, -0.25, 0.5, -0.5]
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"rho_opt": rho_opt, "history": []}))

    # Identity carriers make f == 0 with a zero gradient; a bias-dependent
    # double gives the probe a real function and a direction to scan.
    def ct(doping, bias_voltage, mesh_ref=None):
        return doping * (1.0 + abs(bias_voltage)), doping

    components = stub_components(chargetransport=ct)
    scan = main_module._run_objective_probe(
        r_min=0.05,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        design_path=str(checkpoint),
        direction="gradient",
        spacing=1e-4,
        n_points=5,
        use_containers=False,
        components=components,
        cold=False,
    )
    assert scan.values.shape == (5,)
    assert (tmp_path / "objective_line_scan.pdf").exists()
    assert scan.offsets[len(scan.offsets) // 2] == 0.0
    # The centre sample is the checkpoint design, not the seed: the doubles
    # make f a function of the filtered design field, so the two differ.
    from functools import partial

    from prismo.pipeline import pipeline

    inputs = main_module.build_pipeline_inputs(
        0.05, str(tmp_path / "mesh.msh"), False, components
    )
    f = partial(
        pipeline,
        H=inputs.H_dense,
        H_sum=inputs.H_sum,
        design_nodes=inputs.design_nodes,
        components=components,
    )
    centre = scan.values[len(scan.offsets) // 2]
    assert centre == pytest.approx(float(f(np.asarray(rho_opt))), rel=1e-12)
    assert centre != pytest.approx(float(f(np.asarray(inputs.theta_init))), rel=1e-6)


def test_objective_probe_checkpoint_size_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    import prismo.main as main_module
    from _doubles import stub_components

    _probe_mesh_stubs(monkeypatch, tmp_path)
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"rho_opt": [0.1, 0.2], "history": []}))
    with pytest.raises(ValueError, match="design variables"):
        main_module._run_objective_probe(
            r_min=0.05,
            mesh_path=str(tmp_path / "mesh.msh"),
            output_dir=str(tmp_path),
            design_path=str(checkpoint),
            direction="gradient",
            spacing=1e-4,
            n_points=5,
            use_containers=False,
            components=stub_components(),
            cold=False,
        )


def test_objective_probe_cold_resets_before_every_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import prismo.main as main_module
    from _doubles import stub_components

    _probe_mesh_stubs(monkeypatch, tmp_path)
    resets: list[str] = []
    main_module._run_objective_probe(
        r_min=0.05,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        design_path=None,
        direction="random",
        spacing=1e-4,
        n_points=7,
        use_containers=False,
        components=stub_components(
            reset_chargetransport=lambda: resets.append("reset")
        ),
        cold=True,
    )
    # One to prove the seam, one before the gradient, one per sample.
    assert len(resets) == 1 + 1 + 7


def test_objective_probe_cold_requires_a_reset_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import prismo.main as main_module
    from _doubles import stub_components

    _probe_mesh_stubs(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="--cold requires"):
        main_module._run_objective_probe(
            r_min=0.05,
            mesh_path=str(tmp_path / "mesh.msh"),
            output_dir=str(tmp_path),
            design_path=None,
            direction="gradient",
            spacing=1e-4,
            n_points=3,
            use_containers=False,
            components=stub_components(),
            cold=True,
        )


def test_objective_probe_zero_gradient_points_at_random_direction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import prismo.main as main_module
    from _doubles import stub_components

    _probe_mesh_stubs(monkeypatch, tmp_path)
    # The identity double has f == 0 everywhere, so the gradient is zero.
    with pytest.raises(RuntimeError, match="--direction random"):
        main_module._run_objective_probe(
            r_min=0.05,
            mesh_path=str(tmp_path / "mesh.msh"),
            output_dir=str(tmp_path),
            design_path=None,
            direction="gradient",
            spacing=1e-4,
            n_points=3,
            use_containers=False,
            components=stub_components(),
            cold=False,
        )


def test_container_run_survives_a_failing_cold_resolve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A design that solves only warm loses its cold number, not the run.

    After the reset the ChargeTransport double raises (a cold ramp that does
    not converge); the run prints a warning, reports the warm value, and still
    generates the figures (ticket 23).
    """
    from dataclasses import replace

    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    output_captured: dict[str, object] = {}
    calls: list[str] = []
    history = [
        {"iteration": 1, "delta_n_eff": 1e-3, "delta_rho": 0.0, "grad_norm": 1e-4,
         "wall_time": 0.1},
    ]
    base = _container_components([[0.5, 0.0], [1.5, 0.0]])
    components = _solve_components_with_reset(base, calls)

    def cold_fails(doping, bias, mesh_ref=None):
        if "reset" in calls:
            raise RuntimeError("biased solve at -5.0 V failed: ConvergenceError")
        return doping, doping

    components = replace(components, chargetransport=cold_fails)

    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        optimizer_module,
        "optimize_doping",
        lambda **kwargs: (np.asarray([0.2, 0.3, 0.4]), history),
    )
    monkeypatch.setattr(
        outputs_module,
        "generate_outputs",
        lambda **kwargs: (output_captured.update(kwargs) or []),
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
        components=components,
    )

    assert calls == ["reset"]
    assert output_captured["cold_reevaluation"] is None
    out = capsys.readouterr().out
    assert "WARNING: cold re-solve of the reported design FAILED" in out
    assert "Best Delta_n_eff (warm) = +1.000000e-03" in out


# ---------------------------------------------------------------------------
# Ticket 25: loss-aware objective, junction seeds, geometry knobs
# ---------------------------------------------------------------------------


def _rib_coords() -> np.ndarray:
    """A rib-on-slab node set (µm) so the non-lateral seeds have a rib to find."""
    slab = np.array([[x, y] for y in (0.0, 0.1) for x in np.linspace(-1.0, 1.0, 9)])
    rib = np.array([[x, y] for y in (0.2, 0.32) for x in np.linspace(-0.25, 0.25, 5)])
    return np.vstack([slab, rib])


def test_run_seed_option_selects_the_initial_design(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module
    from prismo.pipeline import seed_design_field

    coords = _rib_coords()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        optimizer_module,
        "optimize_doping",
        lambda **kwargs: (captured.update(kwargs) or (kwargs["initial_rho"], [])),
    )
    monkeypatch.setattr(outputs_module, "generate_outputs", lambda **kwargs: [])
    _stub_mesh_transfer(monkeypatch, n_nodes=coords.shape[0], n_design=2)

    main_module._run_pipeline(
        r_min=50e-9,
        max_iter=1,
        ftol_rel=1e-5,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        no_jit=True,
        use_containers=True,
        components=_container_components([[0.0, 0.25], [0.1, 0.25]]),
        seed="u",
    )

    expected = np.asarray(seed_design_field(coords, "u"))
    np.testing.assert_array_equal(captured["initial_rho"], expected)
    assert not np.array_equal(expected, np.asarray(seed_design_field(coords, "lateral")))


def test_run_binds_loss_weight_and_mode_overlap_to_the_optimizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Overlap weights + weight reach the optimizer; headline reports loss and FOM."""
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    captured: dict[str, object] = {}
    calls: list[str] = []
    history = [
        {"iteration": 1, "objective": 4e-4, "delta_n_eff": 5e-4,
         "modal_loss_db_cm": 100.0, "delta_rho": 0.0, "grad_norm": 1e-4, "wall_time": 0.1},
        {"iteration": 2, "objective": 7e-4, "delta_n_eff": 1e-3,
         "modal_loss_db_cm": 300.0, "delta_rho": 1e-3, "grad_norm": 1e-4, "wall_time": 0.2},
    ]
    components = _solve_components_with_reset(
        _container_components([[0.5, 0.0], [1.5, 0.0]]), calls
    )
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return np.asarray([0.2, 0.3, 0.4]), history

    monkeypatch.setattr(optimizer_module, "optimize_doping", fake_optimize)
    monkeypatch.setattr(outputs_module, "generate_outputs", lambda **kwargs: [])
    _stub_mesh_transfer(monkeypatch, n_nodes=coords.shape[0], n_design=2)

    main_module._run_pipeline(
        r_min=50e-9,
        max_iter=2,
        ftol_rel=1e-5,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        no_jit=True,
        use_containers=True,
        components=components,
        loss_weight=1e-6,
    )

    assert captured["loss_weight"] == pytest.approx(1e-6)
    # The mean-field gyptis double has d(neff^2)/d(eps_cell) = 1/n over the two
    # design cells of the stubbed transfer.
    np.testing.assert_allclose(captured["mode_overlap"], [0.5, 0.5])
    out = capsys.readouterr().out
    assert "Loss weight: 1e-06" in out
    # The best design is the best *objective*; its loss and FOM are reported.
    assert "Best objective (warm) = +7.000000e-04" in out
    assert "modal loss" in out.lower()
    assert "V·dB" in out


def test_run_without_a_gyptis_backend_needs_no_overlap_unless_weighted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import prismo.main as main_module
    import prismo.optimizer as optimizer_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        optimizer_module,
        "optimize_doping",
        lambda **kwargs: (captured.update(kwargs) or (kwargs["initial_rho"], [])),
    )
    monkeypatch.setattr(outputs_module, "generate_outputs", lambda **kwargs: [])
    _stub_mesh_transfer(monkeypatch, n_nodes=coords.shape[0], n_design=2)
    common = dict(
        r_min=50e-9,
        max_iter=1,
        ftol_rel=1e-5,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        no_jit=True,
        use_containers=True,
        components=_container_components([[0.5, 0.0], [1.5, 0.0]]),  # gyptis=None
    )

    main_module._run_pipeline(**common)
    assert captured["mode_overlap"] is None
    assert "mode-overlap weights unavailable" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="loss-weight"):
        main_module._run_pipeline(**common, loss_weight=1e-6)


def test_cli_run_exposes_the_ticket_25_knobs() -> None:
    main_module = import_module("prismo.main")
    result = runner.invoke(main_module.app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--loss-weight", "--seed", "--contact-offset", "--domain-width"):
        assert flag in result.stdout
    # The default filter radius spans 3-4 elements of the 0.04 µm container mesh.
    assert main_module._DEFAULT_R_MIN == pytest.approx(0.10)


def test_container_run_forwards_the_geometry_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--contact-offset`` / ``--domain-width`` reach the container init."""
    import prismo.main as main_module
    import prismo.pipeline as pipeline_module

    captured: dict[str, object] = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(pipeline_module, "init_tesseract_containers", fake_init)
    result = runner.invoke(
        main_module.app,
        ["run", "--use-containers", "--contact-offset", "0.5", "--domain-width", "3.0"],
    )
    assert result.exit_code != 0
    assert captured["contact_offset"] == pytest.approx(0.5)
    assert captured["domain_width"] == pytest.approx(3.0)


def test_local_geometry_takes_the_contact_offset_and_width() -> None:
    import prismo.main as main_module
    from prismo.waveguide_mesh import RibWaveguideGeometry

    geometry = main_module._local_geometry(
        RibWaveguideGeometry, None, contact_offset=0.5, domain_width=3.0
    )
    assert geometry.contact_offset == pytest.approx(0.5)
    assert geometry.box_width == pytest.approx(3.0)
    with pytest.raises(typer.BadParameter):
        main_module._local_geometry(RibWaveguideGeometry, None, contact_offset=-0.1)


def test_bad_geometry_knobs_fail_before_any_container_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad ``--contact-offset``/``--seed`` is a CLI error, not a container one."""
    import prismo.main as main_module
    import prismo.pipeline as pipeline_module

    started: list[str] = []
    monkeypatch.setattr(
        pipeline_module,
        "init_tesseract_containers",
        lambda **kwargs: started.append("init") or None,
    )
    for command in ("run", "validate-gradient", "probe-objective"):
        for args in (["--contact-offset", "-0.1"], ["--seed", "diagonal"], ["--domain-width", "0"]):
            result = runner.invoke(main_module.app, [command, "--use-containers", *args])
            assert result.exit_code != 0, (command, args)
            assert "must be" in result.output, (command, args, result.output)
    assert started == []


def test_diagnostic_commands_expose_seed_and_geometry_knobs() -> None:
    main_module = import_module("prismo.main")
    for command in ("validate-gradient", "probe-objective"):
        result = runner.invoke(main_module.app, [command, "--help"])
        assert result.exit_code == 0, result.output
        for flag in ("--loss-weight", "--seed", "--contact-offset", "--domain-width"):
            assert flag in result.stdout, (command, flag)


def test_gradient_validation_uses_the_requested_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import prismo.main as main_module
    import prismo.outputs as outputs_module
    import prismo.waveguide_mesh as mesh_module
    from prismo.pipeline import seed_design_field

    coords = _rib_coords()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        mesh_module, "build_rib_waveguide_mesh", lambda **_: tmp_path / "mesh.msh"
    )
    monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)
    monkeypatch.setattr(
        mesh_module,
        "read_mesh_silicon_triangulation",
        lambda _: np.empty((0, 3), dtype=np.intp),
    )

    def fake_validate(pipeline_fn, rho, **kwargs):
        captured["rho"] = np.asarray(rho)
        return types.SimpleNamespace(
            passed=True, worst_rel_error=0.0, tolerance=1e-2,
            best_rel_errors=[0.0], figure_path=tmp_path / "gv.pdf",
        )

    monkeypatch.setattr(outputs_module, "validate_gradient", fake_validate)
    main_module._run_gradient_validation(
        r_min=0.1,
        mesh_path=str(tmp_path / "mesh.msh"),
        output_dir=str(tmp_path),
        tolerance=1e-2,
        n_directions=1,
        use_containers=False,
        components=None,
        seed="vertical",
    )
    np.testing.assert_array_equal(captured["rho"], np.asarray(seed_design_field(coords, "vertical")))


def test_local_geometry_rejects_contacts_outside_the_box() -> None:
    import prismo.main as main_module
    from prismo.waveguide_mesh import RibWaveguideGeometry

    with pytest.raises(typer.BadParameter, match="outside"):
        main_module._local_geometry(
            RibWaveguideGeometry, None, contact_offset=1.5, domain_width=3.0
        )
