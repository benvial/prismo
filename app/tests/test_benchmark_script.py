"""Seam tests for ``scripts/benchmark_multiphysics_optimization.py``.

``make benchmark`` is the only caller, and it has no other coverage, so these
tests pin the two boundaries the script crosses into the rest of the project:
the container-mode pipeline setup (which must build the shared mesh and the
mesh-transfer operator exactly as ``prismo run --use-containers`` does) and the
single-component gyptis benchmark (which must speak the component's current
``design_epsilon`` input schema). Both were once broken against the code they call.
"""

from __future__ import annotations

import argparse
import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark_multiphysics_optimization.py"
)


def _load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_benchmark_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> types.ModuleType:
    return _load_script()


def _args(**overrides: object) -> argparse.Namespace:
    defaults = dict(
        iterations=2,
        n_nodes=None,
        mesh_path=Path("outputs/waveguide.msh"),
        r_min=0.05,
        mode="containers",
        component="full",
        no_jit=True,
        output=Path("outputs/multiphysics-benchmark.json"),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestFullPipelinePreparation:
    """The container benchmark must set up the same pipeline the CLI does."""

    def test_container_mode_builds_the_transfer_from_design_cell_vertices(
        self, script, monkeypatch, tmp_path
    ):
        """Ticket 15 defect 3: the third argument is vertices, not a mesh path.

        ``build_design_transfer(components, coords, mesh_path)`` fed a *string*
        into ``np.asarray(..., dtype=float)``, so the container benchmark
        raised before it timed anything.
        """
        pytest.importorskip("jax")
        import prismo.pipeline as pipeline_module

        vertices = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=float)
        mesh_file = tmp_path / "waveguide.msh"

        def write_mesh(path: str | Path) -> np.ndarray:
            Path(path).write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")
            return vertices

        components = types.SimpleNamespace(write_mesh=write_mesh)

        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        import prismo.waveguide_mesh as mesh_module

        monkeypatch.setattr(mesh_module, "read_mesh_node_coordinates", lambda _: coords)

        seen: dict[str, object] = {}

        def fake_transfer(components_arg, node_coords, design_cell_vertices=None):
            seen["vertices"] = design_cell_vertices
            return np.zeros((vertices.shape[0], node_coords.shape[0]))

        monkeypatch.setattr(pipeline_module, "build_design_transfer", fake_transfer)

        inputs = script._prepare_full_pipeline(_args(mesh_path=mesh_file), components)

        assert inputs.n_nodes == coords.shape[0]
        np.testing.assert_allclose(np.asarray(seen["vertices"]), vertices)
        assert np.asarray(inputs.design_transfer).shape == (1, 4)
        # The shared mesh comes from gyptis, as on the container CLI path.
        assert mesh_file.exists()
        # ChargeTransport must solve on that mesh, not on its 1D fallback.
        assert inputs.mesh_ref is not None
        assert inputs.mesh_ref.n_nodes == coords.shape[0]

    def test_container_mode_requires_a_mesh_authoring_backend(self, script, tmp_path):
        components = types.SimpleNamespace(write_mesh=None)
        with pytest.raises(RuntimeError, match="mesh authoring"):
            script._prepare_full_pipeline(
                _args(mesh_path=tmp_path / "waveguide.msh"), components
            )


class TestGyptisComponentBenchmark:
    """``make benchmark --component gyptis`` must match the gyptis schema."""

    def test_uses_the_design_epsilon_input_sized_to_the_design_region(
        self, script, monkeypatch
    ):
        """Ticket 15 defect 8: ``epsilon`` was removed by the unified mesh."""
        centroids = np.zeros((7, 2))
        calls: dict[str, object] = {}

        class InputSchema:
            def __init__(self, **kwargs: object) -> None:
                calls.setdefault("kwargs", []).append(kwargs)  # type: ignore[union-attr]
                self.kwargs = kwargs

        def apply(inputs: InputSchema) -> object:
            if "design_epsilon" not in inputs.kwargs:
                raise ValueError("design_epsilon is required")
            return types.SimpleNamespace(neff_sq=1.0)

        def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent):
            calls["vjp_inputs"] = set(vjp_inputs)
            calls["vjp_outputs"] = set(vjp_outputs)
            return {"design_epsilon": np.zeros(centroids.shape[0])}

        fake_api = types.SimpleNamespace(
            InputSchema=InputSchema,
            apply=apply,
            vector_jacobian_product=vector_jacobian_product,
            design_cell_centroids=lambda: centroids,
            DEFAULT_CORE_EPSILON=12.08,
        )
        monkeypatch.setattr(script, "_load_component_api", lambda _: fake_api)

        measurements = script._run_gyptis_benchmark(2)

        assert len(measurements) == 2
        assert calls["vjp_inputs"] == {"design_epsilon"}
        assert calls["vjp_outputs"] == {"neff_sq"}
        for kwargs in calls["kwargs"]:  # type: ignore[union-attr]
            assert "epsilon" not in kwargs
            assert np.asarray(kwargs["design_epsilon"]).shape == (7,)
