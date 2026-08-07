"""Seam tests for the ChargeTransport.jl Tesseract component.

Public interface under test: apply() + vector_jacobian_product() in
tesseract_api.py.

Covers schema validation, contract shapes, gradient consistency (VJP
matches finite-difference approximation), and subprocess dispatch.
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "chargetransport_tesseract_api",
    Path(__file__).resolve().parents[1] / "tesseract_api.py",
)
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)

InputSchema = _api.InputSchema
OutputSchema = _api.OutputSchema
apply = _api.apply
vector_jacobian_product = _api.vector_jacobian_product

N_NODES = 5


def _julia_available() -> bool:
    try:
        import subprocess

        subprocess.run(
            ["julia", "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _make_mesh_ref(mesh_path: str, n_nodes: int = 0) -> object:
    from tesseract_photonic_waveguide_shared.schemas import MeshRef

    return MeshRef(path=mesh_path, n_nodes=n_nodes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inputs(doping: np.ndarray | None = None) -> InputSchema:
    if doping is None:
        doping = np.full(N_NODES, 1e15)
    return InputSchema(doping=doping)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_input_schema_accepts_numpy_array() -> None:
    doping = np.array([1e22, -5e21, 2e21])
    inp = InputSchema(doping=doping)
    assert np.asarray(inp.doping).shape == (3,)


def test_input_schema_accepts_list() -> None:
    inp = InputSchema(doping=[1e22, -5e21, 2e21])
    assert np.asarray(inp.doping).shape == (3,)


def test_input_schema_default_bias_voltage_is_zero() -> None:
    inp = InputSchema(doping=[1e22, -5e21])
    assert inp.bias_voltage == 0.0


def test_input_schema_accepts_bias_voltage() -> None:
    inp = InputSchema(doping=[1e22, -5e21], bias_voltage=-5.0)
    assert inp.bias_voltage == -5.0


def test_input_schema_default_mesh_ref_is_none() -> None:
    inp = InputSchema(doping=[1e22, -5e21])
    assert inp.mesh_ref is None


def test_input_schema_accepts_mesh_ref() -> None:
    ref = _make_mesh_ref("/tmp/test.msh")
    inp = InputSchema(doping=[1e22, -5e21], mesh_ref=ref)
    assert inp.mesh_ref is not None
    assert inp.mesh_ref.path == "/tmp/test.msh"


def test_output_schema_electrons_and_holes_are_differentiable() -> None:
    out = OutputSchema(
        electrons=np.array([1.0, 2.0, 3.0]), holes=np.array([4.0, 5.0, 6.0])
    )
    assert np.asarray(out.electrons).shape == (3,)
    assert np.asarray(out.holes).shape == (3,)


# ---------------------------------------------------------------------------
# apply() contract
# ---------------------------------------------------------------------------


def test_apply_returns_output_schema() -> None:
    outputs = apply(make_inputs())
    assert isinstance(outputs, OutputSchema)


def test_apply_returns_fields_per_node_with_same_shape_as_doping() -> None:
    inputs = make_inputs()
    outputs = apply(inputs)
    assert np.asarray(outputs.electrons).shape == (N_NODES,)
    assert np.asarray(outputs.holes).shape == (N_NODES,)


def test_apply_returns_finite_values() -> None:
    doping = np.array([1e22, -5e21, 0.0, 5e21, -1e22])
    outputs = apply(make_inputs(doping))
    assert np.all(np.isfinite(outputs.electrons))
    assert np.all(np.isfinite(outputs.holes))


def test_apply_deterministic() -> None:
    doping = np.array([1e22, -5e21, 2e21, -2e21, 1e20])
    out1 = apply(make_inputs(doping))
    out2 = apply(make_inputs(doping))
    np.testing.assert_allclose(out1.electrons, out2.electrons)
    np.testing.assert_allclose(out1.holes, out2.holes)


def test_apply_stub_passes_doping_through() -> None:
    doping = np.array([1e22, -5e21, 0.0, 5e21, -1e22], dtype=float)
    outputs = apply(make_inputs(doping))
    np.testing.assert_allclose(outputs.electrons, doping)
    np.testing.assert_allclose(outputs.holes, doping)


def test_apply_output_ordering_matches_input() -> None:
    doping = np.arange(N_NODES, dtype=float) * 1e21
    outputs = apply(make_inputs(doping))
    assert np.asarray(outputs.electrons).shape == doping.shape
    assert np.asarray(outputs.holes).shape == doping.shape


# ---------------------------------------------------------------------------
# vector_jacobian_product() contract
# ---------------------------------------------------------------------------


def test_vjp_returns_cotangent_for_requested_input() -> None:
    inputs = make_inputs()
    cotangent = {"electrons": np.ones(N_NODES), "holes": np.ones(N_NODES)}
    result = vector_jacobian_product(
        inputs, {"doping"}, {"electrons", "holes"}, cotangent
    )
    assert set(result.keys()) == {"doping"}
    assert np.asarray(result["doping"]).shape == (N_NODES,)


def test_vjp_returns_finite_values() -> None:
    inputs = make_inputs()
    cotangent = {"electrons": np.ones(N_NODES), "holes": np.ones(N_NODES)}
    result = vector_jacobian_product(
        inputs, {"doping"}, {"electrons", "holes"}, cotangent
    )
    assert np.all(np.isfinite(result["doping"]))


def test_vjp_empty_when_input_not_requested() -> None:
    inputs = make_inputs()
    result = vector_jacobian_product(
        inputs, set(), {"electrons"}, {"electrons": np.ones(N_NODES)}
    )
    assert result == {}


def test_vjp_linear_in_cotangent() -> None:
    inputs = make_inputs()
    cot1 = {"electrons": np.ones(N_NODES), "holes": np.ones(N_NODES)}
    cot2 = {"electrons": np.full(N_NODES, 3.0), "holes": np.full(N_NODES, 3.0)}

    r1 = vector_jacobian_product(inputs, {"doping"}, {"electrons", "holes"}, cot1)
    r2 = vector_jacobian_product(inputs, {"doping"}, {"electrons", "holes"}, cot2)

    ratio = np.asarray(r2["doping"]) / np.asarray(r1["doping"])
    np.testing.assert_allclose(ratio, 3.0, rtol=1e-10)


def test_vjp_stub_sums_cotangents() -> None:
    inputs = make_inputs()
    cot_e = np.full(N_NODES, 2.0)
    cot_h = np.full(N_NODES, 3.0)
    result = vector_jacobian_product(
        inputs,
        {"doping"},
        {"electrons", "holes"},
        {"electrons": cot_e, "holes": cot_h},
    )
    np.testing.assert_allclose(result["doping"], cot_e + cot_h)


def test_vjp_handles_scalar_cotangents() -> None:
    inputs = make_inputs()
    result = vector_jacobian_product(
        inputs,
        {"doping"},
        {"electrons"},
        {"electrons": np.array(1.0)},
    )
    assert np.asarray(result["doping"]).shape == (N_NODES,)
    np.testing.assert_allclose(result["doping"], np.ones(N_NODES))


# ---------------------------------------------------------------------------
# Gradient consistency (VJP ≈ finite-difference)
# ---------------------------------------------------------------------------


def test_vjp_matches_finite_difference() -> None:
    doping = np.array([1e22, -5e21, 2e21, -2e21, 1e20], dtype=float)
    perturbation = np.array([0.1, -0.3, 0.2, 0.5, -0.1], dtype=float)
    h = 1e17

    apply(make_inputs(doping))

    cot_e = np.ones(N_NODES)
    cot_h = np.ones(N_NODES)

    vjp_out = np.asarray(
        vector_jacobian_product(
            make_inputs(doping),
            {"doping"},
            {"electrons", "holes"},
            {"electrons": cot_e, "holes": cot_h},
        )["doping"]
    )

    out_plus = apply(make_inputs(doping + h * perturbation))
    out_minus = apply(make_inputs(doping - h * perturbation))
    obj_plus = float(np.sum(out_plus.electrons) + np.sum(out_plus.holes))
    obj_minus = float(np.sum(out_minus.electrons) + np.sum(out_minus.holes))
    fd_grad_dir = (obj_plus - obj_minus) / (2 * h)

    vjp_dir = float(np.dot(vjp_out, perturbation))
    np.testing.assert_allclose(vjp_dir, fd_grad_dir, rtol=1e-5)


# ---------------------------------------------------------------------------
# Subprocess integration test (Julia available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_apply_with_julia_subprocess() -> None:
    outputs = apply(make_inputs())
    assert isinstance(outputs, OutputSchema)
    assert np.all(np.isfinite(outputs.electrons))
    assert np.all(np.isfinite(outputs.holes))


def _make_minimal_triangle_msh(path: str) -> None:
    """Write a minimal Gmsh v2.2 mesh with physical-group contacts.

    Three nodes form a right triangle. The three boundary edges carry the
    ``contact_anode`` (tag 1) and ``contact_cathode`` (tag 2) physical groups,
    so ExtendableGrids assigns exactly two boundary regions (bregion 1 = anode,
    bregion 2 = cathode) matching the mapping returned by the Julia
    ``get_breking_contacts`` helper.
    """
    content = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$PhysicalNames\n2\n"
        '1 1 "contact_anode"\n'
        '1 2 "contact_cathode"\n'
        "$EndPhysicalNames\n"
        "$Nodes\n3\n"
        "1 0.0 0.0 0.0\n"
        "2 1e-7 0.0 0.0\n"
        "3 0.0 1e-7 0.0\n"
        "$EndNodes\n"
        "$Elements\n4\n"
        "1 1 2 1 1 1 2\n"  # edge 1-2: contact_anode (bregion 1)
        "2 1 2 2 2 2 3\n"  # edge 2-3: contact_cathode (bregion 2)
        "3 1 2 1 3 3 1\n"  # edge 3-1: contact_anode (bregion 1)
        "4 2 2 1 4 1 2 3\n"  # triangle: cell region
        "$EndElements\n"
    )
    Path(path).write_text(content)


@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_apply_with_mesh_ref_and_physical_groups() -> None:
    """Forward solve on a 2D mesh with physical-group contacts.

    The mesh maps ``contact_anode``/``contact_cathode`` to bregions 1 and 2;
    the Julia forward path must set OhmicContact only on those regions and
    return finite carrier densities per mesh node.
    """
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        _make_minimal_triangle_msh(f.name)
        mesh_path = f.name
    try:
        doping = np.full(3, 1e22, dtype=float)
        ref = _make_mesh_ref(mesh_path, n_nodes=3)
        outputs = apply(InputSchema(doping=doping, mesh_ref=ref))
        assert isinstance(outputs, OutputSchema)
        assert np.asarray(outputs.electrons).shape == (3,)
        assert np.asarray(outputs.holes).shape == (3,)
        assert np.all(np.isfinite(outputs.electrons))
        assert np.all(np.isfinite(outputs.holes))
    finally:
        Path(mesh_path).unlink(missing_ok=True)


@pytest.mark.skipif(_julia_available(), reason="Julia installed; stub path not active")
def test_apply_mesh_ref_stub_path_works() -> None:
    """Without Julia, apply with a mesh_ref must still pass doping through."""
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        _make_minimal_triangle_msh(f.name)
        mesh_path = f.name
    try:
        doping = np.array([1e22, -5e21, 2e21], dtype=float)
        ref = _make_mesh_ref(mesh_path, n_nodes=3)
        outputs = apply(InputSchema(doping=doping, mesh_ref=ref))
        assert isinstance(outputs, OutputSchema)
        np.testing.assert_allclose(outputs.electrons, doping)
        np.testing.assert_allclose(outputs.holes, doping)
    finally:
        Path(mesh_path).unlink(missing_ok=True)
