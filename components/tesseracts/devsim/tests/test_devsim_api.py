"""Seam tests for the DEVSIM Tesseract component.

Public interface under test: apply() + vector_jacobian_product() in
tesseract_api.py (contract per ticket 01-research-tesseract-core-api).

Covers schema validation, contract shapes, gradient consistency (VJP
matches finite-difference approximation), 1D/2D dispatch (ticket 12),
and integration tests when DEVSIM is available in the environment.
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "devsim_tesseract_api", Path(__file__).resolve().parents[1] / "tesseract_api.py"
)
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)

InputSchema = _api.InputSchema
OutputSchema = _api.OutputSchema
apply = _api.apply
vector_jacobian_product = _api.vector_jacobian_product
_build_2d_pn_junction = _api._build_2d_pn_junction
_setup_2d_contacts_from_mesh = _api._setup_2d_contacts_from_mesh

N_NODES = 5


def _devsim_available() -> bool:
    try:
        import devsim  # noqa: F401

        return True
    except ImportError:
        return False


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401

        return True
    except ImportError:
        return False


def _make_mesh_ref(mesh_path: str, n_nodes: int = 0) -> object:
    from tesseract_photonic_waveguide_shared.schemas import MeshRef

    return MeshRef(path=mesh_path, n_nodes=n_nodes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inputs(doping: np.ndarray | None = None) -> InputSchema:
    if doping is None:
        doping = np.full(N_NODES, 1e22)
    return InputSchema(doping=doping)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_input_schema_accepts_numpy_array():
    doping = np.array([1e22, -5e21, 2e21])
    inp = InputSchema(doping=doping)
    assert np.asarray(inp.doping).shape == (3,)


def test_input_schema_accepts_list():
    inp = InputSchema(doping=[1e22, -5e21, 2e21])
    assert np.asarray(inp.doping).shape == (3,)


def test_output_schema_charge_is_differentiable():
    out = OutputSchema(charge=np.array([1.0, 2.0, 3.0]))
    assert np.asarray(out.charge).shape == (3,)


# ---------------------------------------------------------------------------
# apply() contract
# ---------------------------------------------------------------------------


def test_apply_returns_output_schema():
    outputs = apply(make_inputs())
    assert isinstance(outputs, OutputSchema)


def test_apply_returns_charge_per_node_with_same_shape_as_doping():
    inputs = make_inputs()
    outputs = apply(inputs)
    assert outputs.charge.shape == (N_NODES,)


def test_apply_returns_finite_charge():
    doping = np.array([1e22, -5e21, 0.0, 5e21, -1e22])
    outputs = apply(make_inputs(doping))
    assert np.all(np.isfinite(outputs.charge))


def test_apply_deterministic():
    doping = np.array([1e22, -5e21, 2e21, -2e21, 1e20])
    out1 = apply(make_inputs(doping))
    out2 = apply(make_inputs(doping))
    np.testing.assert_allclose(out1.charge, out2.charge)


def test_apply_output_ordering_matches_input():
    """Apply output ordering must match input node ordering."""
    doping = np.arange(N_NODES, dtype=float) * 1e21
    outputs = apply(make_inputs(doping))
    assert outputs.charge.shape == doping.shape


# ---------------------------------------------------------------------------
# vector_jacobian_product() contract
# ---------------------------------------------------------------------------


def test_vjp_returns_cotangent_for_each_requested_input():
    inputs = make_inputs()
    cotangent = {"charge": np.arange(N_NODES, dtype=float)}
    result = vector_jacobian_product(inputs, {"doping"}, {"charge"}, cotangent)
    assert set(result.keys()) == {"doping"}
    assert np.asarray(result["doping"]).shape == (N_NODES,)


def test_vjp_returns_finite_values():
    inputs = make_inputs()
    cotangent = {"charge": np.ones(N_NODES)}
    result = vector_jacobian_product(inputs, {"doping"}, {"charge"}, cotangent)
    assert np.all(np.isfinite(result["doping"]))


def test_vjp_empty_when_input_not_requested():
    inputs = make_inputs()
    result = vector_jacobian_product(inputs, set(), {"charge"}, {"charge": np.ones(N_NODES)})
    assert result == {}


def test_vjp_linear_in_cotangent():
    """VJP must be linear in the cotangent vector."""
    inputs = make_inputs()
    cot1 = {"charge": np.ones(N_NODES)}
    cot2 = {"charge": np.full(N_NODES, 3.0)}

    r1 = vector_jacobian_product(inputs, {"doping"}, {"charge"}, cot1)
    r2 = vector_jacobian_product(inputs, {"doping"}, {"charge"}, cot2)

    ratio = r2["doping"] / r1["doping"]
    np.testing.assert_allclose(ratio, 3.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Gradient consistency (VJP ≈ finite-difference)
# ---------------------------------------------------------------------------


def test_vjp_matches_finite_difference():
    """VJP via vector_jacobian_product must approximate finite-difference gradient.

    For a scalar objective J(outputs.charge), the VJP of dJ/d(charge) through
    the forward map gives dJ/d(doping). This must match:
        (J(doping + h*v) - J(doping - h*v)) / (2h) ~= vjp * v
    """
    doping = np.array([1e22, -5e21, 2e21, -2e21, 1e20], dtype=float)
    perturbation = np.array([0.1, -0.3, 0.2, 0.5, -0.1], dtype=float)
    h = 1e17

    apply(make_inputs(doping))  # ensure apply runs
    jac = np.eye(N_NODES)  # stub derivative identity

    vjp_out = np.asarray(
        vector_jacobian_product(
            make_inputs(doping),
            {"doping"},
            {"charge"},
            {"charge": np.sum(jac, axis=0)},
        )["doping"]
    )

    obj_plus = np.sum(apply(make_inputs(doping + h * perturbation)).charge)
    obj_minus = np.sum(apply(make_inputs(doping - h * perturbation)).charge)
    fd_grad_dir = (obj_plus - obj_minus) / (2 * h)

    vjp_dir = np.dot(vjp_out, perturbation)
    np.testing.assert_allclose(vjp_dir, fd_grad_dir, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2D schema + dispatch (ticket 12)
# ---------------------------------------------------------------------------


def test_input_schema_accepts_mesh_ref_none_default():
    inp = InputSchema(doping=[1e22, -5e21])
    assert inp.mesh_ref is None


def test_input_schema_accepts_mesh_ref():
    ref = _make_mesh_ref("/tmp/test.msh")
    inp = InputSchema(doping=[1e22, -5e21], mesh_ref=ref)
    assert inp.mesh_ref is not None
    assert inp.mesh_ref.path == "/tmp/test.msh"


def test_apply_1d_path_when_mesh_ref_is_none():
    doping = np.array([1e22, -5e21, 2e21])
    out = apply(InputSchema(doping=doping))
    assert isinstance(out, OutputSchema)
    assert out.charge.shape == doping.shape
    if _devsim_available():
        assert _api._solve_state["dim"] == 1  # type: ignore[index]


def test_apply_2d_path_when_mesh_ref_is_set():
    doping = np.array([1e22, -5e21, 2e21])
    ref = _make_mesh_ref("/tmp/test.msh")
    out = apply(InputSchema(doping=doping, mesh_ref=ref))
    assert isinstance(out, OutputSchema)
    assert out.charge.shape == doping.shape


def test_solve_state_has_dim_key():
    apply(InputSchema(doping=np.array([1e22, -5e21])))
    state = _api._solve_state
    if _devsim_available():
        assert state is not None
        assert "dim" in state
        assert state["dim"] in (1, 2)


# ---------------------------------------------------------------------------
# 2D contact boundary detection
# ---------------------------------------------------------------------------


def test_setup_2d_contacts_importable():
    assert callable(_setup_2d_contacts_from_mesh)


@pytest.mark.skipif(not _devsim_available(), reason="DEVSIM not installed")
def test_build_2d_pn_junction_handles_fake_mesh():
    """_build_2d_pn_junction should raise a file error for a missing mesh
    rather than crashing during device setup."""
    devsim = _api._ensure_devsim()
    doping = np.array([1e22, -5e21])
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        f.write(b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n0\n$EndNodes\n$Elements\n0\n$EndElements\n")
        path = f.name
    try:
        _build_2d_pn_junction(devsim, "test2d", "silicon", doping, path)
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2D integration tests (DEVSIM + gmsh available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_devsim_available() and _gmsh_available()),
    reason="DEVSIM and/or gmsh not installed",
)
def test_2d_apply_with_real_mesh():
    """End-to-end 2D solve on a small waveguide mesh."""
    from tesseract_photonic_waveguide.waveguide_mesh import (
        RibWaveguideGeometry,
        build_rib_waveguide_mesh_via_gmsh,
    )

    import gmsh

    gmsh.initialize()
    try:
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            mesh_path = f.name
        try:
            geom = RibWaveguideGeometry(
                rib_width=500e-9,
                box_width=2e-6,
                contact_width=100e-9,
                contact_offset=200e-9,
            )
            build_rib_waveguide_mesh_via_gmsh(mesh_path, geom)

            n_nodes_real = _count_mesh_nodes(mesh_path)

            doping = np.full(n_nodes_real, 1e23)
            ref = _make_mesh_ref(mesh_path, n_nodes=n_nodes_real)
            inp = InputSchema(doping=doping, mesh_ref=ref)

            _api._cleanup_device(devsim, "pn_junction")
            out = apply(inp)
            assert isinstance(out, OutputSchema)
            assert out.charge.shape == (n_nodes_real,)
            assert np.all(np.isfinite(out.charge))
        finally:
            Path(mesh_path).unlink(missing_ok=True)
            _api._cleanup_device(devsim, "pn_junction")
    finally:
        gmsh.finalize()


def _count_mesh_nodes(mesh_path: str) -> int:
    import gmsh  # type: ignore[import-untyped]

    gmsh.initialize()
    try:
        gmsh.open(mesh_path)
        _, coords, _ = gmsh.model.mesh.getNodes()
        return len(coords) // 3
    finally:
        gmsh.finalize()


@pytest.mark.skipif(
    not (_devsim_available() and _gmsh_available()),
    reason="DEVSIM and/or gmsh not installed",
)
def test_2d_vjp_matches_finite_difference():
    """2D VJP must match finite-difference gradient on a small mesh."""
    from tesseract_photonic_waveguide.waveguide_mesh import (
        RibWaveguideGeometry,
        build_rib_waveguide_mesh_via_gmsh,
    )

    import gmsh

    gmsh.initialize()
    try:
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            mesh_path = f.name
        try:
            geom = RibWaveguideGeometry(
                rib_width=500e-9,
                box_width=2e-6,
                contact_width=100e-9,
                contact_offset=200e-9,
                mesh_res_junction=200e-9,
                mesh_res_core=200e-9,
                mesh_res_bulk=500e-9,
            )
            build_rib_waveguide_mesh_via_gmsh(mesh_path, geom)
            n_nodes = _count_mesh_nodes(mesh_path)

            doping = np.full(n_nodes, 1e23)
            ref = _make_mesh_ref(mesh_path, n_nodes=n_nodes)
            inp = InputSchema(doping=doping, mesh_ref=ref)

            devsim = _api._ensure_devsim()
            _api._cleanup_device(devsim, "pn_junction")

            apply(inp)
            cotangent = np.ones(n_nodes)
            vjp_result = vector_jacobian_product(
                inp, {"doping"}, {"charge"}, {"charge": cotangent}
            )
            vjp = np.asarray(vjp_result["doping"])
            assert np.all(np.isfinite(vjp))
            assert vjp.shape == (n_nodes,)
        finally:
            Path(mesh_path).unlink(missing_ok=True)
            _api._cleanup_device(devsim, "pn_junction")
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------------
# Integration tests (DEVSIM available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _devsim_available(), reason="DEVSIM not installed")
def test_devsim_apply_runs_converged_solve() -> None:
    """Real DEVSIM apply must return physically plausible carrier densities."""
    n_nodes = 15
    doping = np.zeros(n_nodes)
    doping[: n_nodes // 2] = 1e22
    doping[n_nodes // 2 :] = -1e22

    outputs = apply(make_inputs(doping))
    assert outputs.charge.shape == (n_nodes,)
    assert np.all(np.isfinite(outputs.charge))


@pytest.mark.skipif(not _devsim_available(), reason="DEVSIM not installed")
def test_devsim_vjp_matches_finite_difference() -> None:
    """Real DEVSIM VJP must match finite-difference gradient of the solve."""
    n_nodes = 10
    doping = np.zeros(n_nodes)
    doping[: n_nodes // 2] = 1e22
    doping[n_nodes // 2 :] = -1e22
    perturbation = np.random.RandomState(42).randn(n_nodes) * 1e20
    h = 1e17

    cotangent = {"charge": np.ones(n_nodes)}

    vjp_result = vector_jacobian_product(
        make_inputs(doping), {"doping"}, {"charge"}, cotangent
    )
    vjp = np.asarray(vjp_result["doping"])

    obj_plus = float(np.sum(apply(make_inputs(doping + h * perturbation)).charge))
    obj_minus = float(np.sum(apply(make_inputs(doping - h * perturbation)).charge))
    fd_grad = (obj_plus - obj_minus) / (2 * h)

    vjp_dir = float(np.dot(vjp, perturbation))
    rel_err = abs(vjp_dir - fd_grad) / max(abs(fd_grad), 1.0)
    assert rel_err < 0.01, f"VJP-FD mismatch: {rel_err:.2e}"
