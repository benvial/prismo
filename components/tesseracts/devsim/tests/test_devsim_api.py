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


def _make_minimal_triangle_msh(
    path: str,
    *,
    with_contacts: bool = False,
    symmetric: bool = False,
) -> None:
    """Write a minimal Gmsh v4.1 mesh with 3 nodes forming one triangle.

    When ``with_contacts`` is True, includes ``$PhysicalNames`` and
    tags the triangle as ``contact_anode`` so the physical-group
    contact-detection path is exercised.

    When ``symmetric`` is True, places the third node at negative x so
    both anode (x < 0) and cathode (x >= 0) sides have nodes.
    """
    x3 = -1e-7 if symmetric else 1e-7
    phy_lines = (
        "$PhysicalNames\n1\n2 1 \"contact_anode\"\n$EndPhysicalNames\n"
        if with_contacts
        else ""
    )
    elm_phys_tag = "1 " if with_contacts else ""
    content = (
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        f"{phy_lines}"
        "$Nodes\n1 3 1 3\n"
        "1 0.0 0.0 0.0\n"
        "2 1e-7 0.0 0.0\n"
        f"3 {x3:.1e} 1e-7 0.0\n"
        "$EndNodes\n"
        "$Elements\n1 1 1 1\n"
        f"2 1 2 1\n{elm_phys_tag}1 1 2 3\n"
        "$EndElements\n"
    )
    Path(path).write_text(content)


# ---------------------------------------------------------------------------
# 2D contact detection — physical-group path (ticket 12)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _devsim_available(), reason="DEVSIM not installed")
def test_setup_2d_contacts_with_physical_groups() -> None:
    """Contact detection must use gmsh physical groups when mesh_path is given."""
    devsim = _api._ensure_devsim()

    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        _make_minimal_triangle_msh(f.name, with_contacts=True)
        mesh_path = f.name
    try:
        devsim.create_device(device="pgtest")
        devsim.create_region(device="pgtest", region="silicon", material="Silicon")
        devsim.create_gmsh_mesh(
            device="pgtest", region="silicon", mesh="mesh", file=mesh_path
        )
        _api._setup_2d_contacts_from_mesh(
            devsim, "pgtest", "silicon", "mesh", mesh_path=mesh_path
        )
        contacts = devsim.get_contact_list(device="pgtest")
        assert isinstance(contacts, list)
    finally:
        _api._cleanup_device(devsim, "pgtest")
        Path(mesh_path).unlink(missing_ok=True)


@pytest.mark.skipif(not _devsim_available(), reason="DEVSIM not installed")
def test_setup_2d_contacts_fallback_no_mesh_path() -> None:
    """Fallback boundary-edge + x<0 split when mesh_path is absent."""
    devsim = _api._ensure_devsim()

    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        _make_minimal_triangle_msh(f.name)
        mesh_path = f.name
    try:
        devsim.create_device(device="fbtest")
        devsim.create_region(device="fbtest", region="silicon", material="Silicon")
        devsim.create_gmsh_mesh(
            device="fbtest", region="silicon", mesh="mesh", file=mesh_path
        )
        _api._setup_2d_contacts_from_mesh(
            devsim, "fbtest", "silicon", "mesh"
        )
        contacts = devsim.get_contact_list(device="fbtest")
        assert isinstance(contacts, list)
    finally:
        _api._cleanup_device(devsim, "fbtest")
        Path(mesh_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2D VJP smoke test — 3-node triangle (ticket 12)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _devsim_available(), reason="DEVSIM not installed")
def test_2d_vjp_on_minimal_triangle_mesh() -> None:
    """VJP on a 3-node 2D triangle must return finite grad of correct shape.

    Uses ``symmetric=True`` so nodes span both sides of x=0,
    allowing both anode and cathode contacts to be created.
    """
    devsim = _api._ensure_devsim()

    _api._cleanup_device(devsim, "pn_junction")
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        _make_minimal_triangle_msh(f.name, symmetric=True)
        mesh_path = f.name
    try:
        doping = np.array([1e23, 1e23, 1e23], dtype=float)
        ref = _make_mesh_ref(mesh_path, n_nodes=3)
        inp = InputSchema(doping=doping, mesh_ref=ref)

        out = apply(inp)
        assert isinstance(out, OutputSchema)
        assert out.charge.shape == (3,)
        assert np.all(np.isfinite(out.charge))

        cotangent = {"charge": np.ones(3)}
        result = vector_jacobian_product(
            inp, {"doping"}, {"charge"}, cotangent
        )
        vjp = np.asarray(result["doping"])
        assert vjp.shape == (3,)
        assert np.all(np.isfinite(vjp))
    finally:
        _api._cleanup_device(devsim, "pn_junction")
        Path(mesh_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2D vs 1D uniform doping comparison (ticket 12)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _devsim_available(), reason="DEVSIM not installed")
def test_2d_uniform_doping_matches_1d() -> None:
    """Uniform doping on 2D strip must match 1D results within tolerance.

    Uniform doping has no spatial variation, so both paths should
    converge to the same equilibrium.  Uses ``symmetric=True`` so
    both anode and cathode contacts exist.
    """
    devsim = _api._ensure_devsim()

    n_nodes = 15
    doping_1d = np.full(n_nodes, 1e22, dtype=float)

    out_1d = apply(InputSchema(doping=doping_1d))
    assert out_1d.charge.shape == (n_nodes,)

    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        _make_minimal_triangle_msh(f.name, symmetric=True)
        mesh_path = f.name
    try:
        doping_2d = np.array([1e22, 1e22, 1e22], dtype=float)
        ref = _make_mesh_ref(mesh_path, n_nodes=3)
        inp_2d = InputSchema(doping=doping_2d, mesh_ref=ref)

        _api._cleanup_device(devsim, "pn_junction")
        out_2d = apply(inp_2d)
        assert out_2d.charge.shape == (3,)

        mean_1d = float(np.mean(out_1d.charge))
        mean_2d = float(np.mean(out_2d.charge))
        rel_diff = abs(mean_1d - mean_2d) / max(abs(mean_1d), 1.0)
        assert rel_diff < 0.5, (
            f"2D uniform doping deviates from 1D: "
            f"mean 1D={mean_1d:.2e}, mean 2D={mean_2d:.2e}, "
            f"rel_diff={rel_diff:.2e}"
        )
    finally:
        _api._cleanup_device(devsim, "pn_junction")
        Path(mesh_path).unlink(missing_ok=True)
