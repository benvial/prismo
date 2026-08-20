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
    from prismo_shared.schemas import MeshRef

    return MeshRef(path=mesh_path, n_nodes=n_nodes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inputs(doping: np.ndarray | None = None) -> InputSchema:
    if doping is None:
        doping = np.full(N_NODES, 1e15)
    return InputSchema(doping=doping)


def _smooth_mixed_sign_pn_profile() -> np.ndarray:
    magnitude = np.geomspace(1e14, 1e20, 62)
    return np.where(np.arange(62) < 31, -magnitude[::-1], magnitude)


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


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_apply_returns_output_schema() -> None:
    outputs = apply(make_inputs())
    assert isinstance(outputs, OutputSchema)


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_apply_returns_fields_per_node_with_same_shape_as_doping() -> None:
    inputs = make_inputs()
    outputs = apply(inputs)
    assert np.asarray(outputs.electrons).shape == (N_NODES,)
    assert np.asarray(outputs.holes).shape == (N_NODES,)


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_apply_returns_finite_values() -> None:
    doping = np.array([1e22, -5e21, 0.0, 5e21, -1e22])
    outputs = apply(make_inputs(doping))
    assert np.all(np.isfinite(outputs.electrons))
    assert np.all(np.isfinite(outputs.holes))


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_apply_deterministic() -> None:
    doping = np.array([1e22, -5e21, 2e21, -2e21, 1e20])
    out1 = apply(make_inputs(doping))
    out2 = apply(make_inputs(doping))
    np.testing.assert_allclose(out1.electrons, out2.electrons)
    np.testing.assert_allclose(out1.holes, out2.holes)


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_apply_output_ordering_matches_input() -> None:
    doping = np.arange(N_NODES, dtype=float) * 1e21
    outputs = apply(make_inputs(doping))
    assert np.asarray(outputs.electrons).shape == doping.shape
    assert np.asarray(outputs.holes).shape == doping.shape


# ---------------------------------------------------------------------------
# vector_jacobian_product() contract
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_vjp_returns_cotangent_for_requested_input() -> None:
    inputs = make_inputs()
    apply(inputs)
    cotangent = {"electrons": np.ones(N_NODES), "holes": np.ones(N_NODES)}
    result = vector_jacobian_product(
        inputs, {"doping"}, {"electrons", "holes"}, cotangent
    )
    assert set(result.keys()) == {"doping"}
    assert np.asarray(result["doping"]).shape == (N_NODES,)


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_vjp_returns_finite_values() -> None:
    inputs = make_inputs()
    apply(inputs)
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


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_vjp_linear_in_cotangent() -> None:
    inputs = make_inputs()
    apply(inputs)
    cot1 = {"electrons": np.ones(N_NODES), "holes": np.ones(N_NODES)}
    cot2 = {"electrons": np.full(N_NODES, 3.0), "holes": np.full(N_NODES, 3.0)}

    r1 = vector_jacobian_product(inputs, {"doping"}, {"electrons", "holes"}, cot1)
    r2 = vector_jacobian_product(inputs, {"doping"}, {"electrons", "holes"}, cot2)

    ratio = np.asarray(r2["doping"]) / np.asarray(r1["doping"])
    np.testing.assert_allclose(ratio, 3.0, rtol=1e-10)


@pytest.mark.skipif(
    _julia_available(), reason="exercises the no-Julia error path (ticket 04)"
)
def test_apply_without_backend_raises() -> None:
    """No physics-free identity fallback: a solve without Julia is a hard error."""
    with pytest.raises(RuntimeError, match="Julia drift-diffusion backend"):
        apply(make_inputs())


@pytest.mark.skipif(
    _julia_available(), reason="exercises the no-Julia error path (ticket 04)"
)
def test_vjp_without_backend_raises() -> None:
    """No physics-free identity VJP: an adjoint without Julia is a hard error."""
    with pytest.raises(RuntimeError, match="Julia adjoint backend"):
        vector_jacobian_product(
            make_inputs(),
            {"doping"},
            {"electrons", "holes"},
            {"electrons": np.full(N_NODES, 2.0), "holes": np.full(N_NODES, 3.0)},
        )


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_vjp_handles_scalar_cotangents() -> None:
    inputs = make_inputs()
    apply(inputs)
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


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
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


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_vjp_rejects_inputs_without_matching_forward() -> None:
    _api._session_registry.clear()

    with pytest.raises(RuntimeError, match="preceding apply"):
        vector_jacobian_product(
            make_inputs(),
            {"doping"},
            {"electrons"},
            {"electrons": np.ones(N_NODES)},
        )


def test_vjp_reuses_matching_persistent_worker_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public VJP follows its matching forward in one Julia worker."""
    requests: list[dict[str, object]] = []

    class FakeWorker:
        generation = 1

        def request(self, request: dict[str, object]) -> dict[str, bool]:
            requests.append(request)
            doping = np.load(str(request["doping_path"]))
            if request["operation"] == "forward":
                np.savez(
                    str(request["output_path"]),
                    electrons=doping * 2.0,
                    holes=doping * 3.0,
                )
            else:
                np.save(str(request["output_path"]), np.ones_like(doping))
            return {"ok": True}

    _api._session_registry.clear()
    monkeypatch.setattr(_api, "_julia_available", lambda: True)
    monkeypatch.setattr(_api, "_get_julia_worker", lambda: FakeWorker(), raising=False)

    inputs = make_inputs()
    apply(inputs)
    result = vector_jacobian_product(
        inputs,
        {"doping"},
        {"electrons", "holes"},
        {"electrons": np.ones(N_NODES), "holes": np.ones(N_NODES)},
    )

    assert np.asarray(result["doping"]).shape == (N_NODES,)
    assert [request["operation"] for request in requests] == ["forward", "vjp"]
    assert requests[0]["profile_key"] == requests[1]["profile_key"]


@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_shutdown_invalidates_retained_forward_state() -> None:
    """Worker teardown prevents VJPs from using another run's state."""
    inputs = make_inputs()
    apply(inputs)

    _api.shutdown()

    with pytest.raises(RuntimeError, match="preceding apply"):
        vector_jacobian_product(
            inputs,
            {"doping"},
            {"electrons"},
            {"electrons": np.ones(N_NODES)},
        )


@pytest.mark.parametrize(
    "forward_inputs,vjp_inputs",
    [
        (make_inputs(), make_inputs(np.full(N_NODES, 2e15))),
        (
            InputSchema(doping=np.full(N_NODES, 1e15), bias_voltage=0.0),
            InputSchema(doping=np.full(N_NODES, 1e15), bias_voltage=-5.0),
        ),
        (
            InputSchema(
                doping=np.full(N_NODES, 1e15),
                mesh_ref=_make_mesh_ref("/tmp/shared.msh", n_nodes=N_NODES),
            ),
            InputSchema(
                doping=np.full(N_NODES, 1e15),
                mesh_ref=_make_mesh_ref("/tmp/shared.msh", n_nodes=N_NODES + 1),
            ),
        ),
    ],
)
@pytest.mark.skipif(
    not _julia_available(),
    reason="requires a Julia backend; no physics-free stub (ticket 04)",
)
def test_vjp_rejects_changed_forward_inputs(
    forward_inputs: InputSchema,
    vjp_inputs: InputSchema,
) -> None:
    apply(forward_inputs)

    with pytest.raises(
        RuntimeError, match="identical doping, mesh reference, and bias"
    ):
        vector_jacobian_product(
            vjp_inputs,
            {"doping"},
            {"electrons"},
            {"electrons": np.ones(N_NODES)},
        )


# ---------------------------------------------------------------------------
# Worker timeout behavior (mocked Julia path)
# ---------------------------------------------------------------------------
#
# The forward/VJP calls dispatch through ``_JuliaWorker.request``, which
# raises ``TimeoutError`` when its ``select``-based deadline loop expires (see
# ``_JuliaWorker.request`` in tesseract_api.py). Mocking ``subprocess.run``
# does not reach that path any more, so these stubs patch ``_get_julia_worker``
# directly to exercise the real timeout handling in ``_run_julia_forward`` /
# ``_run_julia_adjoint``.


class _AlwaysTimesOutWorker:
    """Worker stub whose every request exceeds its timeout budget."""

    generation = 999999

    def request(self, request: dict[str, object]) -> dict[str, object]:
        raise TimeoutError("Julia worker exceeded 1s request timeout")


class _ForwardSucceedsThenTimesOutWorker:
    """Worker stub: one forward request succeeds, later requests time out.

    Lets a test register a matching forward state via a real ``apply()``
    call before forcing the *next* request (typically a VJP) to time out.
    """

    generation = 999998

    def __init__(self, doping: np.ndarray) -> None:
        self._doping = doping

    def request(self, request: dict[str, object]) -> dict[str, object]:
        if request.get("operation") != "forward":
            raise TimeoutError("Julia worker exceeded 1s request timeout")
        np.savez(
            request["output_path"],
            electrons=self._doping.copy(),
            holes=self._doping.copy(),
        )
        return {"ok": True}


def test_apply_propagates_julia_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doping = np.array([1e22, -5e21, 0.0, 5e21, -1e22], dtype=float)
    monkeypatch.setattr(_api, "_julia_available", lambda: True)
    monkeypatch.setattr(_api, "_get_julia_worker", lambda: _AlwaysTimesOutWorker())
    with pytest.raises(RuntimeError, match="Julia forward solve failed"):
        apply(InputSchema(doping=doping, bias_voltage=-5.0))


def test_vjp_propagates_julia_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doping = np.array([1e22, -5e21, 0.0, 5e21, -1e22], dtype=float)
    inputs = InputSchema(doping=doping, bias_voltage=-5.0)
    monkeypatch.setattr(_api, "_julia_available", lambda: True)
    monkeypatch.setattr(
        _api, "_get_julia_worker", lambda: _ForwardSucceedsThenTimesOutWorker(doping)
    )
    apply(inputs)
    with pytest.raises(RuntimeError, match="Julia adjoint solve failed"):
        vector_jacobian_product(
            inputs,
            {"doping"},
            {"electrons", "holes"},
            {"electrons": np.ones(N_NODES), "holes": np.ones(N_NODES)},
        )


# ---------------------------------------------------------------------------
# Subprocess integration test (Julia available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_apply_with_julia_subprocess() -> None:
    outputs = apply(make_inputs())
    assert isinstance(outputs, OutputSchema)
    assert np.all(np.isfinite(outputs.electrons))
    assert np.all(np.isfinite(outputs.holes))


@pytest.mark.parametrize("bias_voltage", [0.0, -5.0])
@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_apply_solves_smooth_mixed_sign_pn_profile(bias_voltage: float) -> None:
    """Public apply returns physical fields for smooth PN profiles."""
    doping = _smooth_mixed_sign_pn_profile()
    outputs = apply(InputSchema(doping=doping, bias_voltage=bias_voltage))

    electrons = np.asarray(outputs.electrons)
    holes = np.asarray(outputs.holes)
    assert np.all(np.isfinite(electrons))
    assert np.all(np.isfinite(holes))
    assert not np.allclose(electrons, doping)
    assert not np.allclose(holes, doping)


_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _shared_mesh_lateral_junction() -> tuple[Path, np.ndarray]:
    """Real 2D waveguide mesh + a lateral +/-1e19 junction in gmsh node order.

    The doping is contiguous in space (left half donors, right half acceptors)
    but flips sign 16 times along the raw node index -- exactly the field that
    scrambles into a many-junction line on the 1D fallback device.
    """
    mesh = _FIXTURES / "waveguide.msh"
    doping = np.load(_FIXTURES / "lateral_junction_doping.npy")
    return mesh, doping


@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_reverse_bias_converges_on_shared_mesh() -> None:
    """A lateral +/-1e19 junction converges at -5 V on the real 2D grid.

    On the 1D fallback (no mesh_ref) the same gmsh-order field is a 16-junction
    line the reverse-bias solve cannot converge on. Given the shared mesh, the
    junction is spatially coherent and the solve converges with correct P/N
    physics. Ref: .scratch/chargetransport-mesh-node-ordering/issues/03.
    """
    mesh, doping = _shared_mesh_lateral_junction()
    mesh_ref = _make_mesh_ref(str(mesh), n_nodes=len(doping))

    outputs = apply(InputSchema(doping=doping, bias_voltage=-5.0, mesh_ref=mesh_ref))
    electrons = np.asarray(outputs.electrons)
    holes = np.asarray(outputs.holes)

    assert electrons.shape == doping.shape
    assert np.all(np.isfinite(electrons))
    assert np.all(np.isfinite(holes))

    # Correct P/N physics: electrons accumulate on the n-side (donors,
    # doping < 0), holes on the p-side (acceptors, doping > 0). A junction
    # scrambled onto the wrong grid nodes would land carriers on the wrong
    # spatial halves.
    n_side = doping < 0
    p_side = doping > 0
    assert electrons[n_side].mean() > electrons[p_side].mean()
    assert holes[p_side].mean() > holes[n_side].mean()


@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_reverse_bias_adjoint_is_nonsingular_on_shared_mesh() -> None:
    """The -5 V adjoint solve must not be singular on the shared 2D mesh.

    A non-conforming mesh (duplicate nodes on internal interfaces) gives the
    finite-volume operator a null space: the forward solve limps through via
    continuation, but the adjoint's direct linear solve raises
    SingularException -- the failure that crashed `make run-containers`. The
    mesh is now generated conforming, so the adjoint returns finite gradients.
    Ref: .scratch/chargetransport-mesh-node-ordering/issues/03.
    """
    mesh, doping = _shared_mesh_lateral_junction()
    mesh_ref = _make_mesh_ref(str(mesh), n_nodes=len(doping))
    inputs = InputSchema(doping=doping, bias_voltage=-5.0, mesh_ref=mesh_ref)

    apply(inputs)
    result = vector_jacobian_product(
        inputs,
        {"doping"},
        {"electrons", "holes"},
        {"electrons": np.ones_like(doping), "holes": np.ones_like(doping)},
    )
    vjp = np.asarray(result["doping"])

    assert vjp.shape == doping.shape
    assert np.all(np.isfinite(vjp))
    assert not np.allclose(vjp, 0.0)


@pytest.mark.parametrize("bias_voltage", [0.0, -5.0])
@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_vjp_is_finite_for_smooth_mixed_sign_pn_profile(bias_voltage: float) -> None:
    """Public VJP returns responsive per-node gradients for supported PN profiles."""
    doping = _smooth_mixed_sign_pn_profile()
    inputs = InputSchema(doping=doping, bias_voltage=bias_voltage)
    apply(inputs)
    result = vector_jacobian_product(
        inputs,
        {"doping"},
        {"electrons", "holes"},
        {"electrons": np.ones_like(doping), "holes": np.ones_like(doping)},
    )
    vjp = np.asarray(result["doping"])

    assert vjp.shape == doping.shape
    assert np.all(np.isfinite(vjp))
    assert not np.allclose(vjp, 0.0)


@pytest.mark.parametrize("bias_voltage", [0.0, -5.0])
@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_vjp_matches_pn_forward_directional_difference(bias_voltage: float) -> None:
    """Public VJP matches finite differences of the public PN forward solve."""
    doping = _smooth_mixed_sign_pn_profile()
    direction = np.sin(np.arange(len(doping)) * 0.37)
    inputs = InputSchema(doping=doping, bias_voltage=bias_voltage)
    apply(inputs)
    cotangent = {"electrons": np.ones_like(doping), "holes": np.ones_like(doping)}
    vjp = np.asarray(
        vector_jacobian_product(inputs, {"doping"}, {"electrons", "holes"}, cotangent)[
            "doping"
        ]
    )

    step = 1e12

    def objective(doping_field: np.ndarray) -> float:
        outputs = apply(InputSchema(doping=doping_field, bias_voltage=bias_voltage))
        return float(np.sum(outputs.electrons) + np.sum(outputs.holes))

    finite_difference = (
        objective(doping + step * direction) - objective(doping - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(np.dot(vjp, direction), finite_difference, rtol=1e-2)


# ---------------------------------------------------------------------------
# Warm-start solver-state reuse (Julia worker persistence)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_warm_started_solve_matches_robust_continuation() -> None:
    """A nearby warm-started solve numerically matches a fresh continuation solve.

    ``apply()`` on a first profile leaves the persistent Julia worker holding
    that profile's converged state. A second, nearby profile then warm-starts
    Newton from it (``solve_at_bias_with_warm_start`` /
    ``solve_equilibrium_with_warm_start`` in ct_common.jl). ``shutdown()``
    discards that retained state, so re-running the second profile afterwards
    forces the robust doping-magnitude/adaptive-bias continuation path
    (``solve_equilibrium`` / ``solve_at_bias``) from scratch. Both must
    converge to the same physical carrier fields.
    """
    bias_voltage = -5.0
    base = _smooth_mixed_sign_pn_profile()
    nearby = base * 1.01  # small perturbation: stays in Newton's basin

    apply(InputSchema(doping=base, bias_voltage=bias_voltage))
    warm_started = apply(InputSchema(doping=nearby, bias_voltage=bias_voltage))

    _api.shutdown()

    cold_continuation = apply(InputSchema(doping=nearby, bias_voltage=bias_voltage))

    np.testing.assert_allclose(
        np.asarray(warm_started.electrons),
        np.asarray(cold_continuation.electrons),
        rtol=1e-6,
        atol=1e-6 * np.max(np.abs(np.asarray(cold_continuation.electrons))),
    )
    np.testing.assert_allclose(
        np.asarray(warm_started.holes),
        np.asarray(cold_continuation.holes),
        rtol=1e-6,
        atol=1e-6 * np.max(np.abs(np.asarray(cold_continuation.holes))),
    )


@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_failed_warm_start_falls_back_to_convergence_or_explicit_error() -> None:
    """A harsh jump that breaks single-step warm-start Newton still resolves cleanly.

    Warm-starting from a mild near-equilibrium profile directly into a large
    reverse bias on a full-magnitude PN profile is exactly the case
    ``solve_at_bias_with_warm_start``/``solve_equilibrium_with_warm_start``
    catch (``VoronoiFVM.ConvergenceError``/``AssemblyError``) and retry
    through the adaptive continuation path (see ct_common.jl comments on
    ticket 17: "A single Newton step from equilibrium fails to converge at
    large reverse bias"). Performance work must not turn that recoverable
    solve into a silent failure: either it still converges to finite carrier
    fields, or it raises the same explicit ``RuntimeError`` any other failed
    ChargeTransport solve would raise.
    """
    harsh_doping = _smooth_mixed_sign_pn_profile()
    # Same node count as harsh_doping: a matching mesh keeps the persistent
    # worker's context alive (no rebuild) so the next apply() genuinely
    # attempts a warm start from this mild equilibrium, rather than the
    # continuation path being reached "for free" via a mesh rebuild.
    apply(InputSchema(doping=np.full(len(harsh_doping), 1e14), bias_voltage=0.0))

    try:
        outputs = apply(InputSchema(doping=harsh_doping, bias_voltage=-5.0))
    except RuntimeError as exc:
        assert "Julia forward solve failed" in str(exc)
    else:
        assert np.all(np.isfinite(np.asarray(outputs.electrons)))
        assert np.all(np.isfinite(np.asarray(outputs.holes)))


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


@pytest.mark.parametrize("bias_voltage", [0.0, -5.0])
@pytest.mark.skipif(not _julia_available(), reason="Julia not installed")
def test_vjp_matches_gmsh_pn_directional_difference(bias_voltage: float) -> None:
    """Public VJP matches a Gmsh-mesh PN directional finite difference."""
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        _make_minimal_triangle_msh(f.name)
        mesh_path = f.name
    try:
        doping = np.array([-1e16, 1e16, 1e16], dtype=float)
        direction = np.array([0.5, -0.3, 0.2], dtype=float)
        step = 1e10
        mesh_ref = _make_mesh_ref(mesh_path, n_nodes=len(doping))
        inputs = InputSchema(
            doping=doping,
            mesh_ref=mesh_ref,
            bias_voltage=bias_voltage,
        )
        apply(inputs)
        vjp = np.asarray(
            vector_jacobian_product(
                inputs,
                {"doping"},
                {"electrons", "holes"},
                {
                    "electrons": np.ones_like(doping),
                    "holes": np.ones_like(doping),
                },
            )["doping"]
        )

        def objective(profile: np.ndarray) -> float:
            outputs = apply(
                InputSchema(
                    doping=profile,
                    mesh_ref=mesh_ref,
                    bias_voltage=bias_voltage,
                )
            )
            return float(np.sum(outputs.electrons) + np.sum(outputs.holes))

        finite_difference = (
            objective(doping + step * direction) - objective(doping - step * direction)
        ) / (2 * step)
        adjoint_direction = float(np.dot(vjp, direction))
        relative_error = abs(adjoint_direction - finite_difference) / max(
            abs(finite_difference),
            1e-30,
        )
        assert relative_error < 0.1
    finally:
        Path(mesh_path).unlink(missing_ok=True)


@pytest.mark.skipif(
    _julia_available(), reason="exercises the no-Julia error path (ticket 04)"
)
def test_apply_mesh_ref_without_backend_raises() -> None:
    """Without Julia, apply must raise -- no identity pass-through of the doping."""
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        _make_minimal_triangle_msh(f.name)
        mesh_path = f.name
    try:
        doping = np.array([1e22, -5e21, 2e21], dtype=float)
        ref = _make_mesh_ref(mesh_path, n_nodes=3)
        with pytest.raises(RuntimeError, match="Julia drift-diffusion backend"):
            apply(InputSchema(doping=doping, mesh_ref=ref))
    finally:
        Path(mesh_path).unlink(missing_ok=True)
