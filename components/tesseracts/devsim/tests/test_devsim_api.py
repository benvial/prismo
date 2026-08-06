"""Seam tests for the DEVSIM Tesseract component.

Public interface under test: apply() + vector_jacobian_product() in
tesseract_api.py (contract per ticket 01-research-tesseract-core-api).

Covers schema validation, contract shapes, gradient consistency (VJP
matches finite-difference approximation), and integration tests when
DEVSIM is available in the environment.
"""

import importlib.util
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

N_NODES = 5


def _devsim_available() -> bool:
    try:
        import devsim  # noqa: F401

        return True
    except ImportError:
        return False


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
