"""Seam tests for the gyptis Tesseract component.

Public interface under test: apply() + vector_jacobian_product() in
tesseract_api.py (contract per ticket 01-research-tesseract-core-api).

Covers schema validation, contract shapes, gradient consistency (VJP
matches finite-difference approximation), and integration tests when
gyptis/FEniCS is available in the environment.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "gyptis_tesseract_api", Path(__file__).resolve().parents[1] / "tesseract_api.py"
)
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)

InputSchema = _api.InputSchema
OutputSchema = _api.OutputSchema
apply = _api.apply
vector_jacobian_product = _api.vector_jacobian_product

N_ELEMENTS = 4


def _gyptis_available() -> bool:
    try:
        import dolfin  # noqa: F401
        import gyptis  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inputs(epsilon: np.ndarray | None = None) -> InputSchema:
    if epsilon is None:
        epsilon = np.full(N_ELEMENTS, 12.0)
    return InputSchema(epsilon=epsilon)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_input_schema_accepts_numpy_array() -> None:
    epsilon = np.array([12.0, 2.0, 1.0, 3.0])
    inp = InputSchema(epsilon=epsilon)
    assert np.asarray(inp.epsilon).shape == (4,)


def test_input_schema_accepts_list() -> None:
    inp = InputSchema(epsilon=[12.0, 2.0, 1.0, 3.0])
    assert np.asarray(inp.epsilon).shape == (4,)


def test_output_schema_neff_sq_is_scalar() -> None:
    out = OutputSchema(neff_sq=2.5)
    assert np.asarray(out.neff_sq).shape == ()


# ---------------------------------------------------------------------------
# apply() contract
# ---------------------------------------------------------------------------


def test_apply_returns_output_schema() -> None:
    outputs = apply(make_inputs())
    assert isinstance(outputs, OutputSchema)


def test_apply_returns_scalar_effective_index_squared() -> None:
    outputs = apply(make_inputs())
    assert np.asarray(outputs.neff_sq).shape == ()


def test_apply_returns_finite_neff_sq() -> None:
    epsilon = np.array([12.0, 11.0, 1.0, 1.0])
    outputs = apply(make_inputs(epsilon))
    assert np.isfinite(float(outputs.neff_sq))


def test_apply_deterministic() -> None:
    epsilon = np.array([12.0, 11.0, 1.0, 3.0])
    out1 = apply(make_inputs(epsilon))
    out2 = apply(make_inputs(epsilon))
    np.testing.assert_allclose(out1.neff_sq, out2.neff_sq)


def test_apply_effective_medium_stub_averages_permittivity() -> None:
    epsilon = np.array([1.0, 2.0, 3.0, 4.0])
    outputs = apply(make_inputs(epsilon))
    np.testing.assert_allclose(outputs.neff_sq, 2.5)


# ---------------------------------------------------------------------------
# vector_jacobian_product() contract
# ---------------------------------------------------------------------------


def test_vjp_returns_cotangent_for_each_requested_input() -> None:
    inputs = make_inputs()
    result = vector_jacobian_product(inputs, {"epsilon"}, {"neff_sq"}, {"neff_sq": 1.0})
    assert set(result.keys()) == {"epsilon"}
    assert np.asarray(result["epsilon"]).shape == (N_ELEMENTS,)


def test_vjp_returns_finite_values() -> None:
    inputs = make_inputs()
    result = vector_jacobian_product(
        inputs, {"epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
    )
    assert np.all(np.isfinite(result["epsilon"]))


def test_vjp_empty_when_input_not_requested() -> None:
    inputs = make_inputs()
    result = vector_jacobian_product(
        inputs, set(), {"neff_sq"}, {"neff_sq": 1.0}
    )
    assert result == {}


def test_vjp_linear_in_cotangent() -> None:
    """VJP must be linear in the cotangent vector."""
    inputs = make_inputs()
    r1 = vector_jacobian_product(
        inputs, {"epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
    )
    r2 = vector_jacobian_product(
        inputs, {"epsilon"}, {"neff_sq"}, {"neff_sq": 3.0}
    )
    ratio = np.asarray(r2["epsilon"]) / np.asarray(r1["epsilon"])
    np.testing.assert_allclose(ratio, 3.0, rtol=1e-10)


def test_vjp_effective_medium_stub_spreads_evenly() -> None:
    inputs = make_inputs()
    result = vector_jacobian_product(
        inputs, {"epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
    )
    np.testing.assert_allclose(result["epsilon"], np.full(N_ELEMENTS, 1 / N_ELEMENTS))


# ---------------------------------------------------------------------------
# Gradient consistency (VJP ~= finite-difference)
# ---------------------------------------------------------------------------


def test_vjp_matches_finite_difference() -> None:
    """VJP must approximate finite-difference gradient of the forward map.

    For neff_sq = f(epsilon), the VJP computes v^T * df/d(epsilon).
    The directional derivative must match:
        (f(eps + h*v) - f(eps - h*v)) / (2h) ~= vjp * v
    """
    epsilon = np.array([12.0, 11.0, 1.0, 1.0], dtype=float)
    perturbation = np.array([0.1, -0.3, 0.05, 0.2], dtype=float)
    h = 1e-5

    apply(make_inputs(epsilon))  # ensure apply runs

    vjp_result = vector_jacobian_product(
        make_inputs(epsilon), {"epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
    )
    vjp = np.asarray(vjp_result["epsilon"])

    f_plus = float(apply(make_inputs(epsilon + h * perturbation)).neff_sq)
    f_minus = float(apply(make_inputs(epsilon - h * perturbation)).neff_sq)
    fd_grad_dir = (f_plus - f_minus) / (2 * h)

    vjp_dir = float(np.dot(vjp, perturbation))
    np.testing.assert_allclose(vjp_dir, fd_grad_dir, rtol=1e-5)


# ---------------------------------------------------------------------------
# Integration tests (gyptis / FEniCS available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_gyptis_apply_runs_eigen_solve() -> None:
    """Real gyptis apply must return physically plausible neff."""
    epsilon = np.array([12.0, 2.0, 1.0])

    outputs = apply(make_inputs(epsilon))
    neff_sq = float(outputs.neff_sq)
    assert np.isfinite(neff_sq)
    assert 1.0 < neff_sq < 20.0  # physically plausible range for silicon


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_gyptis_vjp_matches_finite_difference() -> None:
    """Real gyptis VJP must match finite-difference gradient of the eigen solve."""
    n_domains = 3
    epsilon = np.array([12.0, 2.0, 1.0], dtype=float)
    perturbation = np.random.RandomState(42).randn(n_domains) * 0.1
    h = 1e-3

    vjp_result = vector_jacobian_product(
        make_inputs(epsilon), {"epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
    )
    vjp = np.asarray(vjp_result["epsilon"])

    f_plus = float(apply(make_inputs(epsilon + h * perturbation)).neff_sq)
    f_minus = float(apply(make_inputs(epsilon - h * perturbation)).neff_sq)
    fd_grad_dir = (f_plus - f_minus) / (2 * h)

    vjp_dir = float(np.dot(vjp, perturbation))
    rel_err = abs(vjp_dir - fd_grad_dir) / max(abs(fd_grad_dir), 1.0)
    assert rel_err < 0.01, f"VJP-FD mismatch: {rel_err:.2e}"
