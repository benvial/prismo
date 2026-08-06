"""Seam tests for the gyptis Tesseract component.

Public interface under test: apply() + vector_jacobian_product() in
tesseract_api.py (contract per ticket 01-research-tesseract-core-api).

The stub is a placeholder effective-medium model (neff_sq = mean(epsilon));
tests pin the schema contract and gradient consistency, not physics.
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "gyptis_tesseract_api", Path(__file__).resolve().parents[1] / "tesseract_api.py"
)
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)

InputSchema = _api.InputSchema
apply = _api.apply
vector_jacobian_product = _api.vector_jacobian_product

N_ELEMENTS = 4


def make_inputs(epsilon: np.ndarray | None = None) -> InputSchema:
    if epsilon is None:
        epsilon = np.full(N_ELEMENTS, 12.0)  # ~silicon permittivity
    return InputSchema(epsilon=epsilon)


def test_apply_returns_scalar_squared_effective_index():
    outputs = apply(make_inputs())
    assert np.asarray(outputs.neff_sq).shape == ()


def test_apply_effective_medium_stub_averages_permittivity():
    epsilon = np.array([1.0, 2.0, 3.0, 4.0])
    outputs = apply(make_inputs(epsilon))
    np.testing.assert_allclose(outputs.neff_sq, 2.5)


def test_vjp_returns_cotangent_for_each_requested_input():
    inputs = make_inputs()
    result = vector_jacobian_product(inputs, {"epsilon"}, {"neff_sq"}, {"neff_sq": 1.0})
    assert set(result.keys()) == {"epsilon"}
    assert np.asarray(result["epsilon"]).shape == (N_ELEMENTS,)


def test_vjp_effective_medium_stub_spreads_cotangent_evenly():
    inputs = make_inputs()
    result = vector_jacobian_product(inputs, {"epsilon"}, {"neff_sq"}, {"neff_sq": 1.0})
    np.testing.assert_allclose(result["epsilon"], np.full(N_ELEMENTS, 1 / N_ELEMENTS))
