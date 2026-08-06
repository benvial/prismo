"""Seam tests for the DEVSIM Tesseract component.

Public interface under test: apply() + vector_jacobian_product() in
tesseract_api.py (contract per ticket 01-research-tesseract-core-api).

The stub is a placeholder identity model (charge = doping); tests pin the
schema contract and gradient consistency, not physics.
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "devsim_tesseract_api", Path(__file__).resolve().parents[1] / "tesseract_api.py"
)
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)

InputSchema = _api.InputSchema
apply = _api.apply
vector_jacobian_product = _api.vector_jacobian_product

N_NODES = 5


def make_inputs(doping: np.ndarray | None = None) -> InputSchema:
    if doping is None:
        doping = np.full(N_NODES, 1e22)  # m^-3, typical doping magnitude
    return InputSchema(doping=doping)


def test_apply_returns_charge_per_node_with_same_shape_as_doping():
    inputs = make_inputs()
    outputs = apply(inputs)
    assert outputs.charge.shape == (N_NODES,)


def test_apply_identity_stub_maps_doping_to_charge():
    doping = np.linspace(-1e22, 1e22, N_NODES)
    outputs = apply(make_inputs(doping))
    np.testing.assert_allclose(outputs.charge, doping)


def test_vjp_returns_cotangent_for_each_requested_input():
    inputs = make_inputs()
    cotangent = {"charge": np.arange(N_NODES, dtype=float)}
    result = vector_jacobian_product(inputs, {"doping"}, {"charge"}, cotangent)
    assert set(result.keys()) == {"doping"}
    assert np.asarray(result["doping"]).shape == (N_NODES,)


def test_vjp_identity_stub_passes_cotangent_through():
    inputs = make_inputs()
    cotangent = np.arange(N_NODES, dtype=float)
    result = vector_jacobian_product(
        inputs, {"doping"}, {"charge"}, {"charge": cotangent}
    )
    np.testing.assert_allclose(result["doping"], cotangent)
