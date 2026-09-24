"""A physics-free ChargeTransport double served as a real Tesseract.

Stands in for the Julia drift-diffusion solver in tests: it implements the same
``apply`` / ``vector_jacobian_product`` / ``abstract_eval`` contract -- with the
exact same input/output field names as the real component -- so the production
:func:`build_chargetransport_component` and :func:`tesseract_jax.apply_tesseract`
drive it unchanged. At 0 V the carriers are ``1e18`` scaled by the doping (a
spatially varying doping gives a spatially varying carrier field); under bias
they are zero -- the depletion signal the tests key on.

Calls are recorded on the module (``FORWARD_BIASES`` / ``VJP_BIASES``) so a test
holding this imported module can assert how the pipeline drove it -- the served
handle runs this module in-process, so the lists it appends to are the ones the
test reads.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from prismo_shared.schemas import MeshRef
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

CARRIERS_0V = 1e18

# Bias voltages seen by each endpoint, in call order.
FORWARD_BIASES: list[float] = []
VJP_BIASES: list[float] = []
# Mesh-reference paths ``apply`` saw, in call order (``None`` when omitted).
FORWARD_MESH_PATHS: list[str | None] = []


class InputSchema(BaseModel):
    operation: str = "solve"
    doping: Differentiable[Array[(None,), Float64]] | None = None
    mesh_ref: MeshRef | None = None
    bias_voltage: float = Field(default=0.0)


class OutputSchema(BaseModel):
    electrons: Differentiable[Array[(None,), Float64]]
    holes: Differentiable[Array[(None,), Float64]]


def apply(inputs: InputSchema) -> OutputSchema:
    FORWARD_BIASES.append(inputs.bias_voltage)
    FORWARD_MESH_PATHS.append(inputs.mesh_ref.path if inputs.mesh_ref else None)
    doping = np.asarray(inputs.doping, dtype=float)
    scale = CARRIERS_0V if inputs.bias_voltage == 0.0 else 0.0
    carrier = scale * doping
    return OutputSchema(electrons=carrier, holes=carrier)


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, ShapeDType]:
    shape = tuple(abstract_inputs.doping.shape)
    return {
        "electrons": ShapeDType(shape=shape, dtype="float64"),
        "holes": ShapeDType(shape=shape, dtype="float64"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, npt.ArrayLike],
) -> dict[str, npt.ArrayLike]:
    VJP_BIASES.append(inputs.bias_voltage)
    doping = np.asarray(inputs.doping, dtype=float)
    scale = CARRIERS_0V if inputs.bias_voltage == 0.0 else 0.0
    cot_e = np.asarray(cotangent_vector.get("electrons", 0.0), dtype=float)
    cot_p = np.asarray(cotangent_vector.get("holes", 0.0), dtype=float)
    if cot_e.ndim == 0:
        cot_e = np.full_like(doping, float(cot_e))
    if cot_p.ndim == 0:
        cot_p = np.full_like(doping, float(cot_p))
    return {"doping": scale * (cot_e + cot_p)}
