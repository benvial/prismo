"""A physics-free gyptis double: ``neff_sq = mean(design_epsilon)``.

Stands in for the FEniCS eigensolve with an effective-medium closed form, using
the exact input/output field names of the real component so the production
:func:`build_gyptis_components` and :func:`tesseract_jax.apply_tesseract` drive
it unchanged. The design-epsilon fields each ``apply`` saw are recorded on the
module (``FIELDS``) along with the ``mode_index`` each endpoint received
(``APPLY_MODE_INDICES`` / ``VJP_MODE_INDICES``), so a test holding this imported
module can assert what the pipeline sent -- the served handle runs it in-process.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

FIELDS: list[np.ndarray] = []
APPLY_MODE_INDICES: list[int] = []
VJP_MODE_INDICES: list[int] = []


class InputSchema(BaseModel):
    operation: str = "solve"
    design_epsilon: Differentiable[Array[(None,), Float64]] | None = None
    core_epsilon: float = 12.0
    mode_index: int = Field(0, ge=0)


class OutputSchema(BaseModel):
    neff_sq: Differentiable[Array[(), Float64]] | None = None


def apply(inputs: InputSchema) -> OutputSchema:
    eps = np.asarray(inputs.design_epsilon, dtype=float)
    FIELDS.append(eps)
    APPLY_MODE_INDICES.append(inputs.mode_index)
    return OutputSchema(neff_sq=np.asarray(float(np.mean(eps))))


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, ShapeDType]:
    return {"neff_sq": ShapeDType(shape=(), dtype="float64")}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, npt.ArrayLike],
) -> dict[str, npt.ArrayLike]:
    VJP_MODE_INDICES.append(inputs.mode_index)
    eps = np.asarray(inputs.design_epsilon, dtype=float)
    cot = float(np.asarray(cotangent_vector["neff_sq"]))
    return {"design_epsilon": np.full(eps.shape, cot / eps.size)}
