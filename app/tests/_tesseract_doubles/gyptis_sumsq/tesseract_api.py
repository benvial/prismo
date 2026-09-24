"""A physics-free gyptis double: ``neff_sq = sum(design_epsilon**2)``.

A structure-sensitive stand-in for the FEniCS eigensolve: because it responds to
each cell rather than only the mean, a fixed-mean redistribution moves ``neff_sq``
and its design gradient is spatially resolved. It uses the exact input/output
field names of the real component so the production
:func:`build_gyptis_components` and :func:`tesseract_jax.apply_tesseract` drive
it unchanged. Every design-epsilon field ``apply`` saw is recorded on the module
(``FIELDS``) so a test can confirm the perturbed field carried structure while
the background field stayed uniform.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

FIELDS: list[np.ndarray] = []


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
    return OutputSchema(neff_sq=np.asarray(float(np.sum(eps**2))))


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, ShapeDType]:
    return {"neff_sq": ShapeDType(shape=(), dtype="float64")}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, npt.ArrayLike],
) -> dict[str, npt.ArrayLike]:
    eps = np.asarray(inputs.design_epsilon, dtype=float)
    cot = float(np.asarray(cotangent_vector["neff_sq"]))
    return {"design_epsilon": 2.0 * cot * eps}
