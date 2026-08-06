# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for tesseract_photonic_waveguide_gyptis
# Electromagnetic eigenmode component (gyptis / FEniCS).
#
# STUB: placeholder effective-medium model (neff_sq = mean(epsilon)) pinning
# the interface contract. Real implementation: gyptis eigenmode solve +
# Hellmann-Feynman eigen-adjoint, per tickets 03 and 06.

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64

#
# Schemas
#


class InputSchema(BaseModel):
    """Inputs to the gyptis eigenmode solve."""

    # Relative permittivity per mesh element (from Soref-Bennett coupling).
    epsilon: Differentiable[Array[(None,), Float64]]


class OutputSchema(BaseModel):
    """Outputs of the gyptis eigenmode solve."""

    # Squared effective index of the fundamental mode (eigenvalue proxy).
    neff_sq: Differentiable[Array[(), Float64]]


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward solve (stub: effective-medium placeholder for the eigen solve)."""
    # TODO(ticket 10 follow-up): run gyptis eigenmode solve.
    return OutputSchema(neff_sq=float(np.mean(inputs.epsilon)))


#
# Optional endpoints
#


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, npt.ArrayLike],
) -> dict[str, npt.ArrayLike]:
    """Adjoint gradient pass (stub: uniform effective-medium Jacobian)."""
    # TODO: Hellmann-Feynman adjoint (x^H (dA/deps - lam dB/deps) x)/(x^H B x).
    # Stub: neff_sq = mean(epsilon), so d(neff_sq)/d(epsilon_i) = 1/N.
    vjp: dict[str, npt.ArrayLike] = {}
    if "epsilon" in vjp_inputs and "neff_sq" in vjp_outputs:
        n = len(inputs.epsilon)
        cot = float(np.asarray(cotangent_vector["neff_sq"]))
        vjp["epsilon"] = np.full(n, cot / n)
    return vjp
