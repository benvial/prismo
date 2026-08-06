# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for tesseract_photonic_waveguide_devsim
# Semiconductor drift-diffusion component (DEVSIM).
#
# STUB: placeholder identity model (charge = doping) pinning the interface
# contract. Real implementation: DEVSIM forward solve + implicit-diff VJP
# (Newton Jacobian extraction + adjoint solve), per tickets 02 and 05.

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64

#
# Schemas
#


class InputSchema(BaseModel):
    """Inputs to the DEVSIM drift-diffusion solve."""

    # Net doping at every mesh node [m^-3]; free-form design field (ticket 04).
    doping: Differentiable[Array[(None,), Float64]]


class OutputSchema(BaseModel):
    """Outputs of the DEVSIM drift-diffusion solve."""

    # Net charge at every mesh node [C] (stub: placeholder identity).
    charge: Differentiable[Array[(None,), Float64]]


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward solve (stub: identity placeholder for the DEVSIM solve)."""
    # TODO(ticket 09 follow-up): run DEVSIM drift-diffusion solve.
    return OutputSchema(charge=np.asarray(inputs.doping))


#
# Optional endpoints
#


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, npt.ArrayLike],
) -> dict[str, npt.ArrayLike]:
    """Adjoint gradient pass (stub: identity Jacobian transpose)."""
    # TODO: adjoint solve J^T mu = -grad J, contract with dR/d(theta).
    # Stub is identity (charge = doping), so d(charge)/d(doping) = I.
    vjp: dict[str, npt.ArrayLike] = {}
    if "doping" in vjp_inputs and "charge" in vjp_outputs:
        vjp["doping"] = np.asarray(cotangent_vector["charge"], dtype=float)
    return vjp
