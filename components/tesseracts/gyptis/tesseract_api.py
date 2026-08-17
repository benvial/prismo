# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for prismo_gyptis
# Electromagnetic eigenmode component (gyptis / FEniCS).
#
# Real implementation: 2D waveguide eigenmode solve with a finite-difference
# VJP. Falls back to an effective-medium stub when gyptis/FEniCS is absent.

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from collections import OrderedDict
from tesseract_core.runtime import Array, Differentiable, Float64

#
# Schemas
#


class InputSchema(BaseModel):
    """Inputs to the gyptis eigenmode solve.

    Attributes:
        epsilon: Relative permittivity per subdomain. Each element
            corresponds to one material region of the waveguide cross-section
            (e.g. core, cladding, substrate).
    """

    epsilon: Differentiable[Array[(None,), Float64]]


class OutputSchema(BaseModel):
    """Outputs of the gyptis eigenmode solve.

    Attributes:
        neff_sq: Squared effective index of the fundamental mode
            (neff_sq = kz^2 / k0^2, where kz is the propagation constant
            and k0 is the free-space wavenumber).
    """

    neff_sq: Differentiable[Array[(), Float64]]


#
# Module-level state
#

# Cached after apply() so vector_jacobian_product() can re-access geometry
# and eigenvector without re-solving.
_solve_state: dict[str, Any] | None = None


#
# Internal helpers
#


def _ensure_dolfin() -> Any:
    """Import dolfin lazily; raises ImportError with a helpful message if absent."""
    import dolfin  # type: ignore[import-untyped]

    return dolfin


def _ensure_gyptis() -> Any:
    """Import gyptis lazily; raises ImportError with a helpful message if absent."""
    import gyptis  # type: ignore[import-untyped]

    return gyptis


def _build_waveguide(
    epsilon: np.ndarray, wavelength: float = 1.55
) -> tuple[Any, Any, Any, float]:
    """Build a 2D waveguide simulation with layered geometry.

    Creates a rectangular cross-section with horizontal layers via
    LayeredBoxPML, one layer per epsilon element. Uses gyptis
    Waveguide for the eigenmode formulation.

    Args:
        epsilon: Per-layer relative permittivity values.
        wavelength: Free-space wavelength in meters (default 1.55 μm).

    Returns:
        Tuple of (simulation, geometry, n_domains, wavenumber_k0).
    """
    gyptis = _ensure_gyptis()
    _ensure_dolfin()

    n_domains = len(epsilon)
    k0 = 2.0 * np.pi / wavelength

    width = 2  # 2 μm cross-section width
    height = 1  # 1 μm cross-section height
    layer_thickness = height / n_domains
    thicknesses = OrderedDict({"domain_" + str(i + 1): layer_thickness for i in range(n_domains)})

    geom = gyptis.geometry.LayeredBoxPML2D(
        width, thicknesses=thicknesses, pml_width=(0.5, 0.5)
    )
    geom.build()

    eps_dict: dict[str, float] = {}
    for i, eps in enumerate(epsilon):
        eps_dict["domain_" + str(i + 1)] = float(eps)

    simu = gyptis.Waveguide(geom, epsilon=eps_dict, wavenumber=k0)

    return simu, geom, n_domains, k0


#
# Required endpoint
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward eigenmode solve.

    Args:
        inputs: Relative permittivity per subdomain.

    Returns:
        Squared effective index of the fundamental guided mode.
    """
    epsilon = np.asarray(inputs.epsilon, dtype=float)
    if epsilon.ndim != 1 or epsilon.size == 0:
        raise ValueError("epsilon must contain at least one material domain")

    try:
        _ensure_dolfin()
        _ensure_gyptis()
    except ImportError:
        return OutputSchema(neff_sq=float(np.mean(epsilon)))

    simu, _geom, n_domains, k0 = _build_waveguide(epsilon)

    _sol = simu.eigensolve(n_eig=4, target=k0)

    j_fundamental = 0
    ev_re, ev_im, _rx, _cx = simu.eigensolver.get_eigenpair(j_fundamental)
    lam = ev_re + 1j * ev_im
    kz = np.sqrt(lam)
    neff = float(np.real(kz / k0))

    global _solve_state
    _solve_state = {
        "simu": simu,
        "n_domains": n_domains,
        "k0": k0,
        "eigen_index": j_fundamental,
    }

    return OutputSchema(neff_sq=neff * neff)


#
# Optional endpoint
#


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, npt.ArrayLike],
) -> dict[str, npt.ArrayLike]:
    """Compute a VJP with centered differences of the public forward map.

    gyptis's PML formulation produces a non-Hermitian generalized eigenproblem.
    A right-eigenvector-only Hellmann-Feynman quotient is invalid there; centered
    differences provide a reliable derivative until a left-eigenvector adjoint is
    available.

    Args:
        inputs: Same InputSchema as the preceding apply() call.
        vjp_inputs: Input fields to compute cotangents for ({"epsilon"}).
        vjp_outputs: Output fields the cotangent_vector was taken w.r.t.
            ({"neff_sq"}).
        cotangent_vector: Cotangent on output fields, e.g.
            ``{"neff_sq": v}`` where v is a scalar dL/d(neff_sq).

    Returns:
        Dict mapping requested input fields to their cotangents, e.g.
        ``{"epsilon": dL/d(epsilon)}``.
    """
    vjp: dict[str, npt.ArrayLike] = {}

    if "epsilon" not in vjp_inputs or "neff_sq" not in vjp_outputs:
        return vjp

    cotangent = float(np.asarray(cotangent_vector["neff_sq"]))
    epsilon = np.asarray(inputs.epsilon, dtype=float)
    if epsilon.ndim != 1 or epsilon.size == 0:
        raise ValueError("epsilon must contain at least one material domain")
    n = len(epsilon)

    try:
        _ensure_dolfin()
        _ensure_gyptis()
    except ImportError:
        vjp["epsilon"] = np.full(n, cotangent / n)
        return vjp

    step = 1e-4
    gradient = np.empty(n)
    for index in range(n):
        perturbation = np.zeros(n)
        perturbation[index] = step
        upper = apply(InputSchema(epsilon=epsilon + perturbation))
        lower = apply(InputSchema(epsilon=epsilon - perturbation))
        gradient[index] = (float(upper.neff_sq) - float(lower.neff_sq)) / (2 * step)

    vjp["epsilon"] = cotangent * gradient
    return vjp
