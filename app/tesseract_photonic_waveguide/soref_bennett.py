"""Soref-Bennett free-carrier plasma dispersion coupling layer.

Converts carrier-density perturbations from DEVSIM into optical permittivity
and absorption perturbations for gyptis via the Soref-Bennett model (Soref &
Bennett, IEEE JQE 23(1), 1987).

DEVSIM reports carrier densities in m^-3 while the Soref-Bennett coefficients
are calibrated for cm^-3, so densities are converted before the model is
applied (1 m^-3 = 1e-6 cm^-3).

    Delta_n = -(A_e * Delta_N_e^B_e + A_h * Delta_N_h^B_h)
    Delta_alpha = C_e * Delta_N_e^D_e + C_h * Delta_N_h^D_h
    Delta_epsilon = 2 * n_bg * Delta_n  (first-order expansion)

Pure, element-wise and vectorized — no Python loops. Operates on lists,
NumPy arrays, or JAX arrays (converted via ``np.asarray``).
"""

import numpy as np
from tesseract_photonic_waveguide_shared.schemas import (
    CarrierDensityField,
    SorefBennettCoefficients,
    SorefBennettResult,
)

# 1 cm^-3 = 1e6 m^-3, so m^-3 -> cm^-3 divides by 1e6.
_M3_TO_CM3 = 1e-6


def soref_bennett(
    carriers: CarrierDensityField,
    coeffs: SorefBennettCoefficients | None = None,
) -> SorefBennettResult:
    """Compute Soref-Bennett permittivity and absorption perturbations.

    Args:
        carriers: ``CarrierDensityField`` with ``electrons``, ``holes`` and
            equilibrium densities (all [m^-3]).
        coeffs: ``SorefBennettCoefficients``; defaults to the lambda = 1.55 um
            silicon values if omitted.

    Returns:
        ``SorefBennettResult`` with ``delta_permittivity`` and
        ``delta_absorption`` (both element-wise lists).

    Raises:
        ValueError: if equilibrium carrier densities are not supplied.
    """
    if coeffs is None:
        coeffs = SorefBennettCoefficients()

    if (
        carriers.equilibrium_electrons is None
        or carriers.equilibrium_holes is None
    ):
        raise ValueError(
            "Equilibrium carrier densities required for Soref-Bennett"
        )

    n_e = np.asarray(carriers.electrons, dtype=float)
    n_h = np.asarray(carriers.holes, dtype=float)
    n_e_eq = np.asarray(carriers.equilibrium_electrons, dtype=float)
    n_h_eq = np.asarray(carriers.equilibrium_holes, dtype=float)

    # Convert density perturbations m^-3 -> cm^-3.
    dn_e = (n_e - n_e_eq) * _M3_TO_CM3
    dn_h = (n_h - n_h_eq) * _M3_TO_CM3

    dn = -(coeffs.A_e * dn_e**coeffs.B_e + coeffs.A_h * dn_h**coeffs.B_h)
    dalpha = coeffs.C_e * dn_e**coeffs.D_e + coeffs.C_h * dn_h**coeffs.D_h
    depsilon = 2.0 * coeffs.background_index * dn

    return SorefBennettResult(
        delta_permittivity=depsilon.tolist(),
        delta_absorption=dalpha.tolist(),
    )
