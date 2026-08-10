# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for prismo_gyptis
# Electromagnetic eigenmode component (gyptis / FEniCS).
#
# Real implementation: 2D waveguide eigenmode solve + Hellmann-Feynman
# eigen-adjoint VJP, per tickets 03 and 10. Falls back to effective-medium
# stub when gyptis/FEniCS is not installed.

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
    """Adjoint gradient pass via Hellmann-Feynman eigen-adjoint.

    For the generalized eigenproblem A x = lambda B x where lambda = kz^2,
    the eigenvalue sensitivity to subdomain permittivity epsilon_d is:

        d(lambda)/d(epsilon_d) = ( x^H (dA/d(eps_d) - lambda dB/d(eps_d)) x )
                               / ( x^H B x )

    with d(neff)/d(eps_d) chained through neff = sqrt(lambda)/k0.

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
    n = len(np.asarray(inputs.epsilon))

    try:
        dolfin = _ensure_dolfin()
        _ensure_gyptis()
    except ImportError:
        vjp["epsilon"] = np.full(n, cotangent / n)
        return vjp

    global _solve_state
    if _solve_state is None:
        raise RuntimeError(
            "vector_jacobian_product called before apply(). "
            "Run apply() first to populate the solve state."
        )

    simu = _solve_state["simu"]
    k0 = _solve_state["k0"]
    j = _solve_state["eigen_index"]

    # --- 1. Re-assemble A and B (identical to eigensolve internals) ---
    wf = simu.formulation.weak
    dv = dolfin.PETScVector()

    V = simu.formulation.space
    trial = dolfin.TrialFunction(V)
    dv.init(dolfin.as_backend_type(trial.vector()).vec())

    bcs = simu.formulation.build_boundary_conditions()

    A_mat = dolfin.PETScMatrix()
    B_mat = dolfin.PETScMatrix()
    b = dolfin.PETScVector()
    dolfin.assemble_system(wf[0], dv, bcs, A_tensor=A_mat, b_tensor=b)
    dolfin.assemble_system(wf[1], dv, bcs, A_tensor=B_mat, b_tensor=b)
    _amat = A_mat.mat()
    Bmat = B_mat.mat()

    # --- 2. Get raw eigenvector (real + imag parts, global dof order) ---
    ev_re, ev_im, rx, cx = simu.eigensolver.get_eigenpair(j)
    lam = ev_re + 1j * ev_im
    kz = np.sqrt(lam)
    neff = kz / k0

    # --- 3. Per-subdomain derivative matrices + Hellmann-Feynman ---
    u = simu.formulation.trial
    v = simu.formulation.test

    et = dolfin.as_vector([u[0], u[1], 0.0])
    ez = u[2]
    vt = dolfin.as_vector([v[0], v[1], 0.0])
    vz = v[2]
    zhat = dolfin.as_vector([0.0, 0.0, 1.0])

    dneff_sq_deps = np.zeros(len(np.asarray(inputs.epsilon)))

    for d_idx in range(len(np.asarray(inputs.epsilon))):
        domain_name = str(d_idx + 1)
        dx_d = simu.dx(domain_name)

        # M_tt = -k0^2 int e_t * v_t dx  (dA/d(eps_d))
        form_tt = -dolfin.Constant(k0 * k0) * dolfin.inner(et, vt) * dx_d
        F_tt = form_tt.real + form_tt.imag
        Mtt = dolfin.PETScMatrix()
        dolfin.assemble(F_tt, tensor=Mtt)
        for bc in bcs:
            bc.apply(Mtt)
        Mtt_mat = Mtt.mat()

        # M_zz = +k0^2 int e_z * v_z dx  (dB/d(eps_d))
        form_zz = (
            dolfin.Constant(k0 * k0)
            * dolfin.inner(ez * zhat, vz * zhat)
            * dx_d
        )
        F_zz = form_zz.real + form_zz.imag
        Mzz = dolfin.PETScMatrix()
        dolfin.assemble(F_zz, tensor=Mzz)
        for bc in bcs:
            bc.apply(Mzz)
        Mzz_mat = Mzz.mat()

        # M_d = dA/d(eps_d) - lambda * dB/d(eps_d)
        # Collapsed-real doubled-space Hermitian form:
        #   num = rx^T M_d rx + cx^T M_d cx
        num = rx.dot(Mtt_mat * rx) + cx.dot(Mtt_mat * cx)
        num -= lam * (rx.dot(Mzz_mat * rx) + cx.dot(Mzz_mat * cx))
        den = rx.dot(Bmat * rx) + cx.dot(Bmat * cx)

        dlam_deps = num / den
        dneff_deps = dlam_deps / (2.0 * k0 * kz)

        # Chain: dneff_sq/d(eps_d) = 2 * neff * dneff/d(eps_d)
        dneff_sq_deps[d_idx] = 2.0 * neff * dneff_deps

    vjp["epsilon"] = cotangent * dneff_sq_deps
    return vjp
