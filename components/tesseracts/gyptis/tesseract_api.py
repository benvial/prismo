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
        "epsilon": epsilon.copy(),
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

    For the non-Hermitian generalized eigenproblem A x = lambda B x where
    lambda = kz^2, the eigenvalue sensitivity to subdomain permittivity
    epsilon_d is:

        d(lambda)/d(epsilon_d) = ( y^H (dA/d(eps_d) - lambda dB/d(eps_d)) x )
                               / ( y^H B x )

    where x and y are the right and left eigenvectors, respectively.

    The result is chained through neff = sqrt(lambda)/k0.

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
    if not np.array_equal(epsilon, _solve_state["epsilon"]):
        raise RuntimeError(
            "vector_jacobian_product inputs differ from the preceding apply() call. "
            "Run apply() with these inputs first."
        )
    k0 = _solve_state["k0"]

    # Reassemble exactly as gyptis' eigensolve implementation does, then use
    # slepc4py directly because DOLFIN does not expose left eigenvectors.
    from gyptis.complex import Constant, dot, inner, vector
    from gyptis.materials import Coefficient
    from slepc4py import SLEPc

    wf = simu.formulation.weak
    zero = Constant((0.0, 0.0, 0.0))
    dv = dot(zero, simu.formulation.test) * simu.formulation.dx
    dv = dv.real + dv.imag

    bcs = simu.formulation.build_boundary_conditions()

    A_mat = dolfin.PETScMatrix()
    B_mat = dolfin.PETScMatrix()
    b = dolfin.PETScVector()
    dolfin.assemble_system(wf[0], dv, bcs, A_tensor=A_mat, b_tensor=b)
    dolfin.assemble_system(wf[1], dv, A_tensor=B_mat, b_tensor=b)
    A = A_mat.mat()
    B = B_mat.mat()

    eigensolver = SLEPc.EPS().create(A.getComm())
    eigensolver.setOperators(A, B)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eigensolver.setTarget(k0 * k0)
    spectral_transform = eigensolver.getST()
    spectral_transform.setType(SLEPc.ST.Type.SINVERT)
    spectral_transform.setShift(k0 * k0)
    eigensolver.setDimensions(8)
    eigensolver.setTolerances(1e-6)
    eigensolver.setTwoSided(True)
    eigensolver.solve()
    if eigensolver.getConverged() == 0:
        raise RuntimeError("two-sided SLEPc eigensolve did not converge")

    rx = A.createVecRight()
    cx = A.createVecRight()
    ry = A.createVecRight()
    cy = A.createVecRight()
    lam = eigensolver.getEigenpair(0, rx, cx)
    eigensolver.getLeftEigenvector(0, ry, cy)
    kz = np.sqrt(lam)
    neff = float(np.real(kz / k0))

    def left_right_product(matrix: Any) -> complex:
        matrix_rx = matrix.createVecRight()
        matrix_cx = matrix.createVecRight()
        matrix.mult(rx, matrix_rx)
        matrix.mult(cx, matrix_cx)
        return (
            ry.dot(matrix_rx)
            + cy.dot(matrix_cx)
            + 1j * (ry.dot(matrix_cx) - cy.dot(matrix_rx))
        )

    denominator = left_right_product(B)
    if abs(denominator) == 0.0:
        raise RuntimeError("left/right eigenvectors have zero B-inner product")

    u = simu.formulation.trial
    v = simu.formulation.test

    et = vector([u[0], u[1], 0.0])
    ez = u[2]
    vt = vector([v[0], v[1], 0.0])
    vz = v[2]
    zhat = vector([0.0, 0.0, 1.0])
    epsilon_coefficient = simu.formulation.epsilon
    pml_domains = {pml.applied_domain for pml in epsilon_coefficient.pmls}
    physical_domains = [
        name for name in epsilon_coefficient.dict if name not in pml_domains
    ]

    dneff_sq_deps = np.zeros(len(np.asarray(inputs.epsilon)))

    for d_idx in range(len(np.asarray(inputs.epsilon))):
        domain_name = f"domain_{d_idx + 1}"
        unit_materials = {
            name: float(name == domain_name) for name in physical_domains
        }
        epsilon_derivative = Coefficient(
            unit_materials,
            geometry=epsilon_coefficient.geometry,
            pmls=epsilon_coefficient.pmls,
            dim=epsilon_coefficient.dim,
            degree=epsilon_coefficient.degree,
            element=epsilon_coefficient.element,
        ).as_property(dim=3)
        affected_domains = [
            domain_name,
            *[
                pml.applied_domain
                for pml in epsilon_coefficient.pmls
                if pml.matched_domain == domain_name
            ],
        ]

        # Each physical epsilon also controls its matched PML tensors.
        # Boundary-condition contributions are parameter-independent.
        dA_form = 0
        dB_form = 0
        for region in affected_domains:
            dA_form += (
                -Constant(k0 * k0)
                * inner(epsilon_derivative[region] * et, vt)
                * simu.dx(region)
            )
            dB_form += (
                Constant(k0 * k0)
                * inner(
                    epsilon_derivative[region] * (ez * zhat),
                    vz * zhat,
                )
                * simu.dx(region)
            )

        dA = dolfin.PETScMatrix()
        dB = dolfin.PETScMatrix()
        dolfin.assemble(dA_form.real + dA_form.imag, tensor=dA)
        dolfin.assemble(dB_form.real + dB_form.imag, tensor=dB)

        dlam_deps = (
            left_right_product(dA.mat())
            - lam * left_right_product(dB.mat())
        ) / denominator
        dneff_deps = np.real(dlam_deps / (2.0 * k0 * kz))
        dneff_sq_deps[d_idx] = 2.0 * neff * dneff_deps

    vjp["epsilon"] = cotangent * dneff_sq_deps
    return vjp
