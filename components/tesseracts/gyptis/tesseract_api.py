# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for prismo_gyptis
# Electromagnetic eigenmode component (gyptis / FEniCS).
#
# Field-epsilon implementation (tickets 02/03): the design region carries a
# spatially-varying permittivity -- one value per DG0 design cell -- injected as
# a scalar dolfin.Function over an embedded, PML-inset patch of the silicon core.
# The forward eigensolve and the field-valued Hellmann-Feynman adjoint share a
# single two-sided SLEPc solve (spike 01 decision 3); the adjoint assembles a
# per-design-cell cotangent in one pass (decision 4). Falls back to an
# effective-medium stub when gyptis/FEniCS is not installed.

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from prismo_shared.session import SolveSessionRegistry, array_identity
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64

#
# Geometry / physics constants (spike 01)
#

WAVELENGTH: float = 1.55  # free-space wavelength, micrometres

# Silicon core in oxide; surroundings stay constant (spec: only the core is
# modulated). Defaults double as the effective-medium stub's material stack.
DEFAULT_CORE_EPSILON: float = 3.4757**2
DEFAULT_CLAD_EPSILON: float = 2.10
DEFAULT_SUBSTRATE_EPSILON: float = 2.10

_WIDTH: float = 2.0
_PML_WIDTH: tuple[float, float] = (0.5, 0.5)
_LAYER_THICKNESS: dict[str, float] = {
    "substrate": 0.35,
    "core": 0.30,
    "clad": 0.35,
}
# Refine the core so the fundamental guided mode (n_clad < neff < n_core) is
# resolved rather than only the coarse-mesh leaky mode.
_CORE_MESH_SIZE: float = 0.06

# Embedded design region: interior cells of the core layer, inset from the
# layer's x-extent (+/- _WIDTH/2) and its y-edges so it never touches a PML.
_DESIGN_HALF_WIDTH: float = 0.6
_DESIGN_Y_INSET: float = 0.05


#
# Schemas
#


class InputSchema(BaseModel):
    """Inputs to the gyptis field-epsilon eigenmode solve.

    Attributes:
        operation: ``"solve"`` runs the eigenmode solve. ``"design_cell_centroids"``
            returns the static design-cell geometry needed to construct a mesh
            transfer operator.
        design_epsilon: Relative permittivity per design cell -- the modulated
            silicon on the embedded design region, one value per DG0 cell (order
            given by :func:`design_cell_centroids`). This is the only
            differentiated input.
        core_epsilon: Background silicon permittivity for the core cells outside
            the design region (constant).
        clad_epsilon: Cladding (oxide) permittivity (constant).
        substrate_epsilon: Substrate (oxide) permittivity (constant).
    """

    operation: Literal["solve", "design_cell_centroids"] = "solve"
    design_epsilon: Differentiable[Array[(None,), Float64]] | None = None
    core_epsilon: float = DEFAULT_CORE_EPSILON
    clad_epsilon: float = DEFAULT_CLAD_EPSILON
    substrate_epsilon: float = DEFAULT_SUBSTRATE_EPSILON


class OutputSchema(BaseModel):
    """Outputs of the gyptis eigenmode solve.

    Attributes:
        neff_sq: Squared effective index of the tracked mode
            (neff_sq = kz^2 / k0^2, where kz is the propagation constant
            and k0 is the free-space wavenumber).
        design_cell_centroids: Static design-cell centroids returned by the
            inspection operation, in ``design_epsilon`` order.
    """

    neff_sq: Differentiable[Array[(), Float64]] | None = None
    design_cell_centroids: Array[(None, 2), Float64] | None = None


#
# Module-level state
#

# The forward solve, behind the fixed apply/vjp endpoints. apply() runs the
# single two-sided eigensolve and stores its whole eigenstate here; the adjoint
# retrieves the session whose inputs match and assembles the field sensitivity
# from that state without re-solving. No scope is passed, so the registry keeps
# only the most-recent forward.
_session_registry = SolveSessionRegistry()


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


def _solve_identity(
    design_epsilon: np.ndarray,
    core_epsilon: float,
    clad_epsilon: float,
    substrate_epsilon: float,
) -> tuple[Any, ...]:
    """Hashable fingerprint of a forward solve's inputs (field + constants)."""
    return (
        *array_identity(design_epsilon),
        float(core_epsilon),
        float(clad_epsilon),
        float(substrate_epsilon),
    )


def _build_waveguide(
    core_epsilon: float,
    clad_epsilon: float,
    substrate_epsilon: float,
) -> tuple[Any, float]:
    """Build the layered substrate/core/clad waveguide with a refined core.

    Returns:
        Tuple of (Waveguide simulation, free-space wavenumber k0).
    """
    gyptis = _ensure_gyptis()
    _ensure_dolfin()

    k0 = 2.0 * np.pi / WAVELENGTH
    geom = gyptis.geometry.LayeredBoxPML2D(
        _WIDTH, thicknesses=dict(_LAYER_THICKNESS), pml_width=_PML_WIDTH
    )
    geom.set_size("core", _CORE_MESH_SIZE)
    geom.build()

    eps_bg = {
        "substrate": float(substrate_epsilon),
        "core": float(core_epsilon),
        "clad": float(clad_epsilon),
    }
    simu = gyptis.Waveguide(geom, epsilon=eps_bg, wavenumber=k0)
    return simu, k0


def _design_mask(simu: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return (DG0 space, cell centroids, boolean design-cell mask).

    Design cells are interior core cells whose centroid lies in the embedded,
    PML-inset design box (spike 01 decision 1).
    """
    dolfin = _ensure_dolfin()
    mesh = simu.formulation.function_space.mesh()
    y0 = simu.geometry.y_position["core"]
    y_lo, y_hi = y0, y0 + simu.geometry.thicknesses["core"]

    dg0 = dolfin.FunctionSpace(mesh, "DG", 0)
    coords = dg0.tabulate_dof_coordinates().reshape(-1, 2)
    x, y = coords[:, 0], coords[:, 1]
    mask = (
        (np.abs(x) < _DESIGN_HALF_WIDTH)
        & (y > y_lo + _DESIGN_Y_INSET)
        & (y < y_hi - _DESIGN_Y_INSET)
    )
    return dg0, coords, mask


def _make_core_field(
    dg0: Any, mask: np.ndarray, core_background: float, design_values: np.ndarray
) -> Any:
    """Scalar DG0 Function: ``design_values`` on masked cells, background elsewhere."""
    dolfin = _ensure_dolfin()
    fn = dolfin.Function(dg0)
    vals = np.full(dg0.dim(), float(core_background))
    vals[mask] = design_values
    fn.vector().set_local(vals)
    fn.vector().apply("insert")
    return fn


def design_cell_centroids(
    core_epsilon: float = DEFAULT_CORE_EPSILON,
    clad_epsilon: float = DEFAULT_CLAD_EPSILON,
    substrate_epsilon: float = DEFAULT_SUBSTRATE_EPSILON,
) -> np.ndarray:
    """Centroids ``(n_design, 2)`` of the design cells, in ``design_epsilon`` order.

    The mask depends only on the fixed geometry, not on the permittivity values,
    so the pipeline can call this once to size and build its mesh-transfer
    operator (ticket 04). Requires gyptis/FEniCS.
    """
    simu, _k0 = _build_waveguide(core_epsilon, clad_epsilon, substrate_epsilon)
    _dg0, coords, mask = _design_mask(simu)
    return coords[mask]


#
# Two-sided eigensolve (shared by forward + adjoint; spike 01 decision 3)
#


def _assemble_AB(simu: Any, eps_core_field: Any) -> tuple[Any, Any]:
    """Overwrite the core property with a DG0 field; assemble real A, B matrices."""
    dolfin = _ensure_dolfin()
    from gyptis.complex import Constant, dot

    form = simu.formulation
    # Rebuild the property dict from the scalar background first so the PMLs stay
    # matched to the scalar core (spike 01 decision 2), then overwrite the core.
    form._epsilon = form.epsilon.as_property(dim=3)
    form._epsilon["core"] = eps_core_field

    wf = form.weak
    zero = Constant((0.0, 0.0, 0.0))
    dv = dot(zero, form.test) * form.dx
    dv = dv.real + dv.imag
    bcs = form.build_boundary_conditions()

    A_mat, B_mat = dolfin.PETScMatrix(), dolfin.PETScMatrix()
    b = dolfin.PETScVector()
    dolfin.assemble_system(wf[0], dv, bcs, A_tensor=A_mat, b_tensor=b)
    dolfin.assemble_system(wf[1], dv, A_tensor=B_mat, b_tensor=b)
    return A_mat.mat(), B_mat.mat()


def _two_sided_solver(A: Any, B: Any, target: float, n: int = 12) -> tuple[Any, int]:
    """Two-sided shift-invert SLEPc solve; returns (solver, n_converged)."""
    from slepc4py import SLEPc

    solver = SLEPc.EPS().create(A.getComm())
    solver.setOperators(A, B)
    solver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    solver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    solver.setTarget(target)
    st = solver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target)
    solver.setDimensions(n)
    solver.setTolerances(1e-7)
    solver.setTwoSided(True)
    solver.solve()
    nconv = solver.getConverged()
    if nconv == 0:
        raise RuntimeError("two-sided SLEPc eigensolve did not converge")
    return solver, nconv


def _select_index(
    solver: Any,
    nconv: int,
    k0: float,
    core_epsilon: float,
    clad_epsilon: float,
    ref_lam: complex | None,
) -> int:
    """Index of the tracked eigenpair.

    When ``ref_lam`` is given (adjoint / finite-difference re-solves) the mode is
    tracked by nearest eigenvalue, so forward and adjoint validate one mode. On
    the base forward solve the fundamental *guided* mode is selected: the
    largest-neff eigenpair whose neff lies in the physical window
    ``(sqrt(clad), sqrt(core))``. Falls back to nearest-target when the mesh
    resolves no guided mode (a leaky mode; gradient correctness is unaffected).
    """
    if ref_lam is not None:
        dists = [abs(solver.getEigenvalue(i) - ref_lam) for i in range(nconv)]
        return int(np.argmin(dists))

    n_core = np.sqrt(core_epsilon)
    n_clad = np.sqrt(clad_epsilon)
    best_idx, best_neff = None, -np.inf
    for i in range(nconv):
        neff = float(np.real(np.sqrt(solver.getEigenvalue(i))) / k0)
        if n_clad < neff < n_core and neff > best_neff:
            best_idx, best_neff = i, neff
    if best_idx is not None:
        return best_idx

    target = complex(core_epsilon * k0 * k0, 0.0)
    dists = [abs(solver.getEigenvalue(i) - target) for i in range(nconv)]
    return int(np.argmin(dists))


def _solve_state(
    simu: Any,
    eps_core_field: Any,
    k0: float,
    core_epsilon: float,
    clad_epsilon: float,
    ref_lam: complex | None = None,
) -> dict[str, Any]:
    """Assemble + two-sided solve; return the tracked eigenpair state."""
    A, B = _assemble_AB(simu, eps_core_field)
    target = core_epsilon * k0 * k0
    solver, nconv = _two_sided_solver(A, B, target)
    idx = _select_index(solver, nconv, k0, core_epsilon, clad_epsilon, ref_lam)

    rx, cx = A.createVecRight(), A.createVecRight()
    ry, cy = A.createVecRight(), A.createVecRight()
    lam = solver.getEigenpair(idx, rx, cx)
    solver.getLeftEigenvector(idx, ry, cy)
    kz = np.sqrt(lam)
    neff = float(np.real(kz / k0))
    return {
        "A": A, "B": B, "lam": lam, "kz": kz, "neff": neff,
        "rx": rx, "cx": cx, "ry": ry, "cy": cy,
    }


def _lr_product(matrix: Any, state: dict[str, Any]) -> complex:
    """y^H matrix x for the complex left/right eigenvectors of the real system."""
    rx, cx, ry, cy = state["rx"], state["cx"], state["ry"], state["cy"]
    m_rx, m_cx = matrix.createVecRight(), matrix.createVecRight()
    matrix.mult(rx, m_rx)
    matrix.mult(cx, m_cx)
    return ry.dot(m_rx) + cy.dot(m_cx) + 1j * (ry.dot(m_cx) - cy.dot(m_rx))


#
# Single-pass field sensitivity (spike 01 decision 4; the ticket-03 kernel)
#


def _field_numerator(form: Any, dg0: Any, state: dict[str, Any]) -> np.ndarray:
    """Per-cell complex numerator ``y^H (dA/deps - lam dB/deps) x`` in one pass.

    Reproduces the matrix-level left/right product with a DG0 test function w
    standing in for the per-cell epsilon direction. The eigenvector of the real
    doubled system is complex ``(rx + i cx)``, so ``y^H M x`` is the 4-term
    combination below; each term assembles the epsilon-derivative density against
    ``w`` via ``assemble(.real) + assemble(.imag)``.
    """
    dolfin = _ensure_dolfin()
    from gyptis.complex import Complex, Constant, inner, vector
    from gyptis.utils.helpers import array2function

    k0_sq = Constant((2.0 * np.pi / WAVELENGTH) ** 2)
    fs = form.function_space
    w = dolfin.TestFunction(dg0)

    def phys(vec: Any) -> Any:
        f = array2function(vec.getArray(), fs)
        return vector(Complex([f[0], f[1], f[2]], [f[3], f[4], f[5]]))

    Xr, Xc = phys(state["rx"]), phys(state["cx"])
    Yr, Yc = phys(state["ry"]), phys(state["cy"])

    def assemble_ri(cform: Any) -> np.ndarray:
        re = np.asarray(dolfin.assemble(cform.real).get_local())
        im = np.asarray(dolfin.assemble(cform.imag).get_local())
        return re + im

    def dens_A(P: Any, Q: Any) -> np.ndarray:
        et_P = vector([P[0], P[1], 0.0])
        et_Q = vector([Q[0], Q[1], 0.0])
        return assemble_ri(-k0_sq * inner(et_P, et_Q) * w * form.dx)

    def dens_B(P: Any, Q: Any) -> np.ndarray:
        return assemble_ri(k0_sq * (P[2] * Q[2]) * w * form.dx)

    def lrp_form(dens: Any) -> np.ndarray:
        return (
            dens(Xr, Yr) + dens(Xc, Yc) + 1j * (dens(Xc, Yr) - dens(Xr, Yc))
        )

    return lrp_form(dens_A) - state["lam"] * lrp_form(dens_B)


def _field_sensitivity(
    simu: Any, dg0: Any, k0: float, state: dict[str, Any]
) -> np.ndarray:
    """dneff_sq / d(eps_core) per DG0 cell (a field), in a single assembly pass."""
    num_cell = _field_numerator(simu.formulation, dg0, state)
    denom = _lr_product(state["B"], state)
    if abs(denom) == 0.0:
        raise RuntimeError("left/right eigenvectors have zero B-inner product")
    dlam = num_cell / denom
    dneff = np.real(dlam / (2.0 * k0 * state["kz"]))
    return 2.0 * state["neff"] * dneff


#
# Required endpoint
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward field-epsilon eigenmode solve.

    Args:
        inputs: Design-region permittivity field plus constant surroundings.

    Returns:
        Squared effective index of the tracked mode.
    """
    if inputs.operation == "design_cell_centroids":
        try:
            centroids = design_cell_centroids(
                inputs.core_epsilon, inputs.clad_epsilon, inputs.substrate_epsilon
            )
        except ImportError as exc:
            raise RuntimeError(
                "design-cell centroids require a gyptis/FEniCS backend"
            ) from exc
        return OutputSchema(design_cell_centroids=centroids)

    if inputs.design_epsilon is None:
        raise ValueError("design_epsilon is required for a gyptis solve")
    design = np.asarray(inputs.design_epsilon, dtype=float)
    if design.ndim != 1 or design.size == 0:
        raise ValueError("design_epsilon must contain at least one design cell")

    try:
        _ensure_dolfin()
        _ensure_gyptis()
    except ImportError:
        # Effective-medium fallback: scalar output shaped from the field input.
        return OutputSchema(neff_sq=float(np.mean(design)))

    simu, k0 = _build_waveguide(
        inputs.core_epsilon, inputs.clad_epsilon, inputs.substrate_epsilon
    )
    dg0, _coords, mask = _design_mask(simu)
    n_design = int(mask.sum())
    if design.size != n_design:
        raise ValueError(
            f"design_epsilon has {design.size} values but the design region has "
            f"{n_design} cells; size it with design_cell_centroids()"
        )

    field = _make_core_field(dg0, mask, inputs.core_epsilon, design)
    state = _solve_state(
        simu, field, k0, inputs.core_epsilon, inputs.clad_epsilon
    )

    _session_registry.open(
        _solve_identity(
            design,
            inputs.core_epsilon,
            inputs.clad_epsilon,
            inputs.substrate_epsilon,
        ),
        state={"simu": simu, "dg0": dg0, "mask": mask, "k0": k0, "eigen": state},
    )

    neff = state["neff"]
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
    """Field-valued Hellmann-Feynman adjoint.

    Returns one cotangent per design cell: the sensitivity of neff_sq to the
    design-region permittivity field, assembled in a single pass from the
    eigenstate the preceding ``apply`` stored (no re-solve). Chained through
    ``neff = sqrt(lambda) / k0``.

    Args:
        inputs: Same InputSchema as the preceding apply() call.
        vjp_inputs: Input fields to compute cotangents for ({"design_epsilon"}).
        vjp_outputs: Output fields the cotangent was taken w.r.t. ({"neff_sq"}).
        cotangent_vector: Cotangent on output fields, e.g. ``{"neff_sq": v}``.

    Returns:
        ``{"design_epsilon": dL/d(design_epsilon)}`` when requested.
    """
    vjp: dict[str, npt.ArrayLike] = {}
    if "design_epsilon" not in vjp_inputs or "neff_sq" not in vjp_outputs:
        return vjp

    cotangent = float(np.asarray(cotangent_vector["neff_sq"]))
    design = np.asarray(inputs.design_epsilon, dtype=float)
    if design.ndim != 1 or design.size == 0:
        raise ValueError("design_epsilon must contain at least one design cell")

    try:
        _ensure_dolfin()
        _ensure_gyptis()
    except ImportError:
        n = design.size
        vjp["design_epsilon"] = np.full(n, cotangent / n)
        return vjp

    session = _session_registry.match(
        _solve_identity(
            design,
            inputs.core_epsilon,
            inputs.clad_epsilon,
            inputs.substrate_epsilon,
        )
    )
    if session is None:
        if _session_registry.has_any():
            raise RuntimeError(
                "vector_jacobian_product inputs differ from the preceding "
                "apply() call. Run apply() with these inputs first."
            )
        raise RuntimeError(
            "vector_jacobian_product called before apply(). "
            "Run apply() first to populate the solve state."
        )

    simu = session.state["simu"]
    dg0 = session.state["dg0"]
    mask = session.state["mask"]
    k0 = session.state["k0"]
    eigen = session.state["eigen"]

    dneff_sq_cell = _field_sensitivity(simu, dg0, k0, eigen)
    grad_design = dneff_sq_cell[mask]
    vjp["design_epsilon"] = cotangent * grad_design
    return vjp
