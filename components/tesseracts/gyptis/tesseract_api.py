# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for prismo_gyptis
# Electromagnetic eigenmode component (gyptis / FEniCS).
#
# Field-epsilon implementation (tickets 02/03): the design region carries a
# spatially-varying permittivity -- one value per DG0 design cell -- injected as
# a scalar dolfin.Function over the silicon rib interior. The forward eigensolve
# and the field-valued Hellmann-Feynman adjoint share a single two-sided SLEPc
# solve (spike 01 decision 3); the adjoint assembles a per-design-cell cotangent
# in one pass (decision 4). A missing gyptis/FEniCS backend is a hard error --
# there is no physics-free effective-medium fallback that would fabricate an
# effective index (rethink ticket 04).
#
# Unified mesh (rethink ticket 05): the geometry is the single source of truth
# for BOTH solvers. ``RibBoxPML2D`` extends gyptis ``LayeredBoxPML2D`` to emit a
# real SOI rib cross-section -- rib + slab + oxide clad/substrate + a matched PML
# frame + ``contact_anode``/``contact_cathode`` lines on the slab shoulders, with
# ``silicon``/``oxide`` surface groups. gyptis eigensolves on the full domain and
# modulates only the rib interior; ChargeTransport reads the same ``.msh``
# restricted to the silicon subdomain. The ``write_mesh`` operation emits that
# shared ``.msh`` (as text, so the host -- not this container's uid -- owns the
# file) for the ChargeTransport container and the mesh-transfer operator.

from pathlib import Path
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

# Silicon rib in oxide; surroundings stay constant (spec: only the rib interior
# is modulated). The oxide permittivity is n_ox**2 with n_ox = 1.444 (matching
# the ChargeTransport/shared-mesh oxide index), reconciling the stray clad =
# substrate = 2.10 the slab-only model carried (audit P3 / ticket 05).
N_SILICON: float = 3.4757
N_OXIDE: float = 1.444
DEFAULT_CORE_EPSILON: float = N_SILICON**2  # ~12.08
DEFAULT_OXIDE_EPSILON: float = N_OXIDE**2  # ~2.085
DEFAULT_CLAD_EPSILON: float = DEFAULT_OXIDE_EPSILON
DEFAULT_SUBSTRATE_EPSILON: float = DEFAULT_OXIDE_EPSILON

# SOI rib cross-section, micrometres (spike 01 recipe). A full-width silicon slab
# layer keeps ``pml_x_slab`` matched to silicon; the rib is the only embedded
# surgery and, inset from the x-PML, never touches one. Layer stack bottom->top:
# substrate (oxide) / slab (silicon) / rib band (oxide + embedded silicon rib) /
# clad (oxide).
_WIDTH: float = 2.0
_PML_WIDTH: tuple[float, float] = (0.5, 0.5)
_LAYER_THICKNESS: dict[str, float] = {
    "substrate": 0.35,
    "slab": 0.10,
    "rib": 0.22,
    "clad": 0.35,
}
_RIB_HALF_WIDTH: float = 0.25  # 500 nm rib, centred at x=0
_CONTACT_OFFSET: float = 0.20  # gap from rib edge to the near contact edge
_CONTACT_HALF_WIDTH: float = 0.025  # 50 nm contact line half-width
# Refine the rib so the fundamental guided mode (n_clad < neff < n_core) is
# resolved rather than only the coarse-mesh leaky mode.
_CORE_MESH_SIZE: float = 0.04

# Design region: the silicon rib interior. Every rib cell is modulated; the rib
# is inset from the x-PML by construction and sits in the rib band away from the
# y-PMLs, so no design cell ever touches a PML.
_DESIGN_Y_INSET: float = 0.0


#
# Schemas
#


class InputSchema(BaseModel):
    """Inputs to the gyptis field-epsilon eigenmode solve.

    Attributes:
        operation: ``"solve"`` runs the eigenmode solve. ``"design_cell_centroids"``
            returns the static design-cell geometry needed to construct a mesh
            transfer operator. ``"write_mesh"`` emits the shared unified ``.msh``
            (as text) for the ChargeTransport container and the transfer operator.
        design_epsilon: Relative permittivity per design cell -- the modulated
            silicon on the rib interior, one value per DG0 cell (order given by
            :func:`design_cell_centroids`). This is the only differentiated input.
        core_epsilon: Background silicon permittivity for the rib cells outside
            the design region and the slab (constant).
        clad_epsilon: Cladding (oxide) permittivity (constant).
        substrate_epsilon: Substrate (oxide) permittivity (constant).
    """

    operation: Literal["solve", "design_cell_centroids", "write_mesh"] = "solve"
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
        mesh_text: The shared unified ``.msh`` file content (ASCII), returned by
            the ``write_mesh`` operation so the host can persist it for the
            ChargeTransport container and the mesh-transfer operator.
        design_cell_vertices: ``(n_design, 3, 2)`` vertex coordinates of each
            design cell, in ``design_epsilon`` order, returned by ``write_mesh``.
            The host matches these to shared-mesh node indices to assemble the
            exact node->DG0-cell restriction operator (ticket 05).
    """

    neff_sq: Differentiable[Array[(), Float64]] | None = None
    design_cell_centroids: Array[(None, 2), Float64] | None = None
    mesh_text: str | None = None
    design_cell_vertices: Array[(None, 3, 2), Float64] | None = None


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


def _rib_box_pml_2d() -> Any:
    """Build the ``RibBoxPML2D`` class extending gyptis ``LayeredBoxPML2D``.

    Deferred behind a function so the gyptis import stays lazy (the class body
    subclasses a gyptis type). Ported from the ticket-01 spike: a silicon rib is
    embedded in the oxide rib band, inset from the x-PML, with Ohmic contact
    lines split into the slab-top interface on each shoulder.
    """
    gyptis = _ensure_gyptis()

    class RibBoxPML2D(gyptis.geometry.LayeredBoxPML2D):  # type: ignore[misc]
        """LayeredBoxPML2D with an embedded silicon rib + contact lines."""

        def add_rib(self) -> None:
            # Embed a silicon rectangle in the centre of the oxide "rib" band,
            # inset from +/- WIDTH/2 (the x-PML). The oxide remainder of the band
            # keeps the "rib" material tag so pml_x_rib matches oxide; the centred
            # silicon surface is tagged "rib_silicon".
            y0_rib = self.y_position["rib"]
            rib = self.add_rectangle(
                -_RIB_HALF_WIDTH, y0_rib, 0, 2 * _RIB_HALF_WIDTH, _LAYER_THICKNESS["rib"]
            )
            band = self.subdomains_entities["surfaces"]["rib"]
            self.fragment(rib, band)  # conform the mesh across the rib/oxide edge
            self.synchronize()
            e = 1e-6
            sils = self.model.getEntitiesInBoundingBox(
                -_RIB_HALF_WIDTH - e, y0_rib - e, -e,
                _RIB_HALF_WIDTH + e, y0_rib + _LAYER_THICKNESS["rib"] + e, e, 2,
            )
            rib_tags = [t for (_d, t) in sils]
            allband = self.model.getEntitiesInBoundingBox(
                -self.width / 2 - e, y0_rib - e, -e,
                self.width / 2 + e, y0_rib + _LAYER_THICKNESS["rib"] + e, e, 2,
            )
            oxide_tags = [t for (_d, t) in allband if t not in rib_tags]
            self.add_physical(rib_tags, "rib_silicon")
            self.add_physical(oxide_tags, "rib")

        def add_contacts(self) -> None:
            # Ohmic contacts are dim-1 lines on the slab-top interface (bottom of
            # the rib band), one per shoulder, offset from the rib edge. Splitting
            # the existing interface curve (0d-into-1d fragment) never renumbers
            # surfaces, so the material physical groups survive; embedding each
            # contact into both neighbouring surfaces makes their triangles share
            # its nodes so ExtendableGrids retains it for the CT silicon subgrid.
            e = 1e-6
            y = self.y_position["rib"]  # slab top
            xc_l = -_RIB_HALF_WIDTH - _CONTACT_OFFSET - _CONTACT_HALF_WIDTH
            xc_r = _RIB_HALF_WIDTH + _CONTACT_OFFSET + _CONTACT_HALF_WIDTH
            contacts = {"contact_anode": xc_l, "contact_cathode": xc_r}

            pts = []
            for xc in contacts.values():
                pts.append(self.add_point(xc - _CONTACT_HALF_WIDTH, y, 0))
                pts.append(self.add_point(xc + _CONTACT_HALF_WIDTH, y, 0))
            self.synchronize()

            iface = self.model.getEntitiesInBoundingBox(
                -self.width / 2 - e, y - e, -e, self.width / 2 + e, y + e, e, 1,
            )
            self.occ.fragment([(0, p) for p in pts], iface)
            self.synchronize()

            for name, xc in contacts.items():
                segs = self.model.getEntitiesInBoundingBox(
                    xc - _CONTACT_HALF_WIDTH - e, y - e, -e,
                    xc + _CONTACT_HALF_WIDTH + e, y + e, e, 1,
                )
                tags = [t for (_d, t) in segs]
                self.add_physical(tags, name, dim=1)
                nearby = self.model.getEntitiesInBoundingBox(
                    xc - _CONTACT_HALF_WIDTH - e, y - e, -e,
                    xc + _CONTACT_HALF_WIDTH + e, y + e, e, 2,
                )
                for tag in tags:
                    for _d, surface in nearby:
                        self.model.mesh.embed(1, tag, 2, surface)

    return RibBoxPML2D


def _build_waveguide(
    core_epsilon: float,
    clad_epsilon: float,
    substrate_epsilon: float,
) -> tuple[Any, float]:
    """Build the unified SOI rib waveguide (rib + slab + oxide + PML + contacts).

    The geometry is the single source of truth shared with ChargeTransport
    (ticket 05). ``build()`` meshes and writes the ``.msh`` into a per-build temp
    directory (uid-9999-writable), recorded on ``geom.msh_file`` for the
    ``write_mesh`` operation to read back.

    Returns:
        Tuple of (Waveguide simulation, free-space wavenumber k0).
    """
    import tempfile

    gyptis = _ensure_gyptis()
    _ensure_dolfin()

    k0 = 2.0 * np.pi / WAVELENGTH
    rib_cls = _rib_box_pml_2d()
    out_dir = tempfile.mkdtemp(prefix="prismo_unified_mesh_")
    geom = rib_cls(
        _WIDTH,
        thicknesses=dict(_LAYER_THICKNESS),
        pml_width=_PML_WIDTH,
        data_dir=out_dir,
        mesh_name="unified_mesh.msh",
        binary_mesh=False,
    )
    geom.add_rib()
    geom.add_contacts()
    geom.set_size("rib_silicon", _CORE_MESH_SIZE)
    geom.build()

    # Oxide everywhere except the two silicon regions (slab + rib interior). The
    # slab is background silicon; only the rib interior is modulated.
    eps_bg = {
        "substrate": float(substrate_epsilon),
        "slab": float(core_epsilon),
        "rib": float(clad_epsilon),
        "rib_silicon": float(core_epsilon),
        "clad": float(clad_epsilon),
    }
    simu = gyptis.Waveguide(geom, epsilon=eps_bg, wavenumber=k0)
    return simu, k0


def _design_mask(simu: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return (DG0 space, cell centroids, boolean design-cell mask).

    Design cells are the silicon rib interior: DG0 cells whose centroid lies in
    the rib rectangle. The rib is inset from the x-PML by construction and sits
    in the rib band away from the y-PMLs, so no design cell touches a PML
    (ticket 05).
    """
    dolfin = _ensure_dolfin()
    mesh = simu.formulation.function_space.mesh()
    y0 = simu.geometry.y_position["rib"]
    y_lo, y_hi = y0, y0 + simu.geometry.thicknesses["rib"]

    dg0 = dolfin.FunctionSpace(mesh, "DG", 0)
    coords = dg0.tabulate_dof_coordinates().reshape(-1, 2)
    x, y = coords[:, 0], coords[:, 1]
    tol = 1e-9
    mask = (
        (np.abs(x) < _RIB_HALF_WIDTH - tol)
        & (y > y_lo + _DESIGN_Y_INSET + tol)
        & (y < y_hi - _DESIGN_Y_INSET - tol)
    )
    return dg0, coords, mask


def _make_rib_field(
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


def _design_cell_vertices(simu: Any, dg0: Any, mask: np.ndarray) -> np.ndarray:
    """``(n_design, 3, 2)`` vertex coordinates of each design cell.

    In ``design_epsilon`` (DG0-dof) order. On the shared mesh the host matches
    these coordinates to gmsh node indices to assemble the exact node->DG0-cell
    restriction operator (ticket 05); a design cell is a triangle whose three
    vertices are shared-mesh nodes.
    """
    dolfin = _ensure_dolfin()
    mesh = dg0.mesh()
    mesh_coords = mesh.coordinates()
    cells = mesh.cells()
    dofmap = dg0.dofmap()
    # DG0 has one dof per cell; invert the cell->dof map so a dof (a design cell,
    # in design_epsilon order) resolves to its mesh cell and hence its vertices.
    dof_to_cell = np.empty(dg0.dim(), dtype=np.int64)
    for cell in dolfin.cells(mesh):
        dof_to_cell[dofmap.cell_dofs(cell.index())[0]] = cell.index()

    design_dofs = np.nonzero(mask)[0]
    verts = np.empty((design_dofs.size, 3, 2), dtype=float)
    for row, dof in enumerate(design_dofs):
        verts[row] = mesh_coords[cells[dof_to_cell[dof]]]
    return verts


def write_mesh(
    core_epsilon: float = DEFAULT_CORE_EPSILON,
    clad_epsilon: float = DEFAULT_CLAD_EPSILON,
    substrate_epsilon: float = DEFAULT_SUBSTRATE_EPSILON,
) -> tuple[str, np.ndarray]:
    """Emit the shared unified ``.msh`` text and the design-cell vertices.

    The geometry is the single source of truth for both solvers; this operation
    hands the host the mesh (so it -- not this container's uid -- writes the file
    the ChargeTransport container reads) plus the per-design-cell vertex
    coordinates the host needs to build the exact mesh-transfer operator.
    Requires gyptis/FEniCS.
    """
    simu, _k0 = _build_waveguide(core_epsilon, clad_epsilon, substrate_epsilon)
    _dg0, _coords, mask = _design_mask(simu)
    vertices = _design_cell_vertices(simu, _dg0, mask)
    mesh_text = Path(simu.geometry.msh_file).read_text()
    return mesh_text, vertices


#
# Two-sided eigensolve (shared by forward + adjoint; spike 01 decision 3)
#


def _pin_orphan_dofs(A: Any, B: Any) -> None:
    """Give a unit diagonal to any DOF whose A and B rows are both empty.

    The shared unified mesh carries Ohmic contact lines embedded into the silicon
    surfaces so ChargeTransport can retain them (ticket 05); a handful of the
    edge DOFs those embedded internal lines introduce couple to nothing in the
    curl-curl formulation, leaving zero rows in A and B. The shift-invert
    factorization ``(A - sigma B)`` then hits a zero pivot. Pinning each such DOF
    to ``A[i,i] = B[i,i] = 1`` gives it the isolated eigenvalue ``1`` -- far
    outside the guided window ``(n_clad, n_core)`` at this wavenumber, so it is
    never the tracked mode and never enters the field sensitivity.
    """
    rstart, rend = A.getOwnershipRange()
    orphans = [
        i
        for i in range(rstart, rend)
        if A.getRow(i)[0].size == 0 and B.getRow(i)[0].size == 0
    ]
    if not orphans:
        return
    import petsc4py.PETSc as PETSc

    for mat in (A, B):
        # Orphan rows may have no preallocated diagonal; allow the new entry.
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        for i in orphans:
            mat.setValue(i, i, 1.0)
        mat.assemble()


def _assemble_AB(simu: Any, eps_core_field: Any) -> tuple[Any, Any]:
    """Overwrite the rib property with a DG0 field; assemble real A, B matrices."""
    dolfin = _ensure_dolfin()
    from gyptis.complex import Constant, dot

    form = simu.formulation
    # Rebuild the property dict from the scalar background first so the PMLs stay
    # matched to the scalar background (spike 01 decision 2), then overwrite the
    # silicon rib interior -- the design region -- with the DG0 field.
    form._epsilon = form.epsilon.as_property(dim=3)
    form._epsilon["rib_silicon"] = eps_core_field

    wf = form.weak
    zero = Constant((0.0, 0.0, 0.0))
    dv = dot(zero, form.test) * form.dx
    dv = dv.real + dv.imag
    bcs = form.build_boundary_conditions()

    A_mat, B_mat = dolfin.PETScMatrix(), dolfin.PETScMatrix()
    b = dolfin.PETScVector()
    dolfin.assemble_system(wf[0], dv, bcs, A_tensor=A_mat, b_tensor=b)
    dolfin.assemble_system(wf[1], dv, A_tensor=B_mat, b_tensor=b)
    A, B = A_mat.mat(), B_mat.mat()
    _pin_orphan_dofs(A, B)
    return A, B


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

    if inputs.operation == "write_mesh":
        try:
            mesh_text, vertices = write_mesh(
                inputs.core_epsilon, inputs.clad_epsilon, inputs.substrate_epsilon
            )
        except ImportError as exc:
            raise RuntimeError(
                "the unified mesh requires a gyptis/FEniCS backend"
            ) from exc
        return OutputSchema(mesh_text=mesh_text, design_cell_vertices=vertices)

    if inputs.design_epsilon is None:
        raise ValueError("design_epsilon is required for a gyptis solve")
    design = np.asarray(inputs.design_epsilon, dtype=float)
    if design.ndim != 1 or design.size == 0:
        raise ValueError("design_epsilon must contain at least one design cell")

    try:
        _ensure_dolfin()
        _ensure_gyptis()
    except ImportError as exc:
        raise RuntimeError(
            "gyptis eigenmode solve requires a gyptis/FEniCS backend; none is "
            "available. There is no physics-free effective-medium fallback -- "
            "run the component in its container."
        ) from exc

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

    field = _make_rib_field(dg0, mask, inputs.core_epsilon, design)
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
    except ImportError as exc:
        raise RuntimeError(
            "gyptis adjoint requires a gyptis/FEniCS backend; none is "
            "available. There is no physics-free effective-medium fallback -- "
            "run the component in its container."
        ) from exc

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
