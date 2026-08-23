# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for prismo_gyptis
# Electromagnetic eigenmode component (gyptis / FEniCS).
#
# Field-epsilon implementation: the design region carries a
# spatially-varying permittivity -- one value per DG0 design cell -- injected as
# a scalar dolfin.Function over the silicon rib interior. The forward eigensolve
# and the field-valued Hellmann-Feynman adjoint share a single two-sided SLEPc
# solve; the adjoint assembles a per-design-cell cotangent
# in one pass. A missing gyptis/FEniCS backend is a hard error --
# there is no physics-free effective-medium fallback that would fabricate an
# effective index.
#
# Unified mesh: the geometry is the single source of truth
# for BOTH solvers. ``RibBoxPML2D`` extends gyptis ``LayeredBoxPML2D`` to emit a
# real SOI rib cross-section -- rib + slab + oxide clad/substrate + a matched PML
# frame + ``contact_anode``/``contact_cathode`` lines on the slab shoulders, with
# ``silicon``/``oxide`` surface groups. gyptis eigensolves on the full domain and
# modulates only the rib interior; ChargeTransport reads the same ``.msh``
# restricted to the silicon subdomain. The ``write_mesh`` operation emits that
# shared ``.msh`` (as text, so the host -- not this container's uid -- owns the
# file) for the ChargeTransport container and the mesh-transfer operator.

import itertools
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from prismo_shared.session import SolveSessionRegistry, array_identity
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64

#
# Geometry / physics constants
#

WAVELENGTH: float = 1.55  # free-space wavelength, micrometres

# Silicon rib in oxide; surroundings stay constant (spec: only the rib interior
# is modulated). The oxide permittivity is n_ox**2 with n_ox = 1.444 (matching
# the ChargeTransport/shared-mesh oxide index), reconciling the stray clad =
# substrate = 2.10 an earlier slab-only model carried.
N_SILICON: float = 3.4757
N_OXIDE: float = 1.444
DEFAULT_CORE_EPSILON: float = N_SILICON**2  # ~12.08
DEFAULT_OXIDE_EPSILON: float = N_OXIDE**2  # ~2.085
DEFAULT_CLAD_EPSILON: float = DEFAULT_OXIDE_EPSILON
DEFAULT_SUBSTRATE_EPSILON: float = DEFAULT_OXIDE_EPSILON

# SOI rib cross-section, micrometres. A full-width silicon slab
# layer keeps ``pml_x_slab`` matched to silicon; the rib is the only embedded
# surgery and, inset from the x-PML, never touches one. Layer stack bottom->top:
# substrate (oxide) / slab (silicon) / rib band (oxide + embedded silicon rib) /
# clad (oxide).
_DEFAULT_WIDTH: float = 2.0
_PML_WIDTH: tuple[float, float] = (0.5, 0.5)
_LAYER_THICKNESS: dict[str, float] = {
    "substrate": 0.35,
    "slab": 0.10,
    "rib": 0.22,
    "clad": 0.35,
}
_RIB_HALF_WIDTH: float = 0.25  # 500 nm rib, centred at x=0
_DEFAULT_CONTACT_OFFSET: float = 0.20  # gap from rib edge to the near contact edge
_CONTACT_HALF_WIDTH: float = 0.025  # 50 nm contact line half-width


def _positive_env_float(
    name: str, default: float, env: Mapping[str, str] | None = None
) -> float:
    """A positive float knob from the environment, ``default`` when unset."""
    source = os.environ if env is None else env
    value = float(source.get(name, default))
    if value <= 0.0:
        raise ValueError(f"{name} must be positive [µm]")
    return value


def _validate_geometry(width: float, contact_offset: float) -> None:
    """Both contact footprints must lie inside the physical box, clear of the PML."""
    contact_outer = _RIB_HALF_WIDTH + contact_offset + 2 * _CONTACT_HALF_WIDTH
    if contact_outer >= width / 2.0:
        raise ValueError(
            f"contact offset {contact_offset} µm puts the contact edge at "
            f"{contact_outer} µm, outside the {width} µm wide domain"
        )


# Slab/contact geometry knobs: the physical box width and the gap
# from the rib edge to the near contact edge, in µm. Foundry designs keep the
# contacts 0.5-1 µm from the rib; the default 0.2 µm puts them in the
# mode tail. Read once at import like the mesh size, so every operation of a
# served container builds the same geometry.
_WIDTH: float = _positive_env_float("PRISMO_GYPTIS_WIDTH", _DEFAULT_WIDTH)
_CONTACT_OFFSET: float = _positive_env_float(
    "PRISMO_GYPTIS_CONTACT_OFFSET", _DEFAULT_CONTACT_OFFSET
)
_validate_geometry(_WIDTH, _CONTACT_OFFSET)
# Refine the rib so the fundamental guided mode (n_clad < neff < n_core) is
# resolved rather than only the coarse-mesh leaky mode. ``PRISMO_GYPTIS_MESH_SIZE``
# overrides it at container start (the host sets it from ``prismo run
# --mesh-size``) so a mesh-refinement study does not need an image rebuild. It is
# read once, at import, so every operation of a served container -- ``write_mesh``,
# ``solve``, the VJP, ``mode_field`` -- builds the same geometry; the shared
# ``.msh``, the design cells, and the transfer operator therefore stay consistent.
_DEFAULT_CORE_MESH_SIZE: float = 0.04
_CORE_MESH_SIZE: float = _positive_env_float(
    "PRISMO_GYPTIS_MESH_SIZE", _DEFAULT_CORE_MESH_SIZE
)

# Design region: the silicon rib interior. Every rib cell is modulated; the rib
# is inset from the x-PML by construction and sits in the rib band away from the
# y-PMLs, so no design cell ever touches a PML.
_DESIGN_Y_INSET: float = 0.0

# Tolerance for locating geometry entities by coordinate. The smallest
# feature is the 50 nm contact footprint, so 1e-6 um is far below any real
# separation while still absorbing OCC's coordinate round-off.
_GEOMETRY_TOL: float = 1e-6


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
            ``"mode_field"`` solves and returns the tracked mode's ``|E|``
            profile sampled at the mesh vertices, for the headline mode figure.
        design_epsilon: Relative permittivity per design cell -- the modulated
            silicon on the rib interior, one value per DG0 cell (order given by
            :func:`design_cell_centroids`). This is the only differentiated input.
        core_epsilon: Background silicon permittivity for the rib cells outside
            the design region and the slab (constant).
        clad_epsilon: Cladding (oxide) permittivity (constant).
        substrate_epsilon: Substrate (oxide) permittivity (constant).
        mode_index: Which guided mode the solve targets. ``0`` (default) is
            the fundamental: the largest-neff eigenpair inside the guided
            window. ``k`` is the ``k``-th guided mode in descending neff on the
            first solve of a geometry; every later solve of that geometry
            tracks the chosen branch by nearest eigenvalue, exactly as for the
            fundamental, so an optimization stays on one physical mode. The
            index is part of the tracked-branch key and the solve identity, so
            two mode targets on one geometry never share a branch or a VJP
            session. A solve that resolves fewer than ``k + 1`` guided modes
            raises rather than silently returning a lower-order one.
    """

    operation: Literal["solve", "design_cell_centroids", "write_mesh", "mode_field"] = (
        "solve"
    )
    design_epsilon: Differentiable[Array[(None,), Float64]] | None = None
    core_epsilon: float = DEFAULT_CORE_EPSILON
    clad_epsilon: float = DEFAULT_CLAD_EPSILON
    substrate_epsilon: float = DEFAULT_SUBSTRATE_EPSILON
    mode_index: int = Field(0, ge=0)


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
            exact node->DG0-cell restriction operator.
        mode_abs_e: ``(n_vertices,)`` magnitude ``|E|`` of the tracked mode at
            every mesh vertex, normalized to a peak of 1, returned by the
            ``mode_field`` operation. The eigenvector carries an arbitrary
            complex scale, so only the normalized profile is meaningful.
        mode_coordinates: ``(n_vertices, 2)`` mesh vertex coordinates in
            micrometres, in ``mode_abs_e`` order.
    """

    neff_sq: Differentiable[Array[(), Float64]] | None = None
    design_cell_centroids: Array[(None, 2), Float64] | None = None
    mesh_text: str | None = None
    design_cell_vertices: Array[(None, 3, 2), Float64] | None = None
    mode_abs_e: Array[(None,), Float64] | None = None
    mode_coordinates: Array[(None, 2), Float64] | None = None


#
# Module-level state
#

# The forward solve, behind the fixed apply/vjp endpoints. apply() runs the
# single two-sided eigensolve and stores its whole eigenstate here; the adjoint
# retrieves the session whose inputs match and assembles the field sensitivity
# from that state without re-solving.
#
# One pipeline evaluation drives *two* independent applies into this container:
# the rho-independent background field and the perturbed one. They are separate
# ``pure_callback``s, so under ``jit`` XLA -- not the pipeline -- decides which
# runs last. Keeping only the most-recent forward therefore let a background
# apply scheduled after the perturbed one evict the very session the perturbed
# adjoint was about to retrieve. Both are opened under one scope,
# the constant surroundings they share, so the perturbed session survives
# either order. That scope does not turn over between evaluations, so the
# retention cap -- not the scope -- is what keeps the eigenstates from
# accumulating across an optimization.
_MAX_RETAINED_SOLVES = 2
_session_registry = SolveSessionRegistry(max_sessions=_MAX_RETAINED_SOLVES)

# Eigenvalue of the last accepted forward solve, keyed by the geometry it was
# solved on. Successive solves within one optimization or finite-difference
# sweep differ only by a small design perturbation, so tracking the branch by
# nearest eigenvalue keeps every call on the same physical mode. "Largest neff
# in the guided window" cannot: near-degenerate modes swap rank under
# perturbations far smaller than their spacing, so neff_sq jumps between
# neighbouring inputs and no finite-difference reference survives.
# The key carries the targeted mode index as well as the constant
# surroundings: a higher-order target on the same geometry is its own branch.
GeometryKey = tuple[float, float, float, int]
_tracked_lam: dict[GeometryKey, complex] = {}


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
    mode_index: int = 0,
) -> tuple[Any, ...]:
    """Hashable fingerprint of a forward solve's inputs (field + constants + mode)."""
    return (
        *array_identity(design_epsilon),
        float(core_epsilon),
        float(clad_epsilon),
        float(substrate_epsilon),
        int(mode_index),
    )


def _band_columns() -> tuple[tuple[float, float], ...]:
    """The x intervals every silicon band is cut into, left to right.

    The slab and the rib band are built from the *same* columns so that the
    curves along their shared interface are exactly coincident and
    ``remove_all_duplicates`` (run at the end of ``LayeredBoxPML2D.__init__``)
    merges each pair into one curve owned by both bands. Cutting the two bands
    independently -- or fragmenting one of them afterwards, as an earlier version of
    this geometry did -- leaves each side with its own curve and its own 1D mesh.
    The silicon domain ChargeTransport then solves on falls into two
    disconnected pieces: a slab carrying both contacts, and a rib island that
    never sees the applied bias and whose quasi-Fermi levels are fixed only up
    to a free constant.

    The column edges are the rib sides and the two contact footprints, so the
    contacts are real edges of the silicon surfaces rather than lines embedded
    into them.
    """
    contact_inner = _RIB_HALF_WIDTH + _CONTACT_OFFSET
    contact_outer = contact_inner + 2 * _CONTACT_HALF_WIDTH
    edges = (
        -_WIDTH / 2,
        -contact_outer,
        -contact_inner,
        -_RIB_HALF_WIDTH,
        _RIB_HALF_WIDTH,
        contact_inner,
        contact_outer,
        _WIDTH / 2,
    )
    return tuple(itertools.pairwise(edges))


def _contact_span(name: str) -> tuple[float, float]:
    """The (x_min, x_max) footprint of a contact on the slab top."""
    contact_inner = _RIB_HALF_WIDTH + _CONTACT_OFFSET
    contact_outer = contact_inner + 2 * _CONTACT_HALF_WIDTH
    if name == "contact_anode":
        return -contact_outer, -contact_inner
    return contact_inner, contact_outer


def _rib_box_pml_2d() -> Any:
    """Build the ``RibBoxPML2D`` class extending gyptis ``LayeredBoxPML2D``.

    Deferred behind a function so the gyptis import stays lazy (the class body
    subclasses a gyptis type). A silicon rib sits in the centre of the oxide rib
    band, inset from the x-PML, over a full-width silicon slab that carries an
    Ohmic contact line on each shoulder.
    """
    gyptis = _ensure_gyptis()

    class RibBoxPML2D(gyptis.geometry.LayeredBoxPML2D):  # type: ignore[misc]
        """LayeredBoxPML2D with an embedded silicon rib + contact lines."""

        def make_layer(self, y_position: float, thickness: float) -> Any:
            """Cut the two silicon-bearing bands into the shared columns.

            Called by ``LayeredBoxPML2D.__init__`` once per layer. Every other
            layer stays the single full-width rectangle gyptis builds.
            """
            if not any(
                abs(y_position - self._layer_y(name)) < _GEOMETRY_TOL
                for name in ("slab", "rib")
            ):
                return super().make_layer(y_position, thickness)
            return [
                self.add_rectangle(x0, y_position, 0, x1 - x0, thickness)
                for x0, x1 in _band_columns()
            ]

        def _layer_y(self, name: str) -> float:
            """Bottom y of a named layer, usable before ``y_position`` is filled."""
            y = -self.total_thickness / 2
            for layer_name, thickness in self.thicknesses.items():
                if layer_name == name:
                    return y
                y += thickness
            raise KeyError(f"no layer named {name!r}")

        def tag_rib_silicon(self) -> None:
            """Split the rib band's physical group into oxide and silicon.

            The band's centre column is the silicon rib; the oxide remainder
            keeps the ``rib`` material tag so ``pml_x_rib`` still matches the
            oxide background either side of it.
            """
            self.synchronize()
            band = list(self.subdomains_entities["surfaces"]["rib"])
            core = [tag for tag in band if self._spans_rib_core(tag)]
            if len(core) != 1:
                raise RuntimeError(
                    f"expected exactly one silicon column in the rib band, "
                    f"found {len(core)} of {len(band)} surfaces"
                )
            oxide = [tag for tag in band if tag not in core]
            self.model.removePhysicalGroups([(2, self.subdomains["surfaces"]["rib"])])
            self.add_physical(oxide, "rib")
            self.add_physical(core, "rib_silicon")

        def _spans_rib_core(self, surface: int) -> bool:
            xmin, _ymin, _zmin, xmax, _ymax, _zmax = self.model.getBoundingBox(
                2, surface
            )
            return (
                abs(xmin + _RIB_HALF_WIDTH) < _GEOMETRY_TOL
                and abs(xmax - _RIB_HALF_WIDTH) < _GEOMETRY_TOL
            )

        def add_contacts(self) -> None:
            """Tag the two Ohmic contact lines on the slab shoulders.

            The contact footprints are column edges of both silicon bands, so
            each is already a single curve shared by the slab below and the
            oxide above -- no fragmenting, no ``mesh.embed``, and no orphan
            points left over from either.
            """
            self.synchronize()
            y = self.y_position["rib"]  # slab top
            for name in ("contact_anode", "contact_cathode"):
                xmin, xmax = _contact_span(name)
                e = _GEOMETRY_TOL
                segments = self.model.getEntitiesInBoundingBox(
                    xmin - e, y - e, -e, xmax + e, y + e, e, 1
                )
                tags = [tag for (_dim, tag) in segments]
                if len(tags) != 1:
                    raise RuntimeError(
                        f"expected one {name} curve on the slab top, found "
                        f"{len(tags)}; the slab and rib bands are not merged"
                    )
                self.add_physical(tags, name, dim=1)

    return RibBoxPML2D


# One built waveguide per constant-surroundings key. The geometry depends only
# on the three constant permittivities, so rebuilding it on every operation
# both remeshed the identical geometry and leaked its ``tempfile.mkdtemp`` mesh
# directory once per objective evaluation -- several hundred directories over a
# long optimization. Reuse is safe: ``_assemble_AB`` rebuilds the
# formulation's property dict from the scalar background before overwriting the
# rib field, so successive solves on one simulation are idempotent, and the
# adjoint/mode readers touch only epsilon-independent state (function space,
# measures, stored eigenvectors). Requests are served sequentially -- the
# module already assumes that for ``_tracked_lam`` and the session registry.
_waveguide_cache: dict[tuple[float, float, float], tuple[Any, float]] = {}


def _build_waveguide(
    core_epsilon: float,
    clad_epsilon: float,
    substrate_epsilon: float,
) -> tuple[Any, float]:
    """Build (or reuse) the unified SOI rib waveguide.

    The geometry is the single source of truth shared with ChargeTransport.
    ``build()`` meshes and writes the ``.msh`` into a per-build temp directory
    (uid-9999-writable), recorded on ``geom.msh_file`` for the
    ``write_mesh`` operation to read back. Built once per constant-surroundings
    key and cached (see ``_waveguide_cache``).

    Returns:
        Tuple of (Waveguide simulation, free-space wavenumber k0).
    """
    import tempfile

    key = (float(core_epsilon), float(clad_epsilon), float(substrate_epsilon))
    cached = _waveguide_cache.get(key)
    if cached is not None:
        return cached

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
    geom.tag_rib_silicon()
    geom.add_contacts()
    geom.set_size("rib_silicon", _CORE_MESH_SIZE)
    geom.set_size("slab", _CORE_MESH_SIZE)
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
    # One live geometry at a time: the surroundings are constants in practice,
    # so a changed key means the old simulation (and its eigenstate sessions)
    # is stale. Live sessions keep their references; only the cache lets go.
    _waveguide_cache.clear()
    _waveguide_cache[key] = (simu, k0)
    return simu, k0


def _design_mask(simu: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return (DG0 space, cell centroids, boolean design-cell mask).

    Design cells are the silicon rib interior: DG0 cells whose centroid lies in
    the rib rectangle. The rib is inset from the x-PML by construction and sits
    in the rib band away from the y-PMLs, so no design cell touches a PML.
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
    operator. Requires gyptis/FEniCS.
    """
    simu, _k0 = _build_waveguide(core_epsilon, clad_epsilon, substrate_epsilon)
    _dg0, coords, mask = _design_mask(simu)
    return coords[mask]


def _design_cell_vertices(simu: Any, dg0: Any, mask: np.ndarray) -> np.ndarray:
    """``(n_design, 3, 2)`` vertex coordinates of each design cell.

    In ``design_epsilon`` (DG0-dof) order. On the shared mesh the host matches
    these coordinates to gmsh node indices to assemble the exact node->DG0-cell
    restriction operator; a design cell is a triangle whose three
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
# Two-sided eigensolve (shared by forward + adjoint)
#


def _pin_orphan_dofs(A: Any, B: Any) -> None:
    """Give a unit diagonal to any DOF whose A and B rows are both empty.

    A DOF that couples to nothing leaves zero rows in both A and B, and the
    shift-invert factorization ``(A - sigma B)`` then hits a zero pivot. Pinning
    each such DOF to ``A[i,i] = B[i,i] = 1`` gives it the isolated eigenvalue
    ``1`` -- far outside the guided window ``(n_clad, n_core)`` at this
    wavenumber, so it is never the tracked mode and never enters the field
    sensitivity.

    The shared mesh no longer produces any: it used to embed the Ohmic contact
    lines into the silicon surfaces, which left orphan mesh nodes behind, and the
    contacts are now ordinary edges of the silicon columns. This stays as a cheap guard so
    a future geometry change degrades into a harmless spurious eigenvalue rather
    than a factorization failure.
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
    # matched to the scalar background, then overwrite the
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


# Relative nudges applied to the shift-invert shift, tried in order. The first
# is the unperturbed shift, so a healthy solve pays nothing.
#
# Shift-invert factorises ``A - sigma*B``. When sigma lands on (or numerically
# near) an eigenvalue that factorisation is singular and SLEPc aborts out of
# ``solver.solve()`` -- as ``petsc4py.PETSc.Error`` code 76 (PETSC_ERR_LIB), or,
# when the error escapes through the cython wrapper without being converted, as
# a bare ``SystemError``. Moving the shift off the offending eigenvalue recovers
# it: the spectrum does not depend on sigma, only the spectral transform used to
# reach it, so a retry returns the same eigenpairs rather than a different
# answer. Retrying is what makes a degenerate design input a slower solve
# instead of a 500.
_SHIFT_RETRY_OFFSETS = (0.0, 1e-6, -1e-6, 1e-4, -1e-4, 1e-2)


# Direct solver behind the shift-invert transform, and the Krylov-Schur
# tolerance. PETSc's native sparse LU has no threshold pivoting,
# and on this indefinite saddle-point system (A carries zero diagonals) its
# back-substitution is accurate to only ~1e-5 relative: every Ritz pair
# "converges" in the transformed operator while its true residual stays at
# 1e-5, lambda lands ~4e-7 relative off the eigenvalue, and -- because the
# roundoff pattern changes with the input -- neighbouring designs whose
# epsilon differ by 5e-11 return lambda values that jump by 5e-7 relative. That
# was the ~2e-3 relative noise floor on Delta_neff that stalled the optimizer
# around an earlier optimum. UMFPACK, MUMPS and SuperLU all pivot, all
# agree with an independent ARPACK solve to 1e-13, and leave lambda smooth to
# 1e-13 along the same line; UMFPACK was the fastest of the three here. The
# tolerance is then meaningful again: 1e-10 puts the true residual at 1e-13
# for one extra Krylov iteration.
_SHIFT_INVERT_FACTOR_SOLVER = "umfpack"
_EIGENSOLVER_TOLERANCE = 1e-10


def _attempt_two_sided_solve(A: Any, B: Any, shift: float, n: int) -> tuple[Any, int]:
    """One two-sided shift-invert SLEPc solve at a given shift."""
    from slepc4py import SLEPc

    solver = SLEPc.EPS().create(A.getComm())
    solver.setOperators(A, B)
    solver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    solver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    solver.setTarget(shift)
    st = solver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(shift)
    pc = st.getKSP().getPC()
    pc.setType("lu")
    pc.setFactorSolverType(_SHIFT_INVERT_FACTOR_SOLVER)
    solver.setDimensions(n)
    solver.setTolerances(_EIGENSOLVER_TOLERANCE)
    solver.setTwoSided(True)
    solver.solve()
    return solver, solver.getConverged()


# Eigenpairs requested from the two-sided solve. The shift sits at the core
# line, so the guided modes are the first pairs below it; 12 comfortably covers
# the fundamental and a few higher-order branches of this rib. Each further
# targeted order adds headroom so the ranked guided list reaches it: every
# physical mode costs two converged pairs (the real doubling), so four per
# order is a 2x margin.
_BASE_EIGENPAIRS = 12
_EIGENPAIRS_PER_ORDER = 4


def _eigenpair_budget(mode_index: int) -> int:
    """Number of eigenpairs to converge for a given targeted mode order."""
    return _BASE_EIGENPAIRS + _EIGENPAIRS_PER_ORDER * int(mode_index)


def _two_sided_solver(
    A: Any, B: Any, target: float, n: int = _BASE_EIGENPAIRS
) -> tuple[Any, int]:
    """Two-sided shift-invert SLEPc solve; returns (solver, n_converged).

    The shift is retried along ``_SHIFT_RETRY_OFFSETS`` when the factorisation
    fails or nothing converges. ``target`` is kept as the eigenvalue the caller
    selects around, so a retry changes only how the spectrum is reached.
    """
    failures = []
    for offset in _SHIFT_RETRY_OFFSETS:
        shift = target * (1.0 + offset)
        try:
            solver, nconv = _attempt_two_sided_solve(A, B, shift, n)
        except Exception as exc:
            # Deliberately broad: the failure arrives as petsc4py.PETSc.Error,
            # or as a SystemError raised when that error escapes the cython
            # wrapper unconverted. Both mean the same thing here -- this shift
            # did not work -- and both are re-raised below if no shift does.
            failures.append(f"shift {shift:.6g}: {type(exc).__name__}: {exc}")
            continue
        if nconv > 0:
            return solver, nconv
        failures.append(f"shift {shift:.6g}: converged 0 eigenpairs")
    raise RuntimeError(
        "two-sided SLEPc eigensolve failed at every retried shift around "
        f"{target:.6g}: " + "; ".join(failures)
    )


def _is_guided(neff: float, core_epsilon: float, clad_epsilon: float) -> bool:
    """Whether ``neff`` lies in the physical guided window of this waveguide.

    A mode bound to the silicon is slower than the core and faster than the
    cladding, i.e. ``sqrt(clad) < neff < sqrt(core)``. Anything else is
    radiating, leaky, or an artefact of the discretisation -- the isolated
    eigenvalue a pinned orphan DOF contributes, say. Both the branch selection
    and the branch *update* consult the same window, so a solve that lands
    outside it can neither be tracked to nor become what later solves track.
    """
    return float(np.sqrt(clad_epsilon)) < neff < float(np.sqrt(core_epsilon))


def _neff_of(solver: Any, index: int, k0: float) -> float:
    """Effective index of a converged eigenpair."""
    return float(np.real(np.sqrt(solver.getEigenvalue(index))) / k0)


# Two converged eigenpairs whose effective indices agree to this relative
# tolerance are one physical mode. The real-doubled system returns each mode
# twice as a complex-conjugate pair: equal real parts (equal neff), imaginary
# parts of opposite sign, which the PML makes non-negligible (Im/Re up to ~1e-2
# on this domain) -- so the copies are told apart by neff, not by the complex
# eigenvalue. The copies agree in neff to ~1e-10; genuinely distinct guided
# modes of this rib differ by ~1e-1 relative, so the gap is wide.
_DEGENERATE_REL_TOL = 1e-6


def _distinct_by_descending_neff(
    solver: Any, indices: list[int], k0: float
) -> list[int]:
    """One representative index per distinct mode (by neff), largest neff first."""
    ranked = sorted(indices, key=lambda i: _neff_of(solver, i, k0), reverse=True)
    distinct: list[int] = []
    for i in ranked:
        neff = _neff_of(solver, i, k0)
        if distinct:
            previous = _neff_of(solver, distinct[-1], k0)
            if abs(neff - previous) <= _DEGENERATE_REL_TOL * previous:
                continue
        distinct.append(i)
    return distinct


def _select_index(
    solver: Any,
    nconv: int,
    k0: float,
    core_epsilon: float,
    clad_epsilon: float,
    ref_lam: complex | None,
    mode_index: int = 0,
) -> int:
    """Index of the tracked eigenpair.

    When ``ref_lam`` is given -- the eigenvalue this geometry was last solved on
    -- the mode is tracked by nearest eigenvalue *among the guided candidates*,
    so every solve after the first stays on one physical branch rather than
    hopping to whichever root happened to converge closest. On the first solve
    of a geometry there is nothing to track, so the targeted guided mode is
    selected: the *distinct* eigenvalues inside the window ranked by descending
    neff, the fundamental at ``mode_index`` 0, the first higher-order mode at
    1, and so on. Distinct matters: the assembled system is the real doubling
    of the complex problem, so every physical mode converges twice, as a
    complex-conjugate eigenvalue pair with the same neff, and a naive rank
    would return the fundamental again at index 1. A first solve that resolves fewer distinct
    guided modes than the target needs raises -- silently returning a
    lower-order mode would optimize the wrong device. With no target beyond the
    fundamental, a solve that resolves no guided mode at all falls back to the
    nearest candidate over the whole converged set (a leaky mode; gradient
    correctness is unaffected).
    """
    guided = [
        i
        for i in range(nconv)
        if _is_guided(_neff_of(solver, i, k0), core_epsilon, clad_epsilon)
    ]

    if ref_lam is not None:
        candidates = guided if guided else list(range(nconv))
        dists = [abs(solver.getEigenvalue(i) - ref_lam) for i in candidates]
        return int(candidates[int(np.argmin(dists))])

    if guided:
        ranked = _distinct_by_descending_neff(solver, guided, k0)
        if mode_index >= len(ranked):
            raise RuntimeError(
                f"mode_index {mode_index} requested but only {len(ranked)} "
                f"distinct guided mode(s) converged in the window; this "
                f"waveguide may not support that many guided modes, or more "
                f"eigenpairs are needed"
            )
        return int(ranked[mode_index])

    if mode_index > 0:
        raise RuntimeError(
            f"mode_index {mode_index} requested but no guided mode converged "
            f"in the window"
        )

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
    mode_index: int = 0,
) -> dict[str, Any]:
    """Assemble + two-sided solve; return the tracked eigenpair state.

    The state carries ``guided``: whether the selected eigenpair landed in the
    physical guided window. ``apply`` gates the tracked-branch update on it, so
    a shift retry that misses the branch costs one solve rather than
    redirecting every later solve of this geometry.
    """
    A, B = _assemble_AB(simu, eps_core_field)
    target = core_epsilon * k0 * k0
    # A higher-order target needs more converged eigenpairs below the shift to
    # be sure of resolving it; the fundamental keeps the original budget.
    solver, nconv = _two_sided_solver(A, B, target, n=_eigenpair_budget(mode_index))
    idx = _select_index(
        solver, nconv, k0, core_epsilon, clad_epsilon, ref_lam, mode_index
    )

    rx, cx = A.createVecRight(), A.createVecRight()
    ry, cy = A.createVecRight(), A.createVecRight()
    lam = solver.getEigenpair(idx, rx, cx)
    solver.getLeftEigenvector(idx, ry, cy)
    kz = np.sqrt(lam)
    neff = float(np.real(kz / k0))
    return {
        "A": A,
        "B": B,
        "lam": lam,
        "kz": kz,
        "neff": neff,
        "guided": _is_guided(neff, core_epsilon, clad_epsilon),
        "rx": rx,
        "cx": cx,
        "ry": ry,
        "cy": cy,
    }


def _advance_tracked_lam(
    tracked: dict[GeometryKey, complex],
    geometry: GeometryKey,
    state: dict[str, Any],
) -> None:
    """Record this solve's eigenvalue as the branch later solves track.

    Only a guided solve may advance the branch. Recording a non-guided one
    would make every later solve of this geometry track *it*, with nothing to
    recover from: the reference is what selection is measured against, so once
    it moves off the guided window it never comes back.
    """
    if state["guided"]:
        tracked[geometry] = state["lam"]


def _lr_product(matrix: Any, state: dict[str, Any]) -> complex:
    """y^H matrix x for the complex left/right eigenvectors of the real system."""
    rx, cx, ry, cy = state["rx"], state["cx"], state["ry"], state["cy"]
    m_rx, m_cx = matrix.createVecRight(), matrix.createVecRight()
    matrix.mult(rx, m_rx)
    matrix.mult(cx, m_cx)
    return ry.dot(m_rx) + cy.dot(m_cx) + 1j * (ry.dot(m_cx) - cy.dot(m_rx))


#
# Single-pass field sensitivity
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
        return dens(Xr, Yr) + dens(Xc, Yc) + 1j * (dens(Xc, Yr) - dens(Xr, Yc))

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


def _forward_solve(
    inputs: InputSchema,
) -> tuple[np.ndarray, Any, Any, np.ndarray, float, dict[str, Any], GeometryKey]:
    """Validate a design field, build the waveguide, and run the tracked solve.

    Shared by ``apply`` and the ``mode_field`` operation so both reach the same
    eigenpair from the same inputs. The tracked eigenvalue is *read* here but not
    written: only ``apply`` -- the forward a VJP can be taken against -- advances
    the tracked branch, so an out-of-band field query cannot redirect tracking.

    Returns:
        ``(design, simu, dg0, mask, k0, state, geometry_key)``.
    """
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
    # Track the branch this geometry was last solved on, so neighbouring design
    # inputs -- an optimizer step, or the two sides of a finite difference --
    # return the same mode rather than whichever near-degenerate mode currently
    # ranks highest in the guided window.
    geometry: GeometryKey = (
        float(inputs.core_epsilon),
        float(inputs.clad_epsilon),
        float(inputs.substrate_epsilon),
        int(inputs.mode_index),
    )
    state = _solve_state(
        simu,
        field,
        k0,
        inputs.core_epsilon,
        inputs.clad_epsilon,
        ref_lam=_tracked_lam.get(geometry),
        mode_index=inputs.mode_index,
    )
    return design, simu, dg0, mask, k0, state, geometry


def _vertex_abs_e(form: Any, state: dict[str, Any]) -> np.ndarray:
    """``|E|`` of the tracked mode at every mesh vertex.

    The eigenvector of the real doubled system is itself complex, ``rx + i cx``,
    and each of those halves already packs the complex 3-vector E as six real
    components ``[Re E0..2, Im E0..2]``. So the physical field is

        ``E = (rx[:3] - cx[3:]) + i (rx[3:] + cx[:3])``

    and ``|E|`` is the pointwise magnitude of that 3-vector. It is invariant
    under the eigenvector's arbitrary global phase; its arbitrary scale is
    removed by normalizing to a peak of 1.
    """
    from gyptis.utils.helpers import array2function

    fs = form.function_space
    mesh = fs.mesh()

    def components(vec: Any) -> np.ndarray:
        fn = array2function(vec.getArray(), fs)
        return np.asarray(fn.compute_vertex_values(mesh)).reshape(6, -1)

    a, b = components(state["rx"]), components(state["cx"])
    e_real, e_imag = a[:3] - b[3:], a[3:] + b[:3]
    magnitude = np.sqrt((e_real**2 + e_imag**2).sum(axis=0))
    peak = float(magnitude.max())
    if peak == 0.0:
        raise RuntimeError("tracked eigenmode has an identically zero field")
    return magnitude / peak


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

    if inputs.operation == "mode_field":
        _design, simu, _dg0, _mask, _k0, state, _geometry = _forward_solve(inputs)
        return OutputSchema(
            neff_sq=state["neff"] * state["neff"],
            mode_abs_e=_vertex_abs_e(simu.formulation, state),
            mode_coordinates=np.asarray(
                simu.formulation.function_space.mesh().coordinates(), dtype=float
            ),
        )

    design, simu, dg0, mask, k0, state, geometry = _forward_solve(inputs)
    _advance_tracked_lam(_tracked_lam, geometry, state)

    _session_registry.open(
        _solve_identity(
            design,
            inputs.core_epsilon,
            inputs.clad_epsilon,
            inputs.substrate_epsilon,
            inputs.mode_index,
        ),
        state={"simu": simu, "dg0": dg0, "mask": mask, "k0": k0, "eigen": state},
        scope=geometry,
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
            inputs.mode_index,
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
