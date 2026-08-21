"""Shared waveguide mesh generation for the PRISMO pipeline.

Generates a 2D SOI rib waveguide cross-section mesh as a Gmsh ``.msh`` file.
Both the ChargeTransport.jl and gyptis Tesseracts consume the same mesh file at runtime via
``MeshRef.path``.  Physical-group naming is defined here so both integrators
agree on which subdomains are contacts, silicon, and oxide.

Units and vocabulary follow the *shared-mesh contract*, not SI (ticket 15).
A container run authors the shared mesh with gyptis, an in-process run authors
it here, and ChargeTransport reads whichever one was written: ``ct_common.jl``
scales the grid it loads by ``MICROMETRES_TO_METRES`` and collects its silicon
cells from the ``slab`` and ``rib_silicon`` physical groups. Both authors
therefore emit **micrometre** coordinates under gyptis' group names -- and so
do the geometry dimensions below, the ``--r-min`` filter radius, and the node
coordinates the plots consume. Authoring this mesh in metres instead made the
µm-scaled default filter radius an all-pairs filter over a 3 µm device (which
collapses the design field to its global mean) and left the Julia silicon
lookup with no group it recognised.

Geometry (cross-section, not to scale)::

    ┌───────────────────────────────────────┐  ← SiO₂ cladding
    │                                       │
    │         ┌─────────────┐               │  ← Si rib (220 nm x 500 nm)
    │         │             │               │
    ├─────────┤             ├─────────┬─────┤  ← Si slab (100 nm)
    │         │             │         │     │  ← Al contacts on slab shoulders
    │         └─────────────┘         │     │
    │                                       │
    ├───────────────────────────────────────┤  ← SiO₂ substrate
    │                                       │
    └───────────────────────────────────────┘

Physical groups exported to the ``.msh`` file:
``slab``, ``rib_silicon``, ``oxide``, ``contact_anode``, ``contact_cathode``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class RibWaveguideGeometry:
    """SOI rib waveguide cross-section dimensions and material properties.

    Every length is in **micrometres**, matching the shared-mesh contract this
    module documents; the refractive indices are dimensionless.
    """

    rib_thickness: float
    rib_width: float
    slab_thickness: float
    substrate_thickness: float
    cladding_thickness: float
    box_width: float
    contact_width: float
    contact_offset: float
    mesh_res_junction: float
    mesh_res_core: float
    mesh_res_bulk: float
    silicon_index: float
    oxide_index: float
    wavelength: float

    def __init__(
        self,
        *,
        rib_thickness: float = 0.22,
        rib_width: float = 0.5,
        slab_thickness: float = 0.1,
        substrate_thickness: float = 0.5,
        cladding_thickness: float = 0.5,
        box_width: float = 3.0,
        contact_width: float = 0.05,
        contact_offset: float = 0.2,
        mesh_res_junction: float = 0.01,
        mesh_res_core: float = 0.02,
        mesh_res_bulk: float = 0.05,
        silicon_index: float = 3.4757,
        oxide_index: float = 1.444,
        wavelength: float = 1.55,
    ) -> None:
        self.rib_thickness = rib_thickness
        self.rib_width = rib_width
        self.slab_thickness = slab_thickness
        self.substrate_thickness = substrate_thickness
        self.cladding_thickness = cladding_thickness
        self.box_width = box_width
        self.contact_width = contact_width
        self.contact_offset = contact_offset
        self.mesh_res_junction = mesh_res_junction
        self.mesh_res_core = mesh_res_core
        self.mesh_res_bulk = mesh_res_bulk
        self.silicon_index = silicon_index
        self.oxide_index = oxide_index
        self.wavelength = wavelength

    @property
    def total_height(self) -> float:
        """Total domain height: substrate + slab + rib + cladding."""
        return (
            self.substrate_thickness
            + self.slab_thickness
            + self.rib_thickness
            + self.cladding_thickness
        )

    @property
    def rib_left(self) -> float:
        """Left edge x-coordinate of the rib (centered at x=0)."""
        return -self.rib_width / 2

    @property
    def rib_right(self) -> float:
        """Right edge x-coordinate of the rib (centered at x=0)."""
        return self.rib_width / 2

    @property
    def half_width(self) -> float:
        """Half the total domain width."""
        return self.box_width / 2

    @property
    def slab_top(self) -> float:
        """Top y-coordinate of the slab layer."""
        return self.substrate_thickness + self.slab_thickness

    @property
    def rib_top(self) -> float:
        """Top y-coordinate of the rib (slab top + rib thickness)."""
        return self.slab_top + self.rib_thickness


# gyptis' vocabulary for the same regions: the silicon is split into the
# full-width ``slab`` and the ``rib_silicon`` core above it, which is what
# ``ct_common.jl`` collects as the silicon domain.
SILICON_GROUP_NAMES: tuple[str, str] = ("slab", "rib_silicon")

PHYSICAL_GROUP_NAMES: list[str] = [
    *SILICON_GROUP_NAMES,
    "oxide",
    "contact_anode",
    "contact_cathode",
]

_DEFAULT_MESH_PATH = Path("outputs") / "waveguide.msh"


def build_rib_waveguide_mesh(
    mesh_path: str | Path | None = None,
    geometry: RibWaveguideGeometry | None = None,
) -> str:
    """Build the SOI rib waveguide mesh and write a ``.msh`` file.

    Args:
        mesh_path: Destination path for the ``.msh`` file. Defaults to
            ``outputs/waveguide.msh``.
        geometry: Geometry parameters. Defaults to ``RibWaveguideGeometry()``.

    Returns:
        Absolute path to the generated mesh file.
    """
    import importlib.util

    if mesh_path is None:
        mesh_path = _DEFAULT_MESH_PATH
    mesh_path = Path(mesh_path)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)

    if geometry is None:
        geometry = RibWaveguideGeometry()

    if importlib.util.find_spec("gmsh") is None:
        return _generate_empty_msh(mesh_path)

    return build_rib_waveguide_mesh_via_gmsh(str(mesh_path), geometry)


def build_rib_waveguide_mesh_via_gmsh(
    mesh_path: str,
    geometry: RibWaveguideGeometry,
    *,
    view: bool = False,
) -> str:
    """Low-level mesh builder using gmsh Python API directly.

    Exported so tests can call it after ``gmsh.initialize()``.
    """
    import gmsh  # type: ignore[import-untyped]

    _ensure_gmsh_ready(gmsh)
    gmsh.model.add("rib_waveguide")

    return _build_mesh_rectangles(gmsh, geometry, mesh_path, view)


def _ensure_gmsh_ready(gmsh: object) -> None:
    """Ensure gmsh is initialized and clear any existing models."""
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()


def _build_mesh_rectangles(
    gmsh: object,
    geometry: RibWaveguideGeometry,
    mesh_path: str,
    view: bool,
) -> str:
    """Build mesh using rectangular patches via gmsh built-in CAD."""
    hw = geometry.half_width
    sub = geometry.substrate_thickness
    slb = geometry.slab_thickness
    rib_h = geometry.rib_thickness
    clad = geometry.cladding_thickness
    rib_l = geometry.rib_left
    rib_r = geometry.rib_right
    ct_w = geometry.contact_width
    ct_off = geometry.contact_offset

    y0 = 0.0
    y1 = sub
    y2 = sub + slb
    y3 = sub + slb + rib_h
    y4 = sub + slb + rib_h + clad

    ct_l_start = rib_l - ct_off - ct_w
    ct_l_end = rib_l - ct_off
    ct_r_start = rib_r + ct_off
    ct_r_end = rib_r + ct_off + ct_w

    _tc = [0]
    _top_curves: dict[int, int] = {}

    def _rect(x1: float, y1: float, x2: float, y2: float, lc: float) -> int:
        _tc[0] += 1
        p1 = gmsh.model.geo.addPoint(x1, y1, 0, lc)
        p2 = gmsh.model.geo.addPoint(x2, y1, 0, lc)
        p3 = gmsh.model.geo.addPoint(x2, y2, 0, lc)
        p4 = gmsh.model.geo.addPoint(x1, y2, 0, lc)
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.geo.addPlaneSurface([cl])
        _top_curves[surf] = l3
        return surf

    core_sz = geometry.mesh_res_core
    bulk_sz = geometry.mesh_res_bulk

    slab_surfs: list[int] = []
    rib_silicon_surfs: list[int] = []
    oxide_surfs: list[int] = []
    anode_surfs: list[int] = []
    cathode_surfs: list[int] = []

    oxide_surfs.append(_rect(-hw, y0, hw, y1, bulk_sz))

    slab_surfs.append(_rect(-hw, y1, ct_l_start, y2, bulk_sz))
    anode_surfs.append(_rect(ct_l_start, y1, ct_l_end, y2, bulk_sz))
    slab_surfs.append(_rect(ct_l_end, y1, rib_l, y2, bulk_sz))
    slab_surfs.append(_rect(rib_l, y1, rib_r, y2, core_sz))
    slab_surfs.append(_rect(rib_r, y1, ct_r_start, y2, bulk_sz))
    cathode_surfs.append(_rect(ct_r_start, y1, ct_r_end, y2, bulk_sz))
    slab_surfs.append(_rect(ct_r_end, y1, hw, y2, bulk_sz))

    rib_silicon_surfs.append(_rect(rib_l, y2, rib_r, y3, core_sz))

    oxide_surfs.append(_rect(-hw, y2, rib_l, y3, bulk_sz))
    oxide_surfs.append(_rect(rib_r, y2, hw, y3, bulk_sz))
    oxide_surfs.append(_rect(-hw, y3, rib_l, y4, bulk_sz))
    oxide_surfs.append(_rect(rib_l, y3, rib_r, y4, core_sz))
    oxide_surfs.append(_rect(rib_r, y3, hw, y4, bulk_sz))

    # The rib is assembled from independently-added rectangles that abut along
    # shared edges. Each ``_rect`` adds its own points and lines, so those
    # shared edges start out as coincident-but-distinct geometry. Without this
    # merge the generated mesh is non-conforming: duplicate nodes sit on every
    # internal interface (~2x the node count), the finite-volume operator gains
    # a null space across the split, and ChargeTransport's adjoint solve hits a
    # SingularException. Merging the duplicate CAD entities makes neighbouring
    # surfaces share real curves, so the triangulation is conforming.
    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()

    # Contacts are dim-1 physical groups on the top edge of the contact
    # patches: ChargeTransport.jl applies voltages to boundary REGIONS
    # (curves in 2D), and ExtendableGrids only reads dim-1 elements as
    # boundary faces (ticket 17). The metal patches themselves stay part of
    # the silicon domain for the electrical solve.
    slab_surfs.extend(anode_surfs)
    slab_surfs.extend(cathode_surfs)

    # gyptis' vocabulary, not a local one: ChargeTransport collects its silicon
    # domain from ``slab`` + ``rib_silicon`` whichever author wrote the shared
    # mesh (ticket 15).
    _add_physical_group(gmsh, 2, slab_surfs, "slab")
    _add_physical_group(gmsh, 2, rib_silicon_surfs, "rib_silicon")
    _add_physical_group(gmsh, 2, oxide_surfs, "oxide")
    _add_physical_group(gmsh, 1, [_top_curves[s] for s in anode_surfs], "contact_anode")
    _add_physical_group(gmsh, 1, [_top_curves[s] for s in cathode_surfs], "contact_cathode")

    gmsh.model.mesh.generate(2)

    if view:
        gmsh.fltk.run()

    gmsh.write(mesh_path)
    return mesh_path


def read_mesh_node_coordinates(mesh_path: str | Path) -> np.ndarray:
    """Extract 2D node coordinates from a Gmsh ``.msh`` v4 file.

    Returns a ``(n_nodes, 2)`` float64 array of (x, y) positions, suitable
    for feeding into :func:`prismo.density_filter.assemble_filter_matrix`.

    Requires ``gmsh`` to be importable.
    """
    import importlib.util

    if importlib.util.find_spec("gmsh") is None:
        return np.empty((0, 2))

    import gmsh  # type: ignore[import-untyped]

    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    was_initialized = gmsh.isInitialized()
    if not was_initialized:
        gmsh.initialize()

    try:
        gmsh.open(str(mesh_path))
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        gmsh.clear()

        if len(node_tags) == 0:
            return np.empty((0, 2))

        n_nodes = len(node_tags)
        coords_3d = np.array(node_coords, dtype=float).reshape(n_nodes, 3)
        return np.ascontiguousarray(coords_3d[:, :2])
    finally:
        if not was_initialized:
            gmsh.finalize()


def read_mesh_silicon_triangulation(mesh_path: str | Path) -> np.ndarray:
    """Extract zero-based silicon triangle node indices from a Gmsh mesh.

    The returned indices address the coordinate rows from
    :func:`read_mesh_node_coordinates`. Only dim-2, three-node triangle
    elements in the :data:`SILICON_GROUP_NAMES` physical groups (``slab`` and
    ``rib_silicon``, the same pair ``ct_common.jl`` collects) are included, so
    the reader spans the whole silicon domain of either mesh author. As with
    the node coordinate reader, an environment without Gmsh returns an empty
    array so no-backend code paths remain importable.
    """
    import importlib.util

    if importlib.util.find_spec("gmsh") is None:
        return np.empty((0, 3), dtype=np.intp)

    import gmsh  # type: ignore[import-untyped]

    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    was_initialized = gmsh.isInitialized()
    if not was_initialized:
        gmsh.initialize()

    try:
        gmsh.open(str(mesh_path))
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        index_by_tag = {int(tag): index for index, tag in enumerate(node_tags)}
        silicon_groups = [
            tag
            for dim, tag in gmsh.model.getPhysicalGroups(2)
            if gmsh.model.getPhysicalName(dim, tag) in SILICON_GROUP_NAMES
        ]
        triangles: list[np.ndarray] = []
        for group_tag in silicon_groups:
            for entity_tag in gmsh.model.getEntitiesForPhysicalGroup(2, group_tag):
                element_types, _, element_nodes = gmsh.model.mesh.getElements(
                    2, entity_tag
                )
                for element_type, node_tags_for_type in zip(
                    element_types, element_nodes, strict=True
                ):
                    _, dim, _, n_nodes, _, _ = gmsh.model.mesh.getElementProperties(
                        element_type
                    )
                    if dim != 2 or n_nodes != 3:
                        continue
                    tags = np.asarray(node_tags_for_type, dtype=np.int64).reshape(-1, 3)
                    triangles.append(
                        np.asarray(
                            [
                                [index_by_tag[int(tag)] for tag in triangle]
                                for triangle in tags
                            ],
                            dtype=np.intp,
                        )
                    )
        gmsh.clear()

        if not triangles:
            return np.empty((0, 3), dtype=np.intp)
        return np.concatenate(triangles)
    finally:
        if not was_initialized:
            gmsh.finalize()


def _add_physical_group(gmsh: object, dim: int, tags: list[int], name: str) -> None:
    """Add a named physical group, handling empty tag lists silently."""
    if not tags:
        return
    pg = gmsh.model.addPhysicalGroup(dim, tags)
    gmsh.model.setPhysicalName(dim, pg, name)


def _generate_empty_msh(mesh_path: Path) -> str:
    """Generate a minimal valid .msh placeholder when gmsh is unavailable."""
    content = """$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
0
$EndNodes
$Elements
0
$EndElements
"""
    mesh_path.write_text(content)
    return str(mesh_path.resolve())
