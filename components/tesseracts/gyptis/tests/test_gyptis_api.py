"""Seam tests for the gyptis field-epsilon Tesseract component.

Public interface under test: apply() + vector_jacobian_product() in
tesseract_api.py. The forward accepts a design-region permittivity *field*
(one value per DG0 design cell) with constant surroundings, and the adjoint
returns a per-design-cell cotangent (tickets 02/03).

Covers schema validation, contract shapes, the hard error raised when
gyptis/FEniCS is absent (ticket 04 -- no physics-free effective-medium
fallback), and gyptis-backed integration tests (guided mode, spatial response,
single-pass field VJP vs finite differences).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "gyptis_tesseract_api", Path(__file__).resolve().parents[1] / "tesseract_api.py"
)
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)

InputSchema = _api.InputSchema
OutputSchema = _api.OutputSchema
apply = _api.apply
vector_jacobian_product = _api.vector_jacobian_product

N_DESIGN = 8


def _gyptis_available() -> bool:
    try:
        import dolfin  # noqa: F401
        import gyptis  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inputs(design_epsilon: np.ndarray | None = None) -> InputSchema:
    if design_epsilon is None:
        design_epsilon = np.full(N_DESIGN, _api.DEFAULT_CORE_EPSILON)
    return InputSchema(design_epsilon=design_epsilon)


def _sized_design(rng_scale: float = 0.5) -> np.ndarray:
    """A structured design field sized to the real gyptis design region."""
    centroids = _api.design_cell_centroids()
    n = centroids.shape[0]
    pattern = np.random.RandomState(0).uniform(-1.0, 1.0, n)
    pattern -= pattern.mean()
    return _api.DEFAULT_CORE_EPSILON + rng_scale * pattern


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_input_schema_accepts_numpy_field() -> None:
    inp = InputSchema(design_epsilon=np.array([12.0, 2.0, 1.0, 3.0]))
    assert np.asarray(inp.design_epsilon).shape == (4,)


def test_input_schema_accepts_list() -> None:
    inp = InputSchema(design_epsilon=[12.0, 2.0, 1.0, 3.0])
    assert np.asarray(inp.design_epsilon).shape == (4,)


def test_input_schema_defaults_constant_surroundings() -> None:
    inp = InputSchema(design_epsilon=[12.0, 12.0])
    assert inp.core_epsilon == _api.DEFAULT_CORE_EPSILON
    assert inp.clad_epsilon == _api.DEFAULT_CLAD_EPSILON
    assert inp.substrate_epsilon == _api.DEFAULT_SUBSTRATE_EPSILON


def test_output_schema_neff_sq_is_scalar() -> None:
    out = OutputSchema(neff_sq=2.5)
    assert np.asarray(out.neff_sq).shape == ()


# ---------------------------------------------------------------------------
# apply() contract
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_apply_returns_output_schema() -> None:
    assert isinstance(apply(make_inputs()), OutputSchema)


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_apply_returns_scalar_neff_sq() -> None:
    assert np.asarray(apply(make_inputs()).neff_sq).shape == ()


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_apply_returns_finite_neff_sq() -> None:
    outputs = apply(make_inputs(np.array([12.0, 11.0, 12.5, 11.5])))
    assert np.isfinite(float(outputs.neff_sq))


def test_apply_rejects_empty_field() -> None:
    with pytest.raises(ValueError):
        apply(make_inputs(np.array([])))


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_apply_inspection_returns_design_centroids_in_forward_order() -> None:
    outputs = apply(InputSchema(operation="design_cell_centroids"))
    np.testing.assert_allclose(
        outputs.design_cell_centroids, _api.design_cell_centroids()
    )


# ---------------------------------------------------------------------------
# vector_jacobian_product() contract
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_vjp_returns_cotangent_shaped_like_design_field() -> None:
    inputs = make_inputs()
    apply(inputs)
    result = vector_jacobian_product(
        inputs, {"design_epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
    )
    assert set(result.keys()) == {"design_epsilon"}
    assert np.asarray(result["design_epsilon"]).shape == (N_DESIGN,)


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_vjp_returns_finite_values() -> None:
    inputs = make_inputs()
    apply(inputs)
    result = vector_jacobian_product(
        inputs, {"design_epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
    )
    assert np.all(np.isfinite(np.asarray(result["design_epsilon"])))


def test_vjp_empty_when_input_not_requested() -> None:
    inputs = make_inputs()
    result = vector_jacobian_product(inputs, set(), {"neff_sq"}, {"neff_sq": 1.0})
    assert result == {}


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_vjp_linear_in_cotangent() -> None:
    inputs = make_inputs()
    apply(inputs)
    r1 = vector_jacobian_product(
        inputs, {"design_epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
    )
    apply(inputs)
    r2 = vector_jacobian_product(
        inputs, {"design_epsilon"}, {"neff_sq"}, {"neff_sq": 3.0}
    )
    ratio = np.asarray(r2["design_epsilon"]) / np.asarray(r1["design_epsilon"])
    np.testing.assert_allclose(ratio, 3.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# No physics-free fallback (gyptis / FEniCS absent) -- ticket 04
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_gyptis_available(), reason="exercises the no-gyptis error path")
def test_apply_without_backend_raises() -> None:
    with pytest.raises(RuntimeError, match="gyptis/FEniCS backend"):
        apply(make_inputs(np.array([1.0, 2.0, 3.0, 4.0])))


@pytest.mark.skipif(_gyptis_available(), reason="exercises the no-gyptis error path")
def test_vjp_without_backend_raises() -> None:
    inputs = make_inputs()
    with pytest.raises(RuntimeError, match="gyptis/FEniCS backend"):
        vector_jacobian_product(
            inputs, {"design_epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
        )


# ---------------------------------------------------------------------------
# Integration tests (gyptis / FEniCS available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_design_cell_centroids_inside_pml_inset_box() -> None:
    centroids = _api.design_cell_centroids()
    assert centroids.ndim == 2 and centroids.shape[1] == 2
    assert centroids.shape[0] > 0
    # Provably clear of the PMLs beyond +/- _WIDTH/2.
    assert np.abs(centroids[:, 0]).max() < _api._WIDTH / 2


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_apply_lands_on_guided_mode() -> None:
    neff_sq = float(apply(make_inputs(_sized_design(0.0))).neff_sq)
    neff = neff_sq**0.5
    assert _api.DEFAULT_CLAD_EPSILON**0.5 < neff < _api.DEFAULT_CORE_EPSILON**0.5


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_apply_responds_to_spatial_pattern_at_fixed_mean() -> None:
    structured = _sized_design(0.5)
    uniform = np.full_like(structured, structured.mean())
    neff_sq_structured = float(apply(make_inputs(structured)).neff_sq)
    neff_sq_uniform = float(apply(make_inputs(uniform)).neff_sq)
    assert abs(neff_sq_structured - neff_sq_uniform) > 1e-6


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_vjp_field_matches_finite_difference() -> None:
    design = _sized_design(0.5)
    apply(make_inputs(design))
    grad = np.asarray(
        vector_jacobian_product(
            make_inputs(design), {"design_epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
        )["design_epsilon"]
    )
    assert grad.shape == design.shape
    assert grad.std() > 0.0  # spatially resolved, not a uniform gradient

    h = 1e-4
    for local in (0, design.size // 2, design.size - 1):
        vp = design.copy()
        vp[local] += h
        vm = design.copy()
        vm[local] -= h
        fp = float(apply(make_inputs(vp)).neff_sq)
        fm = float(apply(make_inputs(vm)).neff_sq)
        fd = (fp - fm) / (2 * h)
        rel = abs(fd - grad[local]) / max(abs(grad[local]), 1e-12)
        assert rel < 1e-3, f"cell {local}: rel-err {rel:.2e}"


# ---------------------------------------------------------------------------
# Shared unified mesh (ticket 14)
# ---------------------------------------------------------------------------


def _silicon_mesh_topology(mesh_path: str) -> dict[str, object]:
    """Silicon triangles, contact-line nodes and orphan nodes of a shared mesh.

    ChargeTransport solves on the silicon subdomain of the shared mesh -- the
    ``slab`` and ``rib_silicon`` groups -- with the applied bias entering through
    the ``contact_anode``/``contact_cathode`` lines. That only describes a device
    if the two silicon groups are one connected domain and the contact lines lie
    on it.
    """
    import gmsh

    was_initialized = gmsh.isInitialized()
    if not was_initialized:
        gmsh.initialize()
    try:
        gmsh.open(mesh_path)
        node_tags, _coords, _ = gmsh.model.mesh.getNodes()
        entity_groups: dict[int, list[str]] = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                entity_groups.setdefault(entity if dim == 2 else -entity, []).append(
                    name
                )

        triangles: list[tuple[int, ...]] = []
        used_nodes: set[int] = set()
        for _dim, entity in gmsh.model.getEntities(2):
            names = entity_groups.get(entity, [])
            element_types, _, element_nodes = gmsh.model.mesh.getElements(2, entity)
            for element_type, nodes in zip(element_types, element_nodes, strict=True):
                _n, dim, _o, nodes_per_element, _p, _q = (
                    gmsh.model.mesh.getElementProperties(element_type)
                )
                if dim != 2 or nodes_per_element != 3:
                    continue
                cells = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
                used_nodes.update(int(t) for t in cells.ravel())
                if any(name in ("slab", "rib_silicon") for name in names):
                    triangles.extend(tuple(int(t) for t in cell) for cell in cells)

        contact_nodes: set[int] = set()
        for _dim, entity in gmsh.model.getEntities(1):
            names = entity_groups.get(-entity, [])
            if not any(name.startswith("contact_") for name in names):
                continue
            _types, _tags, element_nodes = gmsh.model.mesh.getElements(1, entity)
            for nodes in element_nodes:
                contact_nodes.update(int(t) for t in nodes)

        orphans = {int(t) for t in node_tags} - used_nodes
    finally:
        gmsh.clear()
        if not was_initialized:
            gmsh.finalize()

    return {
        "triangles": triangles,
        "contact_nodes": contact_nodes,
        "orphan_nodes": orphans,
    }


def _connected_components(triangles: list[tuple[int, ...]]) -> list[set[int]]:
    """Node sets of the connected components of a triangle soup."""
    adjacency: dict[int, set[int]] = {}
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)

    components: list[set[int]] = []
    unvisited = set(adjacency)
    while unvisited:
        stack = [unvisited.pop()]
        component = set(stack)
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour not in component:
                    component.add(neighbour)
                    unvisited.discard(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_unified_mesh_silicon_is_one_contacted_domain(tmp_path: Path) -> None:
    mesh_text, _vertices = _api.write_mesh()
    mesh_path = tmp_path / "unified.msh"
    mesh_path.write_text(mesh_text)
    topology = _silicon_mesh_topology(str(mesh_path))

    components = _connected_components(topology["triangles"])
    assert len(components) == 1, (
        f"silicon splits into {len(components)} disconnected pieces "
        f"(sizes {sorted(len(c) for c in components)}); the rib would float "
        "with no contact and never see the applied bias"
    )

    silicon_nodes = components[0]
    contact_nodes = topology["contact_nodes"]
    assert contact_nodes, "shared mesh carries no contact line elements"
    assert contact_nodes <= silicon_nodes, (
        "contact-line nodes sit outside the silicon domain: "
        f"{sorted(contact_nodes - silicon_nodes)}"
    )


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_unified_mesh_has_no_orphan_nodes(tmp_path: Path) -> None:
    mesh_text, _vertices = _api.write_mesh()
    mesh_path = tmp_path / "unified.msh"
    mesh_path.write_text(mesh_text)
    topology = _silicon_mesh_topology(str(mesh_path))
    assert not topology["orphan_nodes"], (
        "nodes belong to no cell: "
        f"{sorted(topology['orphan_nodes'])}; they leave zero rows in the "
        "eigenproblem and false contact geometry for ChargeTransport"
    )


# ---------------------------------------------------------------------------
# Mode field (ticket 07 headline figure)
# ---------------------------------------------------------------------------


def test_mode_field_requires_a_design_field() -> None:
    with pytest.raises(ValueError, match="design_epsilon is required"):
        apply(InputSchema(operation="mode_field"))


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_mode_field_returns_a_guided_profile_on_the_rib() -> None:
    """|E| peaks in the rib and carries most of its energy there."""
    outputs = apply(
        InputSchema(operation="mode_field", design_epsilon=_sized_design(0.0))
    )
    abs_e = np.asarray(outputs.mode_abs_e)
    coords = np.asarray(outputs.mode_coordinates)

    assert abs_e.shape == (coords.shape[0],)
    assert abs_e.min() >= 0.0
    assert np.isclose(abs_e.max(), 1.0), "the profile is normalized to its peak"

    neff = float(np.sqrt(outputs.neff_sq))
    assert _api.N_OXIDE < neff < _api.N_SILICON, f"mode is not guided (neff={neff})"

    peak = coords[int(np.argmax(abs_e))]
    y0 = -0.5 * sum(_api._LAYER_THICKNESS.values()) + (
        _api._LAYER_THICKNESS["substrate"] + _api._LAYER_THICKNESS["slab"]
    )
    assert abs(peak[0]) < _api._RIB_HALF_WIDTH, f"|E| peaks off the rib at {peak}"
    assert y0 - _api._LAYER_THICKNESS["slab"] < peak[1] < y0 + _api._LAYER_THICKNESS["rib"]

    in_rib_column = np.abs(coords[:, 0]) < _api._RIB_HALF_WIDTH
    energy = abs_e**2
    assert energy[in_rib_column].sum() / energy.sum() > 0.5


@pytest.mark.skipif(not _gyptis_available(), reason="gyptis/FEniCS not installed")
def test_mode_field_does_not_advance_the_tracked_branch() -> None:
    """A field query is read-only: it must not redirect the tracked mode."""
    design = _sized_design(0.0)
    apply(make_inputs(design))
    tracked = dict(_api._tracked_lam)
    assert tracked, "the forward solve records a tracked eigenvalue"

    apply(InputSchema(operation="mode_field", design_epsilon=design))
    assert _api._tracked_lam == tracked
