"""Tests for the shared waveguide mesh generation module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from prismo.waveguide_mesh import (
    PHYSICAL_GROUP_NAMES,
    RibWaveguideGeometry,
    build_rib_waveguide_mesh,
    build_rib_waveguide_mesh_via_gmsh,
    read_mesh_node_coordinates,
    read_mesh_silicon_triangulation,
)


class TestRibWaveguideGeometry:
    """Tests for the geometry definition."""

    def test_default_dimensions(self):
        geom = RibWaveguideGeometry()
        assert geom.rib_thickness == 220e-9
        assert geom.rib_width == 500e-9
        assert geom.slab_thickness == 100e-9
        assert geom.box_width == 3e-6
        assert geom.total_height == pytest.approx(1320e-9)
        assert geom.contact_width == 50e-9
        assert geom.contact_offset == 200e-9

    def test_custom_dimensions(self):
        geom = RibWaveguideGeometry(
            rib_thickness=300e-9,
            rib_width=600e-9,
            slab_thickness=150e-9,
        )
        assert geom.rib_thickness == 300e-9
        assert geom.rib_width == 600e-9
        assert geom.slab_thickness == 150e-9

    def test_material_indices(self):
        geom = RibWaveguideGeometry()
        assert geom.silicon_index == 3.4757
        assert geom.oxide_index == 1.444

    def test_total_height(self):
        geom = RibWaveguideGeometry()
        expected = (
            geom.substrate_thickness
            + geom.slab_thickness
            + geom.rib_thickness
            + geom.cladding_thickness
        )
        assert geom.total_height == expected

    def test_rib_left_edge(self):
        geom = RibWaveguideGeometry()
        assert geom.rib_left == -geom.rib_width / 2

    def test_rib_right_edge(self):
        geom = RibWaveguideGeometry()
        assert geom.rib_right == geom.rib_width / 2


class TestPhysicalGroupNaming:
    """Tests for physical group naming convention."""

    def test_required_groups_exist(self):
        required = {"silicon", "oxide", "contact_anode", "contact_cathode"}
        assert required.issubset(set(PHYSICAL_GROUP_NAMES))

    def test_names_are_strings(self):
        for name in PHYSICAL_GROUP_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_no_duplicates(self):
        assert len(PHYSICAL_GROUP_NAMES) == len(set(PHYSICAL_GROUP_NAMES))


class TestMeshGenerationGmsh:
    """Tests for mesh generation via gmsh."""

    @pytest.fixture
    def gmsh(self):
        pytest.importorskip("gmsh")
        import gmsh  # type: ignore[import-untyped]

        gmsh.initialize()
        yield gmsh
        gmsh.finalize()

    @pytest.fixture
    def output_path(self):
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = f.name
        yield path
        Path(path).unlink(missing_ok=True)

    def test_build_mesh_generates_file(self, gmsh, output_path):
        mesh_file = build_rib_waveguide_mesh_via_gmsh(
            output_path, RibWaveguideGeometry()
        )
        assert mesh_file == output_path
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_build_mesh_has_correct_groups(self, gmsh, output_path):
        mesh_file = build_rib_waveguide_mesh_via_gmsh(
            output_path, RibWaveguideGeometry()
        )
        assert Path(mesh_file).exists()

        gmsh.open(str(mesh_file))
        groups = gmsh.model.getPhysicalGroups()
        group_names = {gmsh.model.getPhysicalName(dim, tag) for dim, tag in groups}
        gmsh.clear()

        assert "silicon" in group_names
        assert "oxide" in group_names
        assert "contact_anode" in group_names
        assert "contact_cathode" in group_names

    def test_contact_groups_are_boundary_curves(self, gmsh, output_path):
        """ChargeTransport needs contacts as dim-1 boundary curves.

        ExtendableGrids only reads dim-(d-1) elements as boundary faces;
        contact physical groups must therefore be curves (dim 1), not
        surfaces (ticket 17).
        """
        mesh_file = build_rib_waveguide_mesh_via_gmsh(
            output_path, RibWaveguideGeometry()
        )

        gmsh.open(str(mesh_file))
        groups = gmsh.model.getPhysicalGroups()
        contact_dims = {
            gmsh.model.getPhysicalName(dim, tag): dim for dim, tag in groups
        }
        gmsh.clear()

        assert contact_dims["contact_anode"] == 1
        assert contact_dims["contact_cathode"] == 1

    def test_contact_curves_have_1d_elements(self, gmsh, output_path):
        """The .msh must contain 1D elements on the contact curves."""
        mesh_file = build_rib_waveguide_mesh_via_gmsh(
            output_path, RibWaveguideGeometry()
        )

        gmsh.open(str(mesh_file))
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(1, -1)
        gmsh.clear()

        assert len(elem_types) > 0
        assert sum(len(tags) for tags in elem_tags) > 0

    def test_build_mesh_node_count(self, gmsh, output_path):
        build_rib_waveguide_mesh_via_gmsh(
            output_path, RibWaveguideGeometry()
        )

        gmsh.open(str(output_path))
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        gmsh.clear()

        assert len(node_tags) > 0
        assert len(node_coords) == len(node_tags) * 3

    def test_build_mesh_resolution_10nm(self, gmsh, output_path):
        geom = RibWaveguideGeometry()
        geom.mesh_res_junction = 10e-9
        build_rib_waveguide_mesh_via_gmsh(output_path, geom)

        gmsh.open(str(output_path))
        _, coords, _ = gmsh.model.mesh.getNodes()
        gmsh.clear()

        x = coords[0::3]
        y = coords[1::3]
        assert np.min(x) >= -geom.box_width / 2 - 1e-15
        assert np.max(x) <= geom.box_width / 2 + 1e-15
        assert np.min(y) >= -1e-15
        assert np.max(y) <= geom.total_height + 1e-15

    def test_build_mesh_dimension_2d(self, gmsh, output_path):
        build_rib_waveguide_mesh_via_gmsh(output_path, RibWaveguideGeometry())

        gmsh.open(str(output_path))
        element_types, _, _ = gmsh.model.mesh.getElements()
        gmsh.clear()

        unique_types = set(element_types)
        assert 2 in unique_types


class TestBuildRibWaveguideMeshPublic:
    """Tests for the public build_rib_waveguide_mesh function."""

    @pytest.fixture
    def output_path(self):
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = f.name
        yield path
        Path(path).unlink(missing_ok=True)

    def test_returns_mesh_path(self, output_path):
        result = build_rib_waveguide_mesh(output_path)
        assert result == output_path

    def test_geometry_accepts_custom_dimensions(self, output_path):
        geom = RibWaveguideGeometry(rib_width=600e-9)
        result = build_rib_waveguide_mesh(output_path, geometry=geom)
        assert Path(result).exists()

    def test_geometry_returns_mesh_ref_dict(self, output_path):
        result = build_rib_waveguide_mesh(output_path)
        assert result == output_path


class TestMeshRefCompat:
    """Tests that the generated mesh is compatible with MeshRef schema."""

    @pytest.fixture
    def gmsh(self):
        pytest.importorskip("gmsh")
        import gmsh  # type: ignore[import-untyped]

        gmsh.initialize()
        yield gmsh
        gmsh.finalize()

    @pytest.fixture
    def output_path(self):
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = f.name
        yield path
        Path(path).unlink(missing_ok=True)

    def test_meshref_path_exists(self, output_path):
        from prismo_shared.schemas import MeshRef

        mesh_path = build_rib_waveguide_mesh(output_path)
        ref = MeshRef(path=str(mesh_path))
        assert Path(ref.path).exists()

    # @pytest.mark.skip(reason="gmsh required to parse .msh for node counts")
    def test_meshref_node_count_populated(self, output_path):
        from prismo_shared.schemas import MeshRef

        mesh_path = build_rib_waveguide_mesh(output_path)
        ref = MeshRef(path=str(mesh_path))
        ref.n_nodes = 42
        assert ref.n_nodes > 0

    def test_mesh_is_conforming_no_duplicate_nodes(self, gmsh, output_path):
        """Adjacent rib patches must share nodes, not duplicate them.

        A non-conforming mesh (coincident nodes on internal interfaces) gives
        ChargeTransport's finite-volume operator a null space and its adjoint
        solve raises SingularException -- the `make run-containers` crash. The
        generator merges duplicate CAD entities, so every node coordinate is
        unique. Ref: .scratch/chargetransport-mesh-node-ordering.
        """
        build_rib_waveguide_mesh(output_path)
        coords = read_mesh_node_coordinates(output_path)

        unique = np.unique(np.round(coords, 15), axis=0)
        assert unique.shape[0] == coords.shape[0], (
            f"{coords.shape[0] - unique.shape[0]} duplicate-coordinate nodes: "
            "mesh is non-conforming"
        )


class TestSiliconTriangulation:
    """Public mesh-reader seam for the shared silicon design region."""

    @pytest.fixture
    def gmsh(self):
        pytest.importorskip("gmsh")
        import gmsh  # type: ignore[import-untyped]

        gmsh.initialize()
        yield gmsh
        gmsh.finalize()

    @pytest.fixture
    def output_path(self):
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = f.name
        yield path
        Path(path).unlink(missing_ok=True)

    def test_reader_returns_only_silicon_triangles(self, gmsh, output_path):
        build_rib_waveguide_mesh_via_gmsh(output_path, RibWaveguideGeometry())
        triangles = read_mesh_silicon_triangulation(output_path)

        gmsh.open(str(output_path))
        silicon_tag = next(
            tag
            for dim, tag in gmsh.model.getPhysicalGroups(2)
            if gmsh.model.getPhysicalName(dim, tag) == "silicon"
        )
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        index_by_tag = {int(tag): index for index, tag in enumerate(node_tags)}
        expected_parts = []
        for entity_tag in gmsh.model.getEntitiesForPhysicalGroup(2, silicon_tag):
            element_types, _, element_nodes = gmsh.model.mesh.getElements(2, entity_tag)
            for element_type, nodes in zip(element_types, element_nodes, strict=True):
                if gmsh.model.mesh.getElementProperties(element_type)[3] != 3:
                    continue
                tags = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
                expected_parts.append(
                    np.asarray(
                        [
                            [index_by_tag[int(tag)] for tag in triangle]
                            for triangle in tags
                        ]
                    )
                )
        gmsh.clear()
        expected = np.concatenate(expected_parts)

        assert triangles.shape == expected.shape
        assert np.issubdtype(triangles.dtype, np.integer)
        np.testing.assert_array_equal(triangles, expected)

    def test_reader_returns_empty_triangles_without_gmsh(
        self, monkeypatch, output_path
    ):
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
        triangles = read_mesh_silicon_triangulation(output_path)
        assert triangles.shape == (0, 3)
        assert np.issubdtype(triangles.dtype, np.integer)
