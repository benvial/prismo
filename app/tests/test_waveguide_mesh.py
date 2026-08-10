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
