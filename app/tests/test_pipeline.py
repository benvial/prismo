"""Tests for the end-to-end differentiable pipeline.

Ref: ticket 14 -- end-to-end pipeline via JAX composition.
"""

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from prismo.density_filter import assemble_filter_matrix  # noqa: E402
from prismo.differentiable_component import DifferentiableComponent  # noqa: E402
from prismo.mesh_transfer import build_mesh_transfer_operator  # noqa: E402
from prismo.pipeline import (  # noqa: E402
    _CT_MESH_MOUNT,
    PipelineComponents,
    _build_design_epsilon,
    _container_mesh_ref,
    _sb_jax,
    build_chargetransport_component,
    build_default_components,
    build_design_transfer,
    build_gyptis_components,
    init_tesseract_containers,
    pipeline,
    read_gyptis_design_cell_centroids,
)
from prismo.soref_bennett import soref_bennett as _sb_numpy  # noqa: E402
from prismo_shared.schemas import CarrierDensityField, MeshRef  # noqa: E402

RNG = np.random.default_rng(0)


def _components_with(**overrides) -> PipelineComponents:
    """Default in-process components with specific components replaced."""
    return replace(build_default_components(), **overrides)


def test_pipeline_enables_float64() -> None:
    """Effective-index differences must not be quantized at float32 epsilon."""
    assert jax.config.jax_enable_x64


def _grid_coords(n: int = 16, spacing: float = 20e-9) -> np.ndarray:
    xs, ys = np.meshgrid(
        np.arange(n) * spacing,
        np.arange(n) * spacing,
        indexing="xy",
    )
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


class TestSorefBennettJax:
    """Pure-JAX Soref-Bennett matches the numpy reference."""

    N_NODES = 8

    def test_zero_perturbation(self):
        es = jnp.ones(self.N_NODES)
        hs = jnp.ones(self.N_NODES)
        eq_e = jnp.ones(self.N_NODES)
        eq_h = jnp.ones(self.N_NODES)

        depsilon, dalpha = _sb_jax(es, hs, eq_e, eq_h)
        np.testing.assert_allclose(depsilon, 0.0, atol=0.0)
        np.testing.assert_allclose(dalpha, 0.0, atol=0.0)

    def test_matches_numpy_reference(self):
        n = jnp.asarray(RNG.random(self.N_NODES) * 1e24 + 1e20)
        p = jnp.asarray(RNG.random(self.N_NODES) * 1e24 + 1e20)
        n_eq = jnp.asarray(RNG.random(self.N_NODES) * 1e21)
        p_eq = jnp.asarray(RNG.random(self.N_NODES) * 1e21)

        depsilon_jax, dalpha_jax = _sb_jax(n, p, n_eq, p_eq)

        result_np = _sb_numpy(
            CarrierDensityField(
                electrons=np.asarray(n).tolist(),
                holes=np.asarray(p).tolist(),
                equilibrium_electrons=np.asarray(n_eq).tolist(),
                equilibrium_holes=np.asarray(p_eq).tolist(),
            ),
        )
        np.testing.assert_allclose(
            depsilon_jax,
            np.asarray(result_np.delta_permittivity),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            dalpha_jax,
            np.asarray(result_np.delta_absorption),
            rtol=1e-12,
        )

    def test_differentiable_elements(self):
        electrons = jnp.asarray(RNG.random(self.N_NODES) * 1e24 + 1e20)
        holes = jnp.asarray(RNG.random(self.N_NODES) * 1e24 + 1e20)
        eq_e = jnp.asarray(RNG.random(self.N_NODES) * 1e21)
        eq_h = jnp.asarray(RNG.random(self.N_NODES) * 1e21)

        def loss(e):
            depsilon, _ = _sb_jax(e, holes, eq_e, eq_h)
            return jnp.sum(depsilon)

        grad = jax.grad(loss)(electrons)
        assert grad.shape == electrons.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_depletion_increases_permittivity(self):
        """Reverse bias DEPLETES carriers (dn < 0); the index must rise.

        Hand-computed for depletion of Delta_N = 1e18 cm^-3 (input densities
        in m^-3): the Soref-Bennett model extended antisymmetrically gives
        dn_index = +(A_e * dN^B_e + A_h * dN^B_h) with dN = 1e18.
        Ref: ticket 17 — clamping dn at zero made Delta_eps identically zero
        under reverse bias.
        """
        from prismo.pipeline import _DEFAULT_COEFFS

        coeffs = _DEFAULT_COEFFS
        dN = 1e18  # cm^-3 depletion magnitude

        # Fully depleted: carriers drop from 1e24 m^-3 (= 1e18 cm^-3) to 0.
        depletion = jnp.zeros(self.N_NODES)
        eq = jnp.full(self.N_NODES, 1e24)

        depsilon, dalpha = _sb_jax(depletion, depletion, eq, eq)

        exp_dn = coeffs.A_e * dN**coeffs.B_e + coeffs.A_h * dN**coeffs.B_h
        exp_depsilon = 2.0 * coeffs.background_index * exp_dn
        np.testing.assert_allclose(
            depsilon,
            jnp.full(self.N_NODES, exp_depsilon),
            rtol=1e-6,
        )
        assert jnp.all(depsilon > 0)
        # Depletion removes absorption: dalpha < 0.
        exp_dalpha = -(coeffs.C_e * dN**coeffs.D_e + coeffs.C_h * dN**coeffs.D_h)
        np.testing.assert_allclose(
            dalpha,
            jnp.full(self.N_NODES, exp_dalpha),
            rtol=1e-6,
        )

    def test_antisymmetric_in_density_change(self):
        """sb(-dn) == -sb(+dn): injection and depletion are opposite."""
        eq_e = jnp.full(self.N_NODES, 1e24)
        eq_h = jnp.full(self.N_NODES, 1e24)
        up_e = eq_e * 1.5
        up_h = eq_h * 1.5
        down_e = eq_e * 0.5
        down_h = eq_h * 0.5

        dep_up, da_up = _sb_jax(up_e, up_h, eq_e, eq_h)
        dep_down, da_down = _sb_jax(down_e, down_h, eq_e, eq_h)

        np.testing.assert_allclose(dep_down, -dep_up, rtol=1e-6)
        np.testing.assert_allclose(da_down, -da_up, rtol=1e-6)


class TestPipelineStub:
    """Pipeline with no containers available (stubs)."""

    N_NODES = 16

    @pytest.fixture
    def rho(self) -> jax.Array:
        return jnp.asarray(RNG.random(self.N_NODES), dtype=jnp.float64)

    def test_forward_returns_scalar(self, rho):
        result = pipeline(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_ct_units_converted_for_soref(self, rho):
        """CT reports carriers in cm^-3; Soref-Bennett expects m^-3.

        Stub the external solver boundary: 0 V returns 1e18 cm^-3 carriers,
        -5 V returns full depletion. The expected Delta n_eff is
        hand-computed from the Soref-Bennett model with dN = 1e18 cm^-3 —
        if the cm^-3 -> m^-3 conversion were missing, the result would be
        1e6x too small (ticket 17).
        """
        import prismo.pipeline as pl

        n_nodes = self.N_NODES
        hi = jnp.full((n_nodes,), 1e18)  # cm^-3, from CT
        lo = jnp.zeros((n_nodes,))

        def fake_ct(doping, bias_voltage, mesh_ref=None):
            if bias_voltage == 0.0:
                return hi, hi
            return lo, lo

        result = pipeline(rho, components=_components_with(chargetransport=fake_ct))

        coeffs = pl._DEFAULT_COEFFS
        dN = 1e18  # cm^-3 depletion
        exp_dn = coeffs.A_e * dN**coeffs.B_e + coeffs.A_h * dN**coeffs.B_h
        exp_deps = 2.0 * coeffs.background_index * exp_dn
        bg = pl._DEFAULT_BACKGROUND_EPSILON
        # No mean-collapse: the uniform depletion perturbs the whole design
        # field, and the effective-medium stub averages it back to bg + exp_deps.
        exp_result = float(np.sqrt(bg + exp_deps) - np.sqrt(bg))
        np.testing.assert_allclose(float(result), exp_result, rtol=1e-6)
        assert float(result) > 0.0

    def test_polarity_forms_mixed_sign_pn_doping(self):
        """Container pipeline supplies a fixed P/N polarity per mesh node."""
        received: list[np.ndarray] = []

        def fake_ct(doping, bias_voltage, mesh_ref=None):
            received.append(np.asarray(doping))
            carriers = jnp.where(
                bias_voltage == 0.0,
                jnp.full_like(doping, 1e24),
                jnp.zeros_like(doping),
            )
            return carriers, carriers

        result = pipeline(
            jnp.asarray([0.0, 1.0]),
            polarity=jnp.asarray([-1.0, 1.0]),
            components=_components_with(chargetransport=fake_ct),
        )

        np.testing.assert_allclose(received[0], [-1e14, 1e21])
        np.testing.assert_allclose(received[1], [-1e14, 1e21])
        assert float(result) > 0.0

    def test_effective_index_objective_is_positive_for_either_shift_direction(
        self,
    ):
        """Optimization maximizes phase-shift magnitude, not its mode-sign."""

        def fake_ct(doping, bias_voltage, mesh_ref=None):
            carriers = jnp.where(
                bias_voltage == 0.0,
                jnp.zeros_like(doping),
                jnp.full_like(doping, 1e24),
            )
            return carriers, carriers

        result = pipeline(
            jnp.asarray([0.25, 0.25]),
            components=_components_with(chargetransport=fake_ct),
        )

        assert float(result) > 0.0

    def test_effective_index_objective_has_zero_gradient_at_zero_shift(self):
        """The positive phase-shift objective stays differentiable at zero."""

        def fake_ct(doping, bias_voltage, mesh_ref=None):
            return doping, doping

        components = _components_with(chargetransport=fake_ct)
        rho = jnp.asarray([0.25, 0.25])
        grad = jax.grad(lambda r: pipeline(r, components=components))(rho)
        np.testing.assert_allclose(grad, 0.0, atol=1e-20)

    def test_design_epsilon_preserves_spatial_structure(self):
        """No mean-collapse: each design cell keeps its own perturbation."""
        delta = jnp.asarray([1.0, 3.0, 5.0])
        bg, pert = _build_design_epsilon(delta, jnp.asarray(12.0))
        np.testing.assert_array_equal(bg, [12.0, 12.0, 12.0])
        np.testing.assert_allclose(pert, [13.0, 15.0, 17.0])

    def test_design_epsilon_background_is_uniform(self):
        """The background field stays uniform -> exact zero design gradient."""
        bg, _ = _build_design_epsilon(jnp.asarray([1.0, -2.0, 0.5]), jnp.asarray(10.0))
        np.testing.assert_array_equal(bg, [10.0, 10.0, 10.0])

    def test_design_epsilon_applies_mesh_transfer(self):
        """A transfer matrix maps the nodal perturbation onto the design cells."""
        delta = jnp.asarray([1.0, 3.0])  # two shared-mesh nodes
        transfer = jnp.asarray([[1.0, 0.0], [0.5, 0.5]])  # 2 design cells
        _, pert = _build_design_epsilon(delta, jnp.asarray(10.0), transfer)
        np.testing.assert_allclose(pert, [11.0, 12.0])

    def test_design_epsilon_fixed_mean_pattern_is_not_averaged(self):
        """Fixed-mean but varying perturbation stays spatially non-constant."""
        varying = jnp.asarray([0.3, -0.3, 0.1, -0.1])  # mean 0, structured
        uniform = jnp.zeros(4)  # same mean, no structure
        _, pert_varying = _build_design_epsilon(varying, jnp.asarray(12.0))
        _, pert_uniform = _build_design_epsilon(uniform, jnp.asarray(12.0))
        assert float(jnp.std(pert_varying)) > 0.0
        assert float(jnp.std(pert_uniform)) == 0.0
        # A structure-sensitive solve distinguishes them at equal mean.
        sq = lambda e: float(jnp.sum(e**2))
        assert sq(pert_varying) != pytest.approx(sq(pert_uniform))

    def test_pipeline_gradient_is_spatially_resolved(self):
        """End to end: a structure-sensitive gyptis yields a non-constant grad.

        With the mean-collapse removed, the per-cell design field reaches gyptis
        and its per-cell sensitivity flows back to a spatially-varying rho
        gradient -- the payoff the feature exists to enable (ticket 05).
        """

        def fake_ct(doping, bias_voltage, mesh_ref=None):
            # Carriers deplete under bias in proportion to the local doping, so
            # a non-uniform rho produces a spatially-varying permittivity field.
            carriers = jnp.where(bias_voltage == 0.0, doping, jnp.zeros_like(doping))
            return carriers, carriers

        # gyptis stub sensitive to spatial structure: neff_sq = sum(eps^2), whose
        # exact per-cell VJP is 2*eps (non-constant for a structured field).
        structure_sensitive = DifferentiableComponent(
            forward=lambda *a: None,
            vjp=lambda *a: None,
            out_struct=lambda eps, *s: jax.ShapeDtypeStruct((), eps.dtype),
            stub_forward=lambda eps, *s: jnp.sum(eps**2),
            stub_vjp=lambda eps, g, *s: 2.0 * g * eps,
            available=lambda: False,
        )
        components = _components_with(
            chargetransport=fake_ct, gyptis=structure_sensitive
        )

        rho = jnp.asarray([0.2, 0.5, 0.8, 0.4])
        grad = jax.grad(lambda r: pipeline(r, components=components))(rho)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.std(grad)) > 0.0  # spatially resolved, not collapsed

    def test_forward_returns_zero_pre_container(self, rho):
        result = pipeline(rho)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_gradient_is_finite(self, rho):
        grad = jax.grad(pipeline)(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_is_zero_pre_container(self, rho):
        grad = jax.grad(pipeline)(rho)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_jit_works(self, rho):
        jitted = jax.jit(pipeline)
        result = jitted(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_jit_gradient_works(self, rho):
        grad_fn = jax.jit(jax.grad(pipeline))
        grad = grad_fn(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_dtype_preserved(self, rho):
        result32 = pipeline(jnp.asarray(rho, dtype=jnp.float32))
        assert result32.dtype == jnp.float32

        result64 = pipeline(jnp.asarray(rho, dtype=jnp.float64))
        assert result64.dtype == jnp.float64


class TestContainerPipeline:
    def test_reads_design_centroids_from_gyptis_container(self):
        class FakeGyptis:
            def __init__(self):
                self.inputs = None

            def apply(self, inputs):
                self.inputs = inputs
                return {"design_cell_centroids": [[-0.1, 0.2], [0.1, 0.2]]}

        gyptis = FakeGyptis()
        centroids = read_gyptis_design_cell_centroids(container=gyptis)

        assert gyptis.inputs == {"operation": "design_cell_centroids"}
        np.testing.assert_allclose(centroids, [[-0.1, 0.2], [0.1, 0.2]])

    def test_background_eigenmode_is_cached_across_pipeline_calls(self):
        """Only the rho-independent background solve survives a callback."""

        class FakeChargeTransport:
            def apply(self, inputs):
                doping = np.asarray(inputs["doping"], dtype=float)
                carrier = np.full_like(
                    doping,
                    1e18 if inputs["bias_voltage"] == 0.0 else 0.0,
                )
                return {"electrons": carrier, "holes": carrier}

        class FakeGyptis:
            def __init__(self):
                self.epsilon_calls: list[np.ndarray] = []

            def apply(self, inputs):
                epsilon = np.asarray(inputs["design_epsilon"], dtype=float)
                self.epsilon_calls.append(epsilon)
                return {"neff_sq": float(np.mean(epsilon))}

        gyptis = FakeGyptis()

        def build_components() -> PipelineComponents:
            chargetransport = build_chargetransport_component(
                container=FakeChargeTransport()
            )
            perturbed, background = build_gyptis_components(container=gyptis)
            return PipelineComponents(
                chargetransport=chargetransport,
                gyptis=perturbed,
                gyptis_background=background,
            )

        components = build_components()
        rho = jnp.full((4,), 0.25)
        pipeline(rho, components=components)
        pipeline(rho, components=components)

        # Two perturbed solves plus one reused rho-independent background solve.
        assert len(gyptis.epsilon_calls) == 3

        # A fresh bundle starts a new lifecycle: its background cache is empty.
        pipeline(rho, components=build_components())
        assert len(gyptis.epsilon_calls) == 5

    def test_container_startup_failure_raises(self, monkeypatch):
        """Container mode must not continue through a local stub."""
        import sys
        import types

        import prismo.pipeline as pl

        class FailingTesseract:
            @classmethod
            def from_image(cls, image, **kwargs):
                raise RuntimeError(f"image unavailable: {image}")

        monkeypatch.setitem(
            sys.modules,
            "tesseract_core",
            types.SimpleNamespace(Tesseract=FailingTesseract),
        )

        with pytest.raises(RuntimeError, match="ChargeTransport container"):
            pl.init_tesseract_containers()

    def test_gyptis_container_failure_does_not_use_local_stub(self):
        """A failed container request must abort rather than fall to a stub."""

        class FailingTesseract:
            def apply(self, inputs):
                raise RuntimeError("HTTP 500")

        perturbed, _ = build_gyptis_components(container=FailingTesseract())

        with pytest.raises(RuntimeError, match="HTTP 500"):
            perturbed.forward(np.array([12.0, 12.0]))

    def test_two_configured_pipelines_coexist_without_interfering(self):
        """Differently-configured bundles run in one process independently."""

        class FakeGyptis:
            def apply(self, inputs):
                return {"neff_sq": float(np.mean(inputs["design_epsilon"]))}

        def make_bundle(carriers_0v: float) -> PipelineComponents:
            def fake_ct(doping, bias_voltage, mesh_ref=None):
                value = carriers_0v if bias_voltage == 0.0 else 0.0
                filled = jnp.full_like(doping, value)
                return filled, filled

            perturbed, background = build_gyptis_components(container=FakeGyptis())
            return PipelineComponents(
                chargetransport=fake_ct,
                gyptis=perturbed,
                gyptis_background=background,
            )

        depleting = make_bundle(1e24)  # carriers at 0 V, depleted at -5 V
        flat = make_bundle(0.0)  # no carrier change between biases

        rho = jnp.full((4,), 0.25)
        result_depleting = float(pipeline(rho, components=depleting))
        result_flat = float(pipeline(rho, components=flat))

        # Each pipeline reflects its own components; no shared mutable state.
        assert result_depleting > 0.0
        np.testing.assert_allclose(result_flat, 0.0, atol=1e-12)
        # Re-running the first bundle still yields its own result.
        assert float(pipeline(rho, components=depleting)) == pytest.approx(
            result_depleting
        )

    def test_wired_design_transfer_drives_spatial_gradient_through_gyptis(self):
        """Ticket 08 payoff, exercised through the container gyptis path.

        The ``design_transfer`` matrix built by ``build_mesh_transfer_operator``
        carries a spatially varying, fixed-mean perturbation onto the gyptis
        design cells, so a real (container-routed) eigenmode solve yields a
        spatially non-constant design gradient while the background solve still
        sees a uniform field. Unlike ``test_pipeline_gradient_is_spatially_
        resolved`` -- which uses the JAX gyptis stub -- this drives the container
        forward/VJP built by ``build_gyptis_components`` and the transfer wiring
        ticket 08 adds.
        """
        # Shared mesh: a unit square split into two silicon triangles.
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
        silicon_triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp)
        # Two design cells, one interior to each triangle (distinct node mixes).
        design_centroids = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=float)
        design_transfer = jnp.asarray(
            build_mesh_transfer_operator(
                coords, silicon_triangles, design_centroids
            ).dense()
        )
        assert design_transfer.shape == (2, 4)  # design cells <- shared-mesh nodes

        applied_fields: list[np.ndarray] = []

        class StructureSensitiveGyptis:
            """A gyptis backend whose neff_sq responds to spatial structure."""

            def apply(self, inputs):
                eps = np.asarray(inputs["design_epsilon"], dtype=float)
                applied_fields.append(eps)
                return {"neff_sq": float(np.sum(eps**2))}

            def vector_jacobian_product(
                self, inputs, inputs_to_diff, outputs, cotangents
            ):
                eps = np.asarray(inputs["design_epsilon"], dtype=float)
                cot = float(cotangents["neff_sq"])
                return {"design_epsilon": (2.0 * cot * eps).tolist()}

        def fake_ct(doping, bias_voltage, mesh_ref=None):
            # Carriers deplete under bias in proportion to the local doping, so a
            # spatially varying rho makes a spatially varying permittivity field.
            carriers = jnp.where(bias_voltage == 0.0, doping, jnp.zeros_like(doping))
            return carriers, carriers

        perturbed, background = build_gyptis_components(
            container=StructureSensitiveGyptis()
        )
        components = PipelineComponents(
            chargetransport=fake_ct, gyptis=perturbed, gyptis_background=background
        )

        def run(rho: jax.Array) -> jax.Array:
            return pipeline(rho, design_transfer=design_transfer, components=components)

        # Fixed-mean but spatially varying design vector.
        rho = jnp.asarray([0.2, 0.8, 0.6, 0.4])
        grad = jax.grad(run)(rho)
        assert jnp.all(jnp.isfinite(grad))
        # The design gradient is spatially resolved, not collapsed to a scalar.
        assert float(jnp.std(grad)) > 0.0

        # Every solve saw a field of exactly the design-cell count (not the node
        # count) -- the length the real forward's size guard enforces.
        assert applied_fields
        assert all(eps.shape == (2,) for eps in applied_fields)
        # The perturbed solve saw a structured field; the background a uniform one.
        assert any(float(np.std(eps)) > 0.0 for eps in applied_fields)
        assert any(float(np.std(eps)) == 0.0 for eps in applied_fields)

        # The background component contributes an exact zero design gradient
        # regardless of its field, preserving the rho-independence argument.
        bg_cotangent = background.vjp(np.full(2, 12.0), np.asarray(1.0))
        np.testing.assert_array_equal(bg_cotangent, np.zeros(2))

        # Payoff: a fixed-mean redistribution (a permutation of rho) moves neff --
        # the retired mean-collapse path could not have distinguished them.
        permuted = jnp.asarray([0.8, 0.2, 0.4, 0.6])
        np.testing.assert_allclose(float(jnp.mean(permuted)), float(jnp.mean(rho)))
        assert float(run(rho)) != pytest.approx(float(run(permuted)))


class TestBuildDesignTransfer:
    """The container setup assembles the real mesh-transfer matrix (ticket 08)."""

    def test_assembles_real_operator_from_centroids_and_triangulation(
        self, monkeypatch
    ):
        """The production helper drives the real ``build_mesh_transfer_operator``.

        Both static inputs come from live sources -- centroids through the
        component seam, the silicon triangulation from the mesh reader -- so
        this covers the real assembly the container run performs (only the
        gmsh-dependent triangulation read is stubbed).
        """
        import prismo.waveguide_mesh as mesh_module

        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp)
        monkeypatch.setattr(
            mesh_module, "read_mesh_silicon_triangulation", lambda _: triangles
        )
        components = PipelineComponents(
            chargetransport=None,
            gyptis=None,
            gyptis_background=None,
            design_cell_centroids=lambda: np.array([[0.25, 0.25], [0.75, 0.75]]),
        )

        transfer = build_design_transfer(components, coords, "unused.msh")

        assert transfer.shape == (2, 4)  # design cells <- shared-mesh nodes
        # Partition of unity: each design cell's weights sum to one, so a uniform
        # nodal field maps to a uniform design field (keyed to the shared mesh).
        np.testing.assert_allclose(np.asarray(transfer).sum(axis=1), 1.0)

    def test_requires_a_centroid_reader(self):
        """Without a gyptis backend there is no design geometry to key against."""
        components = PipelineComponents(
            chargetransport=None, gyptis=None, gyptis_background=None
        )
        with pytest.raises(RuntimeError, match="design-cell centroids"):
            build_design_transfer(components, np.zeros((3, 2)), "unused.msh")


class TestComponentTiming:
    """Phase timing is owned by the components, read via the bundle."""

    def _fake_ct(self):
        class FakeChargeTransport:
            def apply(self, inputs):
                doping = np.asarray(inputs["doping"], dtype=float)
                carrier = np.full_like(
                    doping, 1e18 if inputs["bias_voltage"] == 0.0 else 0.0
                )
                return {"electrons": carrier, "holes": carrier}

        return build_chargetransport_component(container=FakeChargeTransport())

    def test_component_records_its_own_forward_tallies(self):
        ct = self._fake_ct()
        ct(jnp.full((4,), 0.25), 0.0)
        ct(jnp.full((4,), 0.25), -5.0)

        timings = ct.timer.collect()
        assert timings["ct_forward_0V"]["calls"] == 1
        assert timings["ct_forward_-5V"]["calls"] == 1

    def test_collect_resets_between_callbacks_but_keeps_cold_memory(self):
        ct = self._fake_ct()
        ct(jnp.full((4,), 0.25), 0.0)

        first = ct.timer.collect()
        assert first["ct_forward_0V"]["cold_calls"] == 1
        # A fresh collect with no new calls is empty.
        assert ct.timer.collect() == {}

        # The second call is warm: the phase is no longer cold.
        ct(jnp.full((4,), 0.25), 0.0)
        second = ct.timer.collect()
        assert second["ct_forward_0V"]["calls"] == 1
        assert second["ct_forward_0V"]["cold_calls"] == 0

    def test_bundle_merges_component_timings(self):
        gyptis_recorder = []

        class FakeGyptis:
            def apply(self, inputs):
                gyptis_recorder.append(inputs["design_epsilon"])
                return {"neff_sq": float(np.mean(inputs["design_epsilon"]))}

        perturbed, background = build_gyptis_components(container=FakeGyptis())
        components = PipelineComponents(
            chargetransport=self._fake_ct(),
            gyptis=perturbed,
            gyptis_background=background,
        )
        pipeline(jnp.full((4,), 0.25), components=components)

        merged = components.collect_phase_timing()
        # CT and gyptis boundary phases are collected from their owning timers.
        assert merged["ct_forward_0V"]["calls"] == 1
        assert merged["ct_forward_-5V"]["calls"] == 1
        assert "gyptis_perturbed_forward" in merged
        assert "gyptis_background_forward" in merged


class TestPipelineWithFilter:
    """Pipeline with a synthetic density-filter matrix."""

    N_SIDE = 8

    @pytest.fixture
    def coords(self) -> np.ndarray:
        return _grid_coords(self.N_SIDE)

    @pytest.fixture
    def H_sparse(self, coords):
        return assemble_filter_matrix(coords)

    @pytest.fixture
    def H_dense(self, H_sparse):
        return jnp.asarray(H_sparse.toarray(), dtype=jnp.float64)

    @pytest.fixture
    def H_sum(self, H_dense):
        return jnp.sum(H_dense, axis=1)

    @pytest.fixture
    def n_nodes(self) -> int:
        return self.N_SIDE * self.N_SIDE

    @pytest.fixture
    def rho(self, n_nodes) -> jax.Array:
        return jnp.full((n_nodes,), 0.25, dtype=jnp.float64)

    def test_pipeline_with_filter_runs(self, rho, H_dense, H_sum):
        result = pipeline(rho, H=H_dense, H_sum=H_sum)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_gradient_with_filter(self, rho, H_dense, H_sum):
        grad = jax.grad(
            lambda r: pipeline(r, H=H_dense, H_sum=H_sum),
        )(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_with_filter(self, rho, H_dense, H_sum):
        fn = jax.jit(lambda r: pipeline(r, H=H_dense, H_sum=H_sum))
        result = fn(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_box_bounds_preserved(self, rho, H_dense, H_sum):
        for val in [0.0, 0.5, 1.0]:
            r = jnp.full_like(rho, val)
            result = pipeline(r, H=H_dense, H_sum=H_sum)
            assert jnp.isfinite(result)

    def test_shape_consistency(self, rho, H_dense, H_sum):
        result = pipeline(rho, H=H_dense, H_sum=H_sum)
        d = float(result)
        assert isinstance(d, float)


class TestPipelineDopingMapping:
    """The doping mapping N = 10^(14 + 7*rho) produces sane values."""

    def test_range(self):
        n_nodes = 10
        rho = jnp.linspace(0.0, 1.0, n_nodes)
        result = pipeline(rho)
        assert jnp.isfinite(result)
        assert result.ndim == 0

    def test_edge_cases_finite(self):
        for val in [0.0, 0.001, 0.5, 0.999, 1.0]:
            rho = jnp.array([val])
            result = pipeline(rho)
            assert jnp.isfinite(result), f"Non-finite at rho={val}"

    def test_gradient_exists_for_nonuniform_rho(self):
        n_nodes = 10
        rho = jnp.asarray(RNG.random(n_nodes), dtype=jnp.float64)
        grad = jax.grad(pipeline)(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))


class TestPipelineGradientValidation:
    """Finite-difference validation of the full-pipeline gradient.

    Compares jax.grad(pipeline) against central finite differences for
    random perturbation directions at random rho vectors.
    """

    N_NODES = 10
    N_DIRECTIONS = 3

    @pytest.fixture
    def rho_vectors(self) -> list[jax.Array]:
        return [
            jnp.asarray(RNG.random(self.N_NODES), dtype=jnp.float64)
            for _ in range(self.N_DIRECTIONS)
        ]

    @pytest.fixture
    def directions(self) -> list[jax.Array]:
        dirs = []
        for _ in range(self.N_DIRECTIONS):
            d = jnp.asarray(RNG.standard_normal(self.N_NODES), dtype=jnp.float64)
            d = d / jnp.linalg.norm(d)
            dirs.append(d)
        return dirs

    def test_gradient_vs_fd_central(self, rho_vectors, directions):
        step_sizes = [1e-3, 1e-4, 1e-5, 1e-6]
        grad_fn = jax.grad(pipeline)

        for rho in rho_vectors:
            grad_exact = grad_fn(rho)
            for direction in directions:
                for h in step_sizes:
                    f_plus = pipeline(rho + h * direction)
                    f_minus = pipeline(rho - h * direction)
                    fd_grad_dir = (f_plus - f_minus) / (2.0 * h)
                    exact_grad_dir = jnp.dot(grad_exact, direction)

                    rel_error = float(
                        abs(fd_grad_dir - exact_grad_dir)
                        / max(abs(exact_grad_dir), 1e-30)
                    )
                    assert jnp.isfinite(rel_error)

                    if h <= 1e-4:
                        assert rel_error < 1.0, (
                            f"FD error too large at h={h}: rel_error={rel_error:.2e}"
                        )

    def test_gradient_vs_fd_multiple_rho(self):
        grad_fn = jax.grad(pipeline)
        for seed in [42, 99, 137]:
            rng = np.random.default_rng(seed)
            rho = jnp.asarray(rng.random(self.N_NODES), dtype=jnp.float64)
            direction = jnp.asarray(
                rng.standard_normal(self.N_NODES), dtype=jnp.float64
            )
            direction = direction / jnp.linalg.norm(direction)

            grad_exact = grad_fn(rho)
            exact_dir = float(jnp.dot(grad_exact, direction))

            h = 1e-5
            f_plus = pipeline(rho + h * direction)
            f_minus = pipeline(rho - h * direction)
            fd_dir = float((f_plus - f_minus) / (2.0 * h))

            assert abs(fd_dir - exact_dir) < 1e-8, (
                f"seed={seed}: exact={exact_dir:.4e}, fd={fd_dir:.4e}"
            )


class TestPipelineShapeValidation:
    """Intermediate shapes through the pipeline are consistent."""

    N_NODES = 16

    @pytest.fixture
    def rho(self) -> jax.Array:
        return jnp.asarray(RNG.random(self.N_NODES), dtype=jnp.float64)

    def test_doping_range(self, rho):

        doping = jnp.power(
            jnp.asarray(10.0, dtype=rho.dtype),
            jnp.asarray(14.0, dtype=rho.dtype)
            + jnp.asarray(7.0, dtype=rho.dtype) * rho,
        )
        assert doping.shape == rho.shape
        assert jnp.all(doping >= 1e14)
        assert jnp.all(doping <= 1e21)

    def test_sb_output_shapes(self, rho):
        from prismo.pipeline import _sb_jax

        n = jnp.ones_like(rho) * 1e24
        p = jnp.ones_like(rho) * 1e24
        n_eq = jnp.ones_like(rho) * 1e21
        p_eq = jnp.ones_like(rho) * 1e21

        depsilon, dalpha = _sb_jax(n, p, n_eq, p_eq)
        assert depsilon.shape == rho.shape
        assert dalpha.shape == rho.shape

    def test_gyptis_scalar_output(self, rho):
        result = build_default_components().gyptis(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_ct_call_output_shapes(self, rho):
        n_out, p_out = build_default_components().chargetransport(rho, 0.0)
        assert n_out.shape == rho.shape
        assert p_out.shape == rho.shape


class TestChargeTransportMeshDelivery:
    """The shared mesh must reach the ChargeTransport container.

    The CT container has no access to host paths, so a host ``mesh_ref.path``
    is useless inside it -- the worker silently falls back to its 1D device and
    the gmsh-order lateral junction becomes a many-junction line the -5 V solve
    cannot converge on. The fix is a read-only bind mount plus a rewritten path
    (ticket 02). Ref: .scratch/chargetransport-mesh-node-ordering.
    """

    def test_container_mesh_ref_rewrites_path_to_mount(self):
        ref = MeshRef(path="/host/outputs/waveguide.msh", n_nodes=62)
        rewritten = _container_mesh_ref(ref)
        assert rewritten.path == f"{_CT_MESH_MOUNT}/waveguide.msh"
        # Only the path changes; the rest of the reference is preserved.
        assert rewritten.n_nodes == 62
        assert rewritten.node_ordering == ref.node_ordering
        # The host reference is left untouched (no in-place mutation).
        assert ref.path == "/host/outputs/waveguide.msh"

    def test_container_mesh_ref_none_passthrough(self):
        assert _container_mesh_ref(None) is None

    def test_container_forward_sends_mount_path_not_host_path(self):
        """The apply() request must carry the in-container mesh path."""

        class RecordingTesseract:
            def __init__(self):
                self.inputs = None

            def apply(self, inputs):
                self.inputs = inputs
                doping = np.asarray(inputs["doping"], dtype=float)
                return {"electrons": doping, "holes": doping}

        recorder = RecordingTesseract()
        component = build_chargetransport_component(container=recorder)
        host_ref = MeshRef(path="/host/outputs/waveguide.msh", n_nodes=3)

        component.forward(np.array([1.0, 2.0, 3.0]), 0.0, host_ref)

        assert recorder.inputs["mesh_ref"]["path"] == f"{_CT_MESH_MOUNT}/waveguide.msh"

    def test_init_bind_mounts_mesh_dir_into_ct_container(self, monkeypatch, tmp_path):
        """init_tesseract_containers mounts the mesh dir read-only into CT."""
        import sys
        import types

        captured: dict[str, object] = {}

        class FakeTesseract:
            def __init__(self, image, volumes=None):
                self.image = image
                self.volumes = volumes

            @classmethod
            def from_image(cls, image, volumes=None):
                if "chargetransport" in image:
                    captured["ct_volumes"] = volumes
                return cls(image, volumes)

            def serve(self):
                return None

            def teardown(self):
                return None

        monkeypatch.setitem(
            sys.modules,
            "tesseract_core",
            types.SimpleNamespace(Tesseract=FakeTesseract),
        )
        # Keep the component builders from touching the fake containers further.
        monkeypatch.setattr(
            "prismo.pipeline.build_chargetransport_component",
            lambda container=None, local_api=None: (lambda *a, **k: None),
        )
        monkeypatch.setattr(
            "prismo.pipeline.build_gyptis_components",
            lambda container=None, local_api=None: (
                (lambda *a, **k: None),
                (lambda *a, **k: None),
            ),
        )

        mesh_dir = tmp_path / "outputs"
        init_tesseract_containers(mesh_dir=mesh_dir)

        assert captured["ct_volumes"] == [
            f"{mesh_dir.resolve()}:{_CT_MESH_MOUNT}:ro"
        ]
        # The mount directory is created so the bind mount always resolves.
        assert mesh_dir.exists()
