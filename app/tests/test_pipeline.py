"""Tests for the end-to-end differentiable pipeline.

Ref: ticket 14 -- end-to-end pipeline via JAX composition.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from _doubles import stub_components  # noqa: E402
from prismo.density_filter import assemble_filter_matrix  # noqa: E402
from prismo.mesh_transfer import build_mesh_transfer_operator  # noqa: E402
from prismo.pipeline import (  # noqa: E402
    _CT_MESH_MOUNT,
    _JUNCTION_SEED_THETA,
    DOPING_LOG10_SPAN,
    DOPING_REFERENCE_CM3,
    REVERSE_BIAS_V,
    DesignNodes,
    PipelineComponents,
    _build_design_epsilon,
    _container_mesh_ref,
    _sb_jax,
    build_chargetransport_component,
    build_default_components,
    build_design_transfer,
    build_gyptis_components,
    doping_from_theta,
    init_tesseract_containers,
    pipeline,
    read_gyptis_design_cell_centroids,
    read_gyptis_mode_field,
    vpi_lpi_v_cm,
    write_gyptis_mesh,
)
from prismo.soref_bennett import soref_bennett as _sb_numpy  # noqa: E402
from prismo_shared.schemas import CarrierDensityField, MeshRef  # noqa: E402

RNG = np.random.default_rng(0)


def _components_with(**overrides) -> PipelineComponents:
    """Physics-free component doubles with specific components replaced (ticket 04)."""
    return stub_components(**overrides)


_DOPING_CEILING = DOPING_REFERENCE_CM3 * (10.0**DOPING_LOG10_SPAN - 1.0)


class TestVpiLpi:
    """VπLπ [V·cm] modulation-efficiency headline from signed Δneff (ticket 03)."""

    _WAVELENGTH_CM = 1.55e-4  # 1.55 µm; must match the gyptis solver point.

    def test_matches_closed_form(self):
        """VπLπ = |V_bias|·λ / (2·Δneff)."""
        delta_neff = 1e-4
        expected = abs(REVERSE_BIAS_V) * self._WAVELENGTH_CM / (2.0 * delta_neff)
        assert vpi_lpi_v_cm(delta_neff) == pytest.approx(expected)

    def test_carries_the_sign_of_delta_neff(self):
        """A wrong-polarity (negative) Δneff reports a negative VπLπ."""
        assert vpi_lpi_v_cm(-1e-4) == pytest.approx(-vpi_lpi_v_cm(1e-4))

    def test_smaller_is_better(self):
        """A larger |Δneff| (more efficient) is a smaller |VπLπ|."""
        assert abs(vpi_lpi_v_cm(2e-4)) < abs(vpi_lpi_v_cm(1e-4))

    def test_diverges_at_zero_shift(self):
        """Δneff = 0 (no modulation) is infinite VπLπ, not a crash."""
        assert vpi_lpi_v_cm(0.0) == float("inf")


class TestDopingFromTheta:
    """Signed, zero-referenced net-doping map ``N(theta)`` (ticket 02)."""

    def test_zero_referenced(self):
        """theta=0 is the junction: net-intrinsic (zero net doping)."""
        assert float(doping_from_theta(jnp.asarray(0.0))) == 0.0

    def test_antisymmetric_no_counterdoping(self):
        """N(-theta) = -N(theta), and sign(N) follows sign(theta) everywhere."""
        theta = jnp.linspace(-1.0, 1.0, 21)
        doping = doping_from_theta(theta)
        np.testing.assert_allclose(
            np.asarray(doping), -np.asarray(doping_from_theta(-theta)), rtol=1e-6
        )
        np.testing.assert_array_equal(
            np.sign(np.asarray(doping)), np.sign(np.asarray(theta))
        )

    def test_ceiling_within_physical_window(self):
        """|theta|=1 saturates at ~1e19 cm^-3 (below solid solubility)."""
        ceiling = float(doping_from_theta(jnp.asarray(1.0)))
        assert 9e18 <= ceiling <= 1e20
        assert float(doping_from_theta(jnp.asarray(-1.0))) == -ceiling

    def test_seed_is_a_depletion_modulator_operating_point(self):
        """The seeded junction dopes into the partially-depleted regime.

        At |N| ~ 3e15 the -5 V depletion width (~1.6 um) far exceeds the 500 nm
        rib, so the rib depletes fully and the reverse-bias carrier field carries
        no bulk-doping signal for a gradient to validate against (ticket 06).
        The seed must land in the 1e17-1e18 band a real carrier-depletion
        modulator works in.
        """
        seeded = abs(float(doping_from_theta(jnp.asarray(_JUNCTION_SEED_THETA))))
        assert 1e17 <= seeded <= 1e18

    def test_monotonic_in_theta(self):
        """More positive theta always dopes more n-type; no folding."""
        doping = np.asarray(doping_from_theta(jnp.linspace(-1.0, 1.0, 101)))
        assert np.all(np.diff(doping) > 0)

    def test_grad_finite_across_zero(self):
        """jax.grad stays finite for a field straddling the theta=0 junction."""
        theta = jnp.asarray([-1.0, -0.4, -1e-9, 0.0, 1e-9, 0.4, 1.0])
        grad = jax.grad(lambda t: jnp.sum(doping_from_theta(t)))(theta)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_correct_at_junction(self):
        """At theta=0 (the junction) grad equals the C^1 slope, not zero.

        The map is differentiable through zero with slope 1e17*ln10*span; a naive
        sign*abs autodiff collapses it to 0, so the custom JVP is what makes the
        crossing the optimizer moves honest.
        """
        expected = DOPING_REFERENCE_CM3 * np.log(10.0) * DOPING_LOG10_SPAN
        got = float(jax.grad(doping_from_theta)(jnp.asarray(0.0)))
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_grad_matches_analytic_away_from_kink(self):
        """dN/dtheta = 1e17 * ln10 * span * 10^(span*|theta|) off the crossing."""
        for t in (-1.0, -0.4, 0.4, 1.0):
            expected = (
                DOPING_REFERENCE_CM3
                * np.log(10.0)
                * DOPING_LOG10_SPAN
                * 10.0 ** (DOPING_LOG10_SPAN * abs(t))
            )
            got = float(jax.grad(doping_from_theta)(jnp.asarray(t)))
            np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_moving_zero_crossing_moves_junction(self):
        """The net-doping sign flip tracks the theta zero-crossing (one junction)."""
        x = jnp.linspace(0.0, 1.0, 101)
        step = float(x[1] - x[0])
        for crossing in (0.305, 0.695):
            doping = np.asarray(doping_from_theta(x - crossing))
            flips = np.where(np.diff(np.sign(doping)) != 0)[0]
            assert flips.size == 1
            assert abs(float(x[flips[0]]) - crossing) <= step + 1e-9


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
    """Pipeline composed from explicit JAX-native doubles (ticket 04).

    The implicit no-backend stubs were deleted; these tests inject physics-free
    doubles through ``components=`` instead of relying on a fabricated default.
    """

    N_NODES = 16

    @pytest.fixture
    def rho(self) -> jax.Array:
        return jnp.asarray(RNG.random(self.N_NODES), dtype=jnp.float64)

    def test_no_backend_is_a_hard_error(self, rho):
        """Ticket 04: the default (backend-less) pipeline never fabricates -- it raises.

        ``make run`` without a container gets a clear error rather than an
        effective-medium neff or identity carriers.
        """
        with pytest.raises(Exception, match="backend"):
            pipeline(rho)

    def test_default_components_have_no_physics_free_fallback(self, rho):
        """``build_default_components`` builds, but a call without a live solver raises."""
        components = build_default_components()
        with pytest.raises(Exception, match="backend"):
            pipeline(rho, components=components)

    def test_forward_returns_scalar(self, rho):
        result = pipeline(rho, components=_components_with())
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
        bg = pl.DEFAULT_BACKGROUND_EPSILON
        # No mean-collapse: the uniform depletion perturbs the whole design
        # field, and the effective-medium stub averages it back to bg + exp_deps.
        exp_result = float(np.sqrt(bg + exp_deps) - np.sqrt(bg))
        np.testing.assert_allclose(float(result), exp_result, rtol=1e-6)
        assert float(result) > 0.0

    def test_signed_theta_forms_pn_junction_doping(self):
        """Sign(theta) is the free polarity: theta<0 p-type, theta>0 n-type.

        No fixed polarity mask is plumbed anymore -- the mixed-sign net doping
        fed to ChargeTransport comes straight from the signed design field.
        """
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
            jnp.asarray([-1.0, 1.0]),
            components=_components_with(chargetransport=fake_ct),
        )

        np.testing.assert_allclose(received[0], [-_DOPING_CEILING, _DOPING_CEILING])
        np.testing.assert_allclose(received[1], [-_DOPING_CEILING, _DOPING_CEILING])
        assert float(result) > 0.0

    def test_objective_is_signed_by_physical_bias_response(self):
        """Signed Δneff: depletion reads positive, injection negative.

        No ``hypot`` magnitude fold anymore (ticket 03). The reverse-bias sign
        is physically determined -- depletion raises the index (+Δneff) and
        carrier injection lowers it (-Δneff) -- so the optimizer maximizes the
        signed value directly rather than rewarding either mode-shift direction.
        """

        def depleting_ct(doping, bias_voltage, mesh_ref=None):
            # Carriers present at 0 V, swept out under reverse bias.
            carriers = jnp.where(
                bias_voltage == 0.0,
                jnp.full_like(doping, 1e24),
                jnp.zeros_like(doping),
            )
            return carriers, carriers

        def injecting_ct(doping, bias_voltage, mesh_ref=None):
            # Opposite (non-physical) response: carriers appear under bias.
            carriers = jnp.where(
                bias_voltage == 0.0,
                jnp.zeros_like(doping),
                jnp.full_like(doping, 1e24),
            )
            return carriers, carriers

        depleting = pipeline(
            jnp.asarray([0.25, 0.25]),
            components=_components_with(chargetransport=depleting_ct),
        )
        injecting = pipeline(
            jnp.asarray([0.25, 0.25]),
            components=_components_with(chargetransport=injecting_ct),
        )

        # Opposite physical responses land on opposite sides of zero -- the old
        # hypot fold would have made both positive.
        assert float(depleting) > 0.0
        assert float(injecting) < 0.0

    def test_effective_index_objective_has_zero_gradient_at_zero_shift(self):
        """The signed objective stays finite and differentiable at zero shift."""

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

        # gyptis double sensitive to spatial structure: neff_sq = sum(eps^2),
        # a pure-JAX callable JAX differentiates natively (per-cell grad 2*eps,
        # non-constant for a structured field).
        structure_sensitive = lambda eps, *s: jnp.sum(eps**2)
        components = _components_with(
            chargetransport=fake_ct, gyptis=structure_sensitive
        )

        rho = jnp.asarray([0.2, 0.5, 0.8, 0.4])
        grad = jax.grad(lambda r: pipeline(r, components=components))(rho)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.std(grad)) > 0.0  # spatially resolved, not collapsed

    def test_identity_doubles_give_zero_shift(self, rho):
        """Identity carrier doubles produce an identical field at both biases."""
        result = pipeline(rho, components=_components_with())
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_zero_shift_is_exactly_zero_not_floored(self):
        """No ``hypot(_, 1e-15)`` fudge: a null shift is exactly 0.0 (ticket 03).

        Equal carriers at both biases give an identical permittivity field, so
        the signed objective is exactly zero -- not lifted to a ~1e-15 floor.
        """

        def flat_ct(doping, bias_voltage, mesh_ref=None):
            return doping, doping

        result = pipeline(
            jnp.asarray([0.25, 0.5]),
            components=_components_with(chargetransport=flat_ct),
        )
        assert float(result) == 0.0

    def test_gradient_is_finite(self, rho):
        grad = jax.grad(lambda r: pipeline(r, components=_components_with()))(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_identity_doubles_give_zero_gradient(self, rho):
        grad = jax.grad(lambda r: pipeline(r, components=_components_with()))(rho)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_jit_works(self, rho):
        components = _components_with()
        jitted = jax.jit(lambda r: pipeline(r, components=components))
        result = jitted(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_jit_gradient_works(self, rho):
        components = _components_with()
        grad_fn = jax.jit(jax.grad(lambda r: pipeline(r, components=components)))
        grad = grad_fn(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_dtype_preserved(self, rho):
        components = _components_with()
        result32 = pipeline(jnp.asarray(rho, dtype=jnp.float32), components=components)
        assert result32.dtype == jnp.float32

        result64 = pipeline(
            jnp.asarray(rho, dtype=jnp.float64), components=components
        )
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

    def test_reads_mode_field_from_gyptis_container(self):
        class FakeGyptis:
            def __init__(self):
                self.inputs = None

            def apply(self, inputs):
                self.inputs = inputs
                return {
                    "mode_abs_e": [0.5, 1.0, 0.25],
                    "mode_coordinates": [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
                }

        gyptis = FakeGyptis()
        abs_e, coords = read_gyptis_mode_field(
            np.array([12.0, 12.1]), 11.5, container=gyptis
        )

        assert gyptis.inputs["operation"] == "mode_field"
        np.testing.assert_allclose(gyptis.inputs["design_epsilon"], [12.0, 12.1])
        # The background must match the solve components': it keys the tracked
        # mode branch as well as the device being solved (ticket 13).
        assert gyptis.inputs["core_epsilon"] == 11.5
        np.testing.assert_allclose(abs_e, [0.5, 1.0, 0.25])
        np.testing.assert_allclose(coords, [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])

    def test_mode_field_rejects_mismatched_payload(self):
        class FakeGyptis:
            def apply(self, inputs):
                return {
                    "mode_abs_e": [0.5, 1.0, 0.25],
                    "mode_coordinates": [[0.0, 0.0], [0.1, 0.0]],
                }

        with pytest.raises(ValueError, match="one magnitude per mesh vertex"):
            read_gyptis_mode_field(np.array([12.0]), container=FakeGyptis())

    def test_mode_field_rejects_an_empty_payload(self):
        class FakeGyptis:
            def apply(self, inputs):
                return {}

        with pytest.raises(RuntimeError, match="no field payload"):
            read_gyptis_mode_field(np.array([12.0]), container=FakeGyptis())

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
        # Shared mesh: a unit square split into two silicon triangles. Each design
        # cell is one of those triangles, its vertices are shared-mesh nodes.
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
        silicon_triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp)
        design_vertices = coords[silicon_triangles]  # (2, 3, 2)
        design_transfer = jnp.asarray(
            build_mesh_transfer_operator(coords, design_vertices).dense()
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
    """The container setup assembles the real mesh-transfer matrix (ticket 05)."""

    def test_assembles_real_operator_from_design_cell_vertices(self):
        """The production helper drives the real ``build_mesh_transfer_operator``.

        The design-cell vertices come from the gyptis backend through the
        component seam; on the shared mesh each cell is a triangle whose vertices
        are shared-mesh nodes, so this covers the exact-restriction assembly the
        container run performs.
        """
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp)
        components = PipelineComponents(
            chargetransport=None,
            gyptis=None,
            gyptis_background=None,
            design_cell_vertices=lambda: coords[triangles],
        )

        transfer = build_design_transfer(components, coords)

        assert transfer.shape == (2, 4)  # design cells <- shared-mesh nodes
        # Partition of unity: each design cell's weights sum to one, so a uniform
        # nodal field maps to a uniform design field (keyed to the shared mesh).
        np.testing.assert_allclose(np.asarray(transfer).sum(axis=1), 1.0)

    def test_requires_a_vertex_reader(self):
        """Without a gyptis backend there is no design geometry to key against."""
        components = PipelineComponents(
            chargetransport=None, gyptis=None, gyptis_background=None
        )
        with pytest.raises(RuntimeError, match="design-cell"):
            build_design_transfer(components, np.zeros((3, 2)))


class TestPipelineComponentCalls:
    """The pipeline drives each component exactly once per solve it needs."""

    def _fake_ct(self, biases: list[float] | None = None):
        recorded = biases if biases is not None else []

        class FakeChargeTransport:
            def apply(self, inputs):
                recorded.append(inputs["bias_voltage"])
                doping = np.asarray(inputs["doping"], dtype=float)
                carrier = np.full_like(
                    doping, 1e18 if inputs["bias_voltage"] == 0.0 else 0.0
                )
                return {"electrons": carrier, "holes": carrier}

        return build_chargetransport_component(container=FakeChargeTransport())

    def test_solves_charge_transport_at_both_bias_points(self):
        biases: list[float] = []
        ct = self._fake_ct(biases)
        ct(jnp.full((4,), 0.25), 0.0)
        ct(jnp.full((4,), 0.25), -5.0)

        assert biases == [0.0, -5.0]

    def test_one_pipeline_call_drives_both_backends_once_each(self):
        biases: list[float] = []
        gyptis_calls = []

        class FakeGyptis:
            def apply(self, inputs):
                gyptis_calls.append(np.asarray(inputs["design_epsilon"], dtype=float))
                return {"neff_sq": float(np.mean(inputs["design_epsilon"]))}

        perturbed, background = build_gyptis_components(container=FakeGyptis())
        components = PipelineComponents(
            chargetransport=self._fake_ct(biases),
            gyptis=perturbed,
            gyptis_background=background,
        )
        pipeline(jnp.full((4,), 0.25), components=components)

        assert sorted(biases) == [-5.0, 0.0]
        # One background solve and one perturbed solve.
        assert len(gyptis_calls) == 2


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

    @pytest.fixture
    def components(self) -> PipelineComponents:
        return _components_with()

    def test_pipeline_with_filter_runs(self, rho, H_dense, H_sum, components):
        result = pipeline(rho, H=H_dense, H_sum=H_sum, components=components)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_gradient_with_filter(self, rho, H_dense, H_sum, components):
        grad = jax.grad(
            lambda r: pipeline(r, H=H_dense, H_sum=H_sum, components=components),
        )(rho)
        assert grad.shape == rho.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_with_filter(self, rho, H_dense, H_sum, components):
        fn = jax.jit(
            lambda r: pipeline(r, H=H_dense, H_sum=H_sum, components=components)
        )
        result = fn(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_box_bounds_preserved(self, rho, H_dense, H_sum, components):
        for val in [0.0, 0.5, 1.0]:
            r = jnp.full_like(rho, val)
            result = pipeline(r, H=H_dense, H_sum=H_sum, components=components)
            assert jnp.isfinite(result)

    def test_shape_consistency(self, rho, H_dense, H_sum, components):
        result = pipeline(rho, H=H_dense, H_sum=H_sum, components=components)
        d = float(result)
        assert isinstance(d, float)


class TestPipelineDopingMapping:
    """The signed doping map N(theta) = sign(theta)*1e17*(10^(2|theta|)-1)."""

    def test_range(self):
        n_nodes = 10
        theta = jnp.linspace(-1.0, 1.0, n_nodes)
        result = pipeline(theta, components=_components_with())
        assert jnp.isfinite(result)
        assert result.ndim == 0

    def test_edge_cases_finite(self):
        components = _components_with()
        for val in [-1.0, -0.5, 0.0, 0.001, 0.5, 0.999, 1.0]:
            theta = jnp.array([val])
            result = pipeline(theta, components=components)
            assert jnp.isfinite(result), f"Non-finite at theta={val}"

    def test_gradient_exists_for_nonuniform_rho(self):
        n_nodes = 10
        rho = jnp.asarray(RNG.random(n_nodes), dtype=jnp.float64)
        grad = jax.grad(lambda r: pipeline(r, components=_components_with()))(rho)
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
        components = _components_with()
        run = lambda r: pipeline(r, components=components)
        grad_fn = jax.grad(run)

        for rho in rho_vectors:
            grad_exact = grad_fn(rho)
            for direction in directions:
                for h in step_sizes:
                    f_plus = run(rho + h * direction)
                    f_minus = run(rho - h * direction)
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
        components = _components_with()
        run = lambda r: pipeline(r, components=components)
        grad_fn = jax.grad(run)
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
            f_plus = run(rho + h * direction)
            f_minus = run(rho - h * direction)
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
        theta = jnp.linspace(-1.0, 1.0, rho.shape[0], dtype=rho.dtype)
        doping = doping_from_theta(theta)
        assert doping.shape == theta.shape
        assert jnp.all(jnp.abs(doping) <= _DOPING_CEILING + 1.0)
        # zero-referenced and antisymmetric
        assert float(doping_from_theta(jnp.asarray(0.0, dtype=rho.dtype))) == 0.0

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
        result = _components_with().gyptis(rho)
        assert result.ndim == 0
        assert jnp.isfinite(result)

    def test_ct_call_output_shapes(self, rho):
        n_out, p_out = _components_with().chargetransport(rho, 0.0)
        assert n_out.shape == rho.shape
        assert p_out.shape == rho.shape


def _fake_container_env(monkeypatch) -> dict[str, object]:
    """Stand in for tesseract_core so ``init_tesseract_containers`` runs offline.

    Returns the dict the fake records into: the CT volume list plus the
    environment each image was started with, which is how the mesh-size and
    Julia-deadline knobs reach the containers.
    """
    import sys
    import types

    captured: dict[str, object] = {}

    class FakeTesseract:
        def __init__(self, image, volumes=None, environment=None):
            self.image = image
            self.volumes = volumes
            self.environment = environment

        @classmethod
        def from_image(cls, image, volumes=None, environment=None):
            if "chargetransport" in image:
                captured["ct_volumes"] = volumes
                captured["ct_environment"] = environment
            else:
                captured["gyptis_volumes"] = volumes
                captured["gyptis_environment"] = environment
            return cls(image, volumes, environment)

        def serve(self):
            return None

        def teardown(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "tesseract_core",
        types.SimpleNamespace(Tesseract=FakeTesseract),
    )
    # The dev-mount switch is a host-environment knob; tests opt in explicitly.
    monkeypatch.delenv("PRISMO_DEV_MOUNTS", raising=False)
    monkeypatch.delenv("PRISMO_CT_SOLVE_BUDGET_S", raising=False)
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
    return captured


class TestChargeTransportMeshDelivery:
    """The shared mesh must reach the ChargeTransport container.

    The CT container has no access to host paths, so a host ``mesh_ref.path``
    is useless inside it -- the worker silently falls back to its 1D device and
    the gmsh-order lateral junction becomes a many-junction line the -5 V solve
    cannot converge on. The fix is a read-only bind mount plus a rewritten path
    (ticket 02). Ref: .scratch/chargetransport-mesh-node-ordering.
    """

    def test_write_gyptis_mesh_persists_host_owned_mesh(self, tmp_path):
        """The gyptis mesh payload becomes CT's host-mounted shared mesh."""

        class MeshAuthor:
            def apply(self, inputs):
                assert inputs == {"operation": "write_mesh"}
                return {
                    "mesh_text": "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n",
                    "design_cell_vertices": [[[0.0, 0.0]] * 3],
                }

        mesh_path = tmp_path / "waveguide.msh"
        vertices = write_gyptis_mesh(mesh_path, container=MeshAuthor())

        assert mesh_path.read_text() == "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        assert vertices.shape == (1, 3, 2)

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
        captured = _fake_container_env(monkeypatch)

        mesh_dir = tmp_path / "outputs"
        init_tesseract_containers(mesh_dir=mesh_dir)

        assert captured["ct_volumes"] == [
            f"{mesh_dir.resolve()}:{_CT_MESH_MOUNT}:ro"
        ]
        # The mount directory is created so the bind mount always resolves.
        assert mesh_dir.exists()

    def test_init_passes_mesh_size_to_the_gyptis_mesh_author(
        self, monkeypatch, tmp_path
    ):
        """The gyptis container authors the shared mesh, so the knob goes to it."""
        captured = _fake_container_env(monkeypatch)

        init_tesseract_containers(mesh_dir=tmp_path / "outputs", mesh_size=0.015)

        assert captured["gyptis_environment"] == {"PRISMO_GYPTIS_MESH_SIZE": "0.015"}

    def test_init_defaults_leave_the_container_environments_alone(
        self, monkeypatch, tmp_path
    ):
        """No mesh size and no host override: neither container gets an env."""
        monkeypatch.delenv("PRISMO_CT_JULIA_TIMEOUT_S", raising=False)
        captured = _fake_container_env(monkeypatch)

        init_tesseract_containers(mesh_dir=tmp_path / "outputs")

        assert captured["gyptis_environment"] is None
        assert captured["ct_environment"] is None

    def test_init_forwards_the_ct_julia_timeout_override(self, monkeypatch, tmp_path):
        """A refined mesh needs a longer Newton deadline than the baked 25 s."""
        monkeypatch.setenv("PRISMO_CT_JULIA_TIMEOUT_S", "600")
        captured = _fake_container_env(monkeypatch)

        init_tesseract_containers(mesh_dir=tmp_path / "outputs")

        assert captured["ct_environment"] == {"PRISMO_CT_JULIA_TIMEOUT_S": "600"}

    def test_init_forwards_the_ct_solve_budget_override(self, monkeypatch, tmp_path):
        """The Julia-side solve budget (ticket 18) follows the host override too."""
        captured = _fake_container_env(monkeypatch)
        monkeypatch.setenv("PRISMO_CT_SOLVE_BUDGET_S", "300")

        init_tesseract_containers(mesh_dir=tmp_path / "outputs")

        assert captured["ct_environment"] == {"PRISMO_CT_SOLVE_BUDGET_S": "300"}

    def test_dev_mounts_shadow_both_component_apis(
        self, monkeypatch, tmp_path, capsys
    ):
        """``PRISMO_DEV_MOUNTS=1`` mounts the host api + prismo_shared (ticket 21)."""
        import prismo.pipeline as pl

        captured = _fake_container_env(monkeypatch)
        monkeypatch.setenv("PRISMO_DEV_MOUNTS", "1")

        init_tesseract_containers(mesh_dir=tmp_path / "outputs")

        ct_api = (pl._COMPONENTS_DIR / "chargetransport" / "tesseract_api.py").resolve()
        gy_api = (pl._COMPONENTS_DIR / "gyptis" / "tesseract_api.py").resolve()
        shared = (pl._SHARED_CODE_DIR / "prismo_shared").resolve()
        assert f"{ct_api}:{pl._DEV_API_MOUNT}:ro" in captured["ct_volumes"]
        assert f"{shared}:{pl._DEV_PYTHONPATH_ROOT}/prismo_shared:ro" in (
            captured["ct_volumes"]
        )
        assert captured["gyptis_volumes"] == [
            f"{gy_api}:{pl._DEV_API_MOUNT}:ro",
            f"{shared}:{pl._DEV_PYTHONPATH_ROOT}/prismo_shared:ro",
        ]
        # The dev PYTHONPATH joins the other knobs instead of replacing them.
        assert captured["ct_environment"]["PYTHONPATH"] == pl._DEV_PYTHONPATH_ROOT
        assert captured["gyptis_environment"]["PYTHONPATH"] == pl._DEV_PYTHONPATH_ROOT
        assert "PRISMO_DEV_MOUNTS=1" in capsys.readouterr().out

    def test_dev_mounts_keep_the_mesh_size_knob(self, monkeypatch, tmp_path):
        captured = _fake_container_env(monkeypatch)
        monkeypatch.setenv("PRISMO_DEV_MOUNTS", "1")

        init_tesseract_containers(mesh_dir=tmp_path / "outputs", mesh_size=0.015)

        assert captured["gyptis_environment"] == {
            "PYTHONPATH": "/prismo_dev",
            "PRISMO_GYPTIS_MESH_SIZE": "0.015",
        }

    def test_dev_mounts_off_by_default(self, monkeypatch):
        from prismo.pipeline import dev_mounts_requested

        monkeypatch.delenv("PRISMO_DEV_MOUNTS", raising=False)
        assert not dev_mounts_requested()
        monkeypatch.setenv("PRISMO_DEV_MOUNTS", "0")
        assert not dev_mounts_requested()
        monkeypatch.setenv("PRISMO_DEV_MOUNTS", "1")
        assert dev_mounts_requested()

    def test_reset_reaches_the_ct_container_as_an_operation(self):
        """The reset seam sends the ``reset`` operation to the CT backend."""
        from prismo.pipeline import reset_chargetransport_worker

        class RecordingTesseract:
            def __init__(self):
                self.inputs = None

            def apply(self, inputs):
                self.inputs = inputs
                return {"electrons": [], "holes": []}

        recorder = RecordingTesseract()
        reset_chargetransport_worker(container=recorder)
        assert recorder.inputs == {"operation": "reset"}


class TestDesignNodes:
    """The design set spans the silicon nodes, not every node of the mesh."""

    def test_scatter_places_variables_and_zeroes_the_rest(self):
        """Non-design nodes read theta = 0, i.e. net-intrinsic."""
        nodes = DesignNodes(indices=np.asarray([1, 3]), n_mesh_nodes=5)
        full = nodes.scatter(jnp.asarray([0.4, -0.7]))
        np.testing.assert_allclose(full, [0.0, 0.4, 0.0, -0.7, 0.0])

    def test_scatter_numpy_keeps_the_design_entries_in_place(self):
        """The plotting path places the same values at the same node indices."""
        nodes = DesignNodes(indices=np.asarray([0, 2]), n_mesh_nodes=4)
        design = np.asarray([0.2, -0.5])
        full = nodes.scatter_numpy(design)
        np.testing.assert_allclose(full[nodes.indices], design)
        np.testing.assert_allclose(full, [0.2, 0.0, -0.5, 0.0])

    def test_all_nodes_is_the_identity_design_set(self):
        """The fallback keeps one variable per mesh node."""
        nodes = DesignNodes.all_nodes(3)
        assert len(nodes) == 3
        np.testing.assert_allclose(nodes.scatter(jnp.asarray([1.0, 2.0, 3.0])), [1, 2, 3])

    def test_pipeline_matches_the_scattered_full_field(self):
        """Restricting the design set changes the variables, not the physics.

        A design vector on a subset must produce exactly the objective the
        equivalent full-length field produces, so the smaller MMA problem
        optimizes the same function.
        """
        def fake_ct(doping, bias_voltage, mesh_ref=None):
            carriers = jnp.where(bias_voltage == 0.0, doping, jnp.zeros_like(doping))
            return carriers, carriers

        components = _components_with(
            chargetransport=fake_ct, gyptis=lambda eps, *s: jnp.sum(eps**2)
        )
        nodes = DesignNodes(indices=np.asarray([0, 2]), n_mesh_nodes=4)
        design = jnp.asarray([0.3, -0.4])

        restricted = pipeline(design, design_nodes=nodes, components=components)
        full = pipeline(nodes.scatter(design), components=components)
        assert float(restricted) != 0.0
        assert float(restricted) == pytest.approx(float(full))

    def test_gradient_has_one_entry_per_design_node(self):
        """The adjoint comes back on the design set, not the full mesh."""

        def fake_ct(doping, bias_voltage, mesh_ref=None):
            # Depletion under bias, so a structured design field moves neff.
            carriers = jnp.where(bias_voltage == 0.0, doping, jnp.zeros_like(doping))
            return carriers, carriers

        structure_sensitive = lambda eps, *s: jnp.sum(eps**2)
        components = _components_with(
            chargetransport=fake_ct, gyptis=structure_sensitive
        )
        nodes = DesignNodes(indices=np.asarray([0, 2]), n_mesh_nodes=4)
        grad = jax.grad(
            lambda t: pipeline(t, design_nodes=nodes, components=components)
        )(jnp.asarray([0.3, -0.4]))
        assert grad.shape == (2,)
        assert float(jnp.linalg.norm(grad)) > 0.0
