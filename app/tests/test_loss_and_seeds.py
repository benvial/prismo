"""Tests for the loss-aware objective and the junction seeds.

The modal free-carrier loss is a first-order perturbation: the mode-overlap
weights are the gyptis eigen-adjoint at the uniform background, frozen, and
the per-cell Soref-Bennett absorption at 0 V is summed against them.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from _doubles import stub_components  # noqa: E402
from prismo.pipeline import (  # noqa: E402
    _JUNCTION_SEED_THETA,
    DEFAULT_BACKGROUND_EPSILON,
    NEPER_TO_DB,
    SEED_KINDS,
    PipelineTerms,
    doping_from_theta,
    free_carrier_absorption_cm,
    loss_figure_of_merit_v_db,
    modal_loss_db_cm,
    pipeline,
    pipeline_with_terms,
    read_mode_overlap,
    seed_design_field,
    seed_signed_junction,
    vpi_lpi_v_cm,
)

# -- Seeds -----------------------------------------------------------------------


def _rib_and_slab_coords() -> np.ndarray:
    """A small rib-on-slab node set in micrometres: slab |x| <= 1, rib |x| <= 0.25."""
    xs_slab = np.linspace(-1.0, 1.0, 21)
    ys_slab = np.asarray([0.0, 0.05, 0.10])
    slab = np.array([[x, y] for y in ys_slab for x in xs_slab])
    xs_rib = np.linspace(-0.25, 0.25, 11)
    ys_rib = np.linspace(0.12, 0.32, 11)
    rib = np.array([[x, y] for y in ys_rib for x in xs_rib])
    return np.vstack([slab, rib])


class TestSeeds:
    def test_seed_kinds_are_the_documented_three(self):
        assert SEED_KINDS == ("lateral", "vertical", "u")

    def test_lateral_is_the_legacy_seed(self):
        coords = _rib_and_slab_coords()
        np.testing.assert_array_equal(
            np.asarray(seed_design_field(coords, "lateral")),
            np.asarray(seed_signed_junction(coords)),
        )

    @pytest.mark.parametrize("kind", SEED_KINDS)
    def test_every_seed_is_a_signed_partially_depleted_field(self, kind):
        theta = np.asarray(seed_design_field(_rib_and_slab_coords(), kind))
        assert set(np.unique(theta)) == {-_JUNCTION_SEED_THETA, _JUNCTION_SEED_THETA}

    @pytest.mark.parametrize("kind", SEED_KINDS)
    def test_every_seed_connects_n_to_the_left_contact_and_p_to_the_right(self, kind):
        """n-type must reach the left (anode) slab edge and p-type the right."""
        coords = _rib_and_slab_coords()
        theta = np.asarray(seed_design_field(coords, kind))
        left_edge = coords[:, 0] <= -0.95
        right_edge = coords[:, 0] >= 0.95
        assert np.all(theta[left_edge] > 0.0)
        assert np.all(theta[right_edge] < 0.0)

    def test_vertical_seed_stacks_p_over_n_in_the_rib(self):
        coords = _rib_and_slab_coords()
        theta = np.asarray(seed_design_field(coords, "vertical"))
        rib_centre = (np.abs(coords[:, 0]) < 0.05) & (coords[:, 1] > 0.11)
        ys = coords[rib_centre, 1]
        top, bottom = theta[rib_centre][ys > 0.25], theta[rib_centre][ys < 0.18]
        assert np.all(top < 0.0) and np.all(bottom > 0.0)

    def test_u_seed_wraps_n_under_and_beside_a_p_core(self):
        coords = _rib_and_slab_coords()
        theta = np.asarray(seed_design_field(coords, "u"))
        rib = coords[:, 1] > 0.11
        x, y = coords[:, 0], coords[:, 1]
        core = rib & (np.abs(x) < 0.05) & (y > 0.25)
        floor = rib & (np.abs(x) < 0.05) & (y < 0.14)
        left_wall = rib & (x < -0.2)
        assert np.all(theta[core] < 0.0)
        assert np.all(theta[floor] > 0.0)
        assert np.all(theta[left_wall] > 0.0)

    def test_unknown_seed_raises(self):
        with pytest.raises(ValueError, match="seed"):
            seed_design_field(_rib_and_slab_coords(), "diagonal")


# -- Loss ------------------------------------------------------------------------


class TestFreeCarrierAbsorption:
    def test_matches_soref_bennett_at_1e19(self):
        """~85 cm^-1 for 1e19 electrons; ~60 cm^-1 for 1e19 holes."""
        alpha_e = free_carrier_absorption_cm(jnp.asarray([1e19]), jnp.asarray([0.0]))
        alpha_h = free_carrier_absorption_cm(jnp.asarray([0.0]), jnp.asarray([1e19]))
        assert float(alpha_e[0]) == pytest.approx(85.0)
        assert float(alpha_h[0]) == pytest.approx(60.0)

    def test_intrinsic_silicon_is_lossless(self):
        alpha = free_carrier_absorption_cm(jnp.zeros(3), jnp.zeros(3))
        np.testing.assert_array_equal(np.asarray(alpha), 0.0)


class TestModalLoss:
    def test_uniform_core_loss_is_confinement_weighted(self):
        """alpha_mode = (n_si / neff) * sum(w) * alpha for a uniform core."""
        alpha = jnp.full(5, 85.0)
        weights = jnp.full(5, 0.8 / 5)  # sum(w) = 0.8: the core's share of d(neff^2)
        neff = 2.5
        loss = modal_loss_db_cm(alpha, weights, neff)
        expected = (3.4757 / neff) * 0.8 * 85.0 * NEPER_TO_DB
        assert float(loss) == pytest.approx(expected)

    def test_neper_to_db_is_ten_log10_e(self):
        assert NEPER_TO_DB == pytest.approx(10.0 / np.log(10.0))

    def test_figure_of_merit_is_vpi_lpi_times_loss(self):
        fom = loss_figure_of_merit_v_db(4.0e-4, 100.0)
        assert fom == pytest.approx(vpi_lpi_v_cm(4.0e-4) * 100.0)


def _carrier_stub(n_cm3: float, p_cm3: float):
    """ChargeTransport double: fixed carriers at 0 V, depleted at bias."""

    def ct(doping, bias_voltage, mesh_ref=None):
        n = jnp.full_like(doping, n_cm3)
        p = jnp.full_like(doping, p_cm3)
        if bias_voltage != 0.0:
            return 0.5 * n, 0.5 * p
        return n, p

    return ct


class TestPipelineLossTerm:
    N = 6

    def test_terms_carry_delta_neff_and_modal_loss(self):
        theta = jnp.full(self.N, 0.3)
        components = stub_components(chargetransport=_carrier_stub(1e18, 1e16))
        overlap = jnp.full(self.N, 1.0 / self.N)
        objective, terms = pipeline_with_terms(
            theta, components=components, mode_overlap=overlap
        )
        assert isinstance(terms, PipelineTerms)
        assert float(objective) == pytest.approx(float(terms.delta_neff))
        assert float(terms.delta_neff) == pytest.approx(
            float(pipeline(theta, components=components))
        )
        # Loss at 0 V from the absolute carriers, weighted by the overlap, with the
        # mean-field stub's background neff = sqrt(background epsilon).
        alpha = 8.5e-18 * 1e18 + 6.0e-18 * 1e16
        neff0 = np.sqrt(DEFAULT_BACKGROUND_EPSILON)
        expected = (3.4757 / neff0) * alpha * NEPER_TO_DB
        assert float(terms.modal_loss_db_cm) == pytest.approx(expected, rel=1e-9)

    def test_without_overlap_the_loss_is_nan_and_the_objective_unchanged(self):
        theta = jnp.full(self.N, 0.3)
        components = stub_components(chargetransport=_carrier_stub(1e18, 1e16))
        objective, terms = pipeline_with_terms(theta, components=components)
        assert np.isnan(float(terms.modal_loss_db_cm))
        assert float(objective) == pytest.approx(float(terms.delta_neff))
        assert np.isfinite(float(objective))

    def test_loss_weight_without_overlap_raises(self):
        theta = jnp.full(self.N, 0.3)
        components = stub_components(chargetransport=_carrier_stub(1e18, 1e16))
        with pytest.raises(ValueError, match="mode_overlap"):
            pipeline(theta, components=components, loss_weight=1e-6)

    def test_negative_loss_weight_raises(self):
        theta = jnp.full(self.N, 0.3)
        with pytest.raises(ValueError, match="loss_weight"):
            pipeline(
                theta,
                components=stub_components(),
                loss_weight=-1.0,
                mode_overlap=jnp.full(self.N, 1.0 / self.N),
            )

    def test_mismatched_overlap_shape_raises(self):
        theta = jnp.full(self.N, 0.3)
        with pytest.raises(ValueError, match="mode_overlap"):
            pipeline(
                theta,
                components=stub_components(),
                loss_weight=1e-6,
                mode_overlap=jnp.ones(self.N + 1),
            )

    def test_loss_weight_subtracts_weighted_loss(self):
        theta = jnp.full(self.N, 0.3)
        components = stub_components(chargetransport=_carrier_stub(1e18, 1e16))
        overlap = jnp.full(self.N, 1.0 / self.N)
        w = 2.5e-6
        objective, terms = pipeline_with_terms(
            theta, components=components, mode_overlap=overlap, loss_weight=w
        )
        assert float(objective) == pytest.approx(
            float(terms.delta_neff) - w * float(terms.modal_loss_db_cm)
        )
        assert float(objective) == pytest.approx(
            float(
                pipeline(
                    theta, components=components, mode_overlap=overlap, loss_weight=w
                )
            )
        )

    def test_loss_gradient_flows_through_the_equilibrium_carriers(self):
        """Loss gradient through identity carriers matches central FD, non-zero."""
        theta = jnp.asarray(np.linspace(-0.4, 0.4, self.N))
        components = stub_components()  # identity carriers: n = p = doping
        overlap = jnp.asarray(np.linspace(0.1, 0.3, self.N))
        w = 1e-3

        def f(t):
            return pipeline(
                t, components=components, mode_overlap=overlap, loss_weight=w
            )

        grad = np.asarray(jax.grad(f)(theta))
        assert np.all(np.isfinite(grad))
        assert np.linalg.norm(grad) > 0.0
        h = 1e-6
        for i in (0, self.N // 2, self.N - 1):
            e = np.zeros(self.N)
            e[i] = h
            fd = (float(f(theta + e)) - float(f(theta - e))) / (2 * h)
            assert grad[i] == pytest.approx(fd, rel=1e-4, abs=1e-12)

    def test_loss_term_is_jittable(self):
        theta = jnp.full(self.N, 0.3)
        components = stub_components(chargetransport=_carrier_stub(1e18, 1e16))
        overlap = jnp.full(self.N, 1.0 / self.N)
        f = jax.jit(
            lambda t: pipeline_with_terms(
                t, components=components, mode_overlap=overlap, loss_weight=1e-6
            )
        )
        objective, terms = f(theta)
        assert np.isfinite(float(objective))
        assert np.isfinite(float(terms.modal_loss_db_cm))

    def test_loss_counts_slab_nodes_only_through_the_transfer(self):
        """The overlap lives on the design cells; a transfer maps nodes to cells."""
        n_nodes = 5
        transfer = jnp.asarray(
            [[1 / 3, 1 / 3, 1 / 3, 0.0, 0.0], [0.0, 0.0, 1 / 3, 1 / 3, 1 / 3]]
        )
        theta = jnp.full(n_nodes, 0.3)
        components = stub_components(chargetransport=_carrier_stub(1e18, 1e16))
        overlap = jnp.asarray([0.4, 0.2])
        _objective, terms = pipeline_with_terms(
            theta, components=components, mode_overlap=overlap, design_transfer=transfer
        )
        alpha = 8.5e-18 * 1e18 + 6.0e-18 * 1e16
        neff0 = np.sqrt(DEFAULT_BACKGROUND_EPSILON)
        expected = (3.4757 / neff0) * 0.6 * alpha * NEPER_TO_DB
        assert float(terms.modal_loss_db_cm) == pytest.approx(expected, rel=1e-9)


class TestReadModeOverlap:
    def test_mean_field_stub_gives_uniform_weights_summing_to_one(self):
        components = stub_components()
        weights = read_mode_overlap(components, n_design_cells=4)
        np.testing.assert_allclose(weights, 0.25)

    def test_weights_are_the_background_eigen_adjoint(self):
        backgrounds = []

        def gyptis(design_epsilon, core_epsilon=None):
            backgrounds.append(core_epsilon)
            # A linear "eigenvalue" whose gradient is the weight vector; the
            # weights only matter at the uniform background.
            return jnp.sum(jnp.asarray([0.1, 0.2, 0.7]) * design_epsilon)

        weights = read_mode_overlap(
            stub_components(gyptis=gyptis), n_design_cells=3, background_epsilon=12.0
        )
        np.testing.assert_allclose(weights, [0.1, 0.2, 0.7])
        assert backgrounds == [12.0]


class TestCarrierFields:
    def test_returns_both_bias_states_in_cm3_full_node_order(self):
        from prismo.pipeline import REVERSE_BIAS_V, carrier_fields

        biases = []

        def ct(doping, bias_voltage, mesh_ref=None):
            biases.append(bias_voltage)
            scale = 1.0 if bias_voltage == 0.0 else 0.5
            return scale * jnp.abs(doping), 2.0 * scale * jnp.abs(doping)

        components = stub_components(chargetransport=ct)
        theta = jnp.asarray([0.3, -0.3])
        n0, p0, n1, p1 = carrier_fields(theta, components=components)
        assert sorted(biases) == [REVERSE_BIAS_V, 0.0]
        doping = np.abs(np.asarray(doping_from_theta(theta)))
        np.testing.assert_allclose(np.asarray(n0), doping)
        np.testing.assert_allclose(np.asarray(p0), 2.0 * doping)
        np.testing.assert_allclose(np.asarray(n1), 0.5 * doping)
        np.testing.assert_allclose(np.asarray(p1), doping)


def _depleting_stub(n_cm3: float, p_cm3: float):
    """ChargeTransport double whose carriers deplete linearly with reverse bias."""

    def ct(doping, bias_voltage, mesh_ref=None):
        depletion = 1.0 - 0.1 * abs(float(bias_voltage))
        return (
            jnp.full_like(doping, depletion * n_cm3),
            jnp.full_like(doping, depletion * p_cm3),
        )

    return ct


class TestBiasSweep:
    N = 4
    BIASES = (0.0, -2.5, -5.0)

    def _sweep(self, **kwargs):
        from prismo.pipeline import bias_sweep

        return bias_sweep(
            jnp.full(self.N, 0.3),
            self.BIASES,
            components=stub_components(chargetransport=_depleting_stub(1e18, 1e16)),
            **kwargs,
        )

    def test_one_point_per_bias_in_order(self):
        points = self._sweep()
        assert [p.bias_v for p in points] == list(self.BIASES)

    def test_zero_bias_has_no_shift_and_no_finite_efficiency(self):
        """Δneff is measured against 0 V, so the 0 V point is exactly zero."""
        zero = self._sweep()[0]
        assert zero.delta_neff == 0.0
        assert not np.isfinite(zero.vpi_lpi_v_cm)

    def test_delta_neff_grows_with_reverse_bias(self):
        dneff = [p.delta_neff for p in self._sweep()]
        assert dneff[0] < dneff[1] < dneff[2]

    def test_vpi_lpi_uses_each_point_own_bias(self):
        overlap = jnp.full(self.N, 1.0 / self.N)
        for point in self._sweep(mode_overlap=overlap)[1:]:
            assert point.vpi_lpi_v_cm == pytest.approx(
                vpi_lpi_v_cm(point.delta_neff, point.bias_v)
            )
            assert point.fom_v_db == pytest.approx(
                point.vpi_lpi_v_cm * point.modal_loss_db_cm
            )

    def test_loss_is_read_at_each_bias_not_at_equilibrium(self):
        """Depletion removes carriers, so alpha falls as the bias deepens."""
        overlap = jnp.full(self.N, 1.0 / self.N)
        losses = [p.modal_loss_db_cm for p in self._sweep(mode_overlap=overlap)]
        assert losses[0] > losses[1] > losses[2] > 0.0
        # The 0 V point is the same loss the objective reports.
        alpha = 8.5e-18 * 1e18 + 6.0e-18 * 1e16
        neff0 = np.sqrt(DEFAULT_BACKGROUND_EPSILON)
        assert losses[0] == pytest.approx((3.4757 / neff0) * alpha * NEPER_TO_DB)

    def test_without_overlap_the_loss_and_figure_of_merit_are_nan(self):
        for point in self._sweep():
            assert np.isnan(point.modal_loss_db_cm)
            assert np.isnan(point.fom_v_db)

    def test_the_equilibrium_is_solved_once_and_reused(self):
        from prismo.pipeline import bias_sweep

        biases = []

        def ct(doping, bias_voltage, mesh_ref=None):
            biases.append(float(bias_voltage))
            return _depleting_stub(1e18, 1e16)(doping, bias_voltage)

        bias_sweep(
            jnp.full(self.N, 0.3),
            self.BIASES,
            components=stub_components(chargetransport=ct),
        )
        assert biases == [0.0, -2.5, -5.0]
