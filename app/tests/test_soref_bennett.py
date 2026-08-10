"""Tests for the Soref-Bennett coupling layer."""

import numpy as np
import pytest
from prismo.soref_bennett import soref_bennett
from prismo_shared.schemas import (
    CarrierDensityField,
    SorefBennettCoefficients,
    SorefBennettResult,
)


def test_zero_carriers():
    """Zero density perturbation -> zero output."""
    carriers = CarrierDensityField(
        electrons=[1e24],
        holes=[1e24],
        equilibrium_electrons=[1e24],
        equilibrium_holes=[1e24],
    )
    result = soref_bennett(carriers)
    np.testing.assert_allclose(result.delta_permittivity, [0.0], atol=0.0)
    np.testing.assert_allclose(result.delta_absorption, [0.0], atol=0.0)


def test_hand_computed_values():
    """Hand-computed Soref-Bennett values for Delta_N = 1e18 cm^-3."""
    coeffs = SorefBennettCoefficients()
    dN = 1e18  # cm^-3 (input 1e24 m^-3 converts to 1e18 cm^-3)

    carriers = CarrierDensityField(
        electrons=[1e24],
        holes=[1e24],
        equilibrium_electrons=[0.0],
        equilibrium_holes=[0.0],
    )
    result = soref_bennett(carriers)

    exp_dn = -(
        coeffs.A_e * dN**coeffs.B_e + coeffs.A_h * dN**coeffs.B_h
    )
    exp_dalpha = coeffs.C_e * dN**coeffs.D_e + coeffs.C_h * dN**coeffs.D_h
    exp_depsilon = 2.0 * coeffs.background_index * exp_dn

    np.testing.assert_allclose(
        result.delta_permittivity, [exp_depsilon], rtol=1e-9
    )
    np.testing.assert_allclose(
        result.delta_absorption, [exp_dalpha], rtol=1e-9
    )


def test_equilibrium_subtraction():
    """Verify the equilibrium density is subtracted before scaling."""
    carriers = CarrierDensityField(
        electrons=[2e24],
        holes=[0.0],
        equilibrium_electrons=[1e24],
        equilibrium_holes=[0.0],
    )
    result = soref_bennett(carriers)

    # dn_e = (2e24 - 1e24) * 1e-6 = 1e18 cm^-3, dn_h = 0.
    exp_dalpha = 8.5e-18 * 1e18
    np.testing.assert_allclose(result.delta_absorption, [exp_dalpha], rtol=1e-9)


def test_depletion_increases_permittivity():
    """Carrier DEPLETION (dn < 0, reverse bias) raises the refractive index.

    Hand-computed for depletion of Delta_N = 1e18 cm^-3 (schema densities
    are m^-3): antisymmetric extension of Soref-Bennett gives
    dn_index = +(A_e * dN^B_e + A_h * dN^B_h) with dN = 1e18.
    Ref: ticket 17.
    """
    coeffs = SorefBennettCoefficients()
    dN = 1e18  # cm^-3

    carriers = CarrierDensityField(
        electrons=[0.0],
        holes=[0.0],
        equilibrium_electrons=[1e24],
        equilibrium_holes=[1e24],
    )
    result = soref_bennett(carriers)

    exp_dn = coeffs.A_e * dN**coeffs.B_e + coeffs.A_h * dN**coeffs.B_h
    exp_depsilon = 2.0 * coeffs.background_index * exp_dn
    np.testing.assert_allclose(result.delta_permittivity, [exp_depsilon], rtol=1e-9)
    assert result.delta_permittivity[0] > 0

    exp_dalpha = -(coeffs.C_e * dN**coeffs.D_e + coeffs.C_h * dN**coeffs.D_h)
    np.testing.assert_allclose(result.delta_absorption, [exp_dalpha], rtol=1e-9)


def test_antisymmetric_in_density_change():
    """sb(-dn) == -sb(+dn): depletion and injection are opposite."""
    up = CarrierDensityField(
        electrons=[1.5e24], holes=[1.5e24],
        equilibrium_electrons=[1e24], equilibrium_holes=[1e24],
    )
    down = CarrierDensityField(
        electrons=[0.5e24], holes=[0.5e24],
        equilibrium_electrons=[1e24], equilibrium_holes=[1e24],
    )
    r_up = soref_bennett(up)
    r_down = soref_bennett(down)

    np.testing.assert_allclose(
        r_down.delta_permittivity, [-x for x in r_up.delta_permittivity], rtol=1e-9,
    )
    np.testing.assert_allclose(
        r_down.delta_absorption, [-x for x in r_up.delta_absorption], rtol=1e-9,
    )


def test_missing_equilibrium_raises():
    """Missing equilibrium densities raise ValueError."""
    carriers = CarrierDensityField(electrons=[1e24], holes=[1e24])
    with pytest.raises(ValueError):
        soref_bennett(carriers)


def test_vectorized():
    """Works element-wise across multi-element arrays."""
    electrons = np.array([0.0, 1e24, 2e24, 3e24])
    holes = np.array([0.0, 1e24, 2e24, 3e24])
    carriers = CarrierDensityField(
        electrons=electrons.tolist(),
        holes=holes.tolist(),
        equilibrium_electrons=[0.0, 0.0, 0.0, 0.0],
        equilibrium_holes=[0.0, 0.0, 0.0, 0.0],
    )
    result = soref_bennett(carriers)

    coeffs = SorefBennettCoefficients()
    dn_e = electrons * 1e-6
    dn_h = holes * 1e-6
    exp_dn = -(coeffs.A_e * dn_e**coeffs.B_e + coeffs.A_h * dn_h**coeffs.B_h)
    exp_dalpha = coeffs.C_e * dn_e**coeffs.D_e + coeffs.C_h * dn_h**coeffs.D_h
    exp_depsilon = 2.0 * coeffs.background_index * exp_dn

    assert len(result.delta_permittivity) == 4
    assert len(result.delta_absorption) == 4
    np.testing.assert_allclose(result.delta_permittivity, exp_depsilon)
    np.testing.assert_allclose(result.delta_absorption, exp_dalpha)


def test_returns_correct_types():
    """Returns a SorefBennettResult with lists matching input length."""
    carriers = CarrierDensityField(
        electrons=[1e24, 2e24, 3e24],
        holes=[1e24, 2e24, 3e24],
        equilibrium_electrons=[1e23, 1e23, 1e23],
        equilibrium_holes=[1e23, 1e23, 1e23],
    )
    result = soref_bennett(carriers)

    assert isinstance(result, SorefBennettResult)
    assert isinstance(result.delta_permittivity, list)
    assert isinstance(result.delta_absorption, list)
    assert len(result.delta_permittivity) == 3
    assert len(result.delta_absorption) == 3
    assert all(isinstance(v, float) for v in result.delta_permittivity)
    assert all(isinstance(v, float) for v in result.delta_absorption)


def test_coefficient_defaults():
    """Using default coefficients equals passing them explicitly."""
    carriers = CarrierDensityField(
        electrons=[1e24],
        holes=[1e24],
        equilibrium_electrons=[0.0],
        equilibrium_holes=[0.0],
    )
    default_result = soref_bennett(carriers)
    explicit_result = soref_bennett(carriers, coeffs=SorefBennettCoefficients())

    np.testing.assert_allclose(
        default_result.delta_permittivity,
        explicit_result.delta_permittivity,
    )
    np.testing.assert_allclose(
        default_result.delta_absorption,
        explicit_result.delta_absorption,
    )
