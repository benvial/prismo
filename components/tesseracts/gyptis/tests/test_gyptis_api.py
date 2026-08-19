"""Seam tests for the gyptis field-epsilon Tesseract component.

Public interface under test: apply() + vector_jacobian_product() in
tesseract_api.py. The forward accepts a design-region permittivity *field*
(one value per DG0 design cell) with constant surroundings, and the adjoint
returns a per-design-cell cotangent (tickets 02/03).

Covers schema validation, contract shapes, the effective-medium stub used when
gyptis/FEniCS is absent, and gyptis-backed integration tests (guided mode,
spatial response, single-pass field VJP vs finite differences).
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


def test_apply_returns_output_schema() -> None:
    assert isinstance(apply(make_inputs()), OutputSchema)


def test_apply_returns_scalar_neff_sq() -> None:
    assert np.asarray(apply(make_inputs()).neff_sq).shape == ()


def test_apply_returns_finite_neff_sq() -> None:
    outputs = apply(make_inputs(np.array([12.0, 11.0, 12.5, 11.5])))
    assert np.isfinite(float(outputs.neff_sq))


def test_apply_rejects_empty_field() -> None:
    with pytest.raises(ValueError):
        apply(make_inputs(np.array([])))


# ---------------------------------------------------------------------------
# vector_jacobian_product() contract
# ---------------------------------------------------------------------------


def test_vjp_returns_cotangent_shaped_like_design_field() -> None:
    inputs = make_inputs()
    apply(inputs)
    result = vector_jacobian_product(
        inputs, {"design_epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
    )
    assert set(result.keys()) == {"design_epsilon"}
    assert np.asarray(result["design_epsilon"]).shape == (N_DESIGN,)


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
# Effective-medium stub (gyptis / FEniCS absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_gyptis_available(), reason="exercises the no-gyptis stub")
def test_apply_stub_averages_field() -> None:
    outputs = apply(make_inputs(np.array([1.0, 2.0, 3.0, 4.0])))
    np.testing.assert_allclose(outputs.neff_sq, 2.5)


@pytest.mark.skipif(_gyptis_available(), reason="exercises the no-gyptis stub")
def test_vjp_stub_spreads_field_cotangent_evenly() -> None:
    inputs = make_inputs()
    result = vector_jacobian_product(
        inputs, {"design_epsilon"}, {"neff_sq"}, {"neff_sq": 1.0}
    )
    np.testing.assert_allclose(
        result["design_epsilon"], np.full(N_DESIGN, 1 / N_DESIGN)
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
