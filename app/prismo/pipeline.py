"""End-to-end differentiable pipeline: rho -> delta_n_eff.

Composes density filter -> doping mapping -> ChargeTransport.jl (0V, -5V)
-> Soref-Bennett coupling -> gyptis -> delta_n_eff into a single
JAX-differentiable function.

Ref: tickets 14 (pipeline composition), 15 (optimization loop).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from prismo_shared.schemas import MeshRef, SorefBennettCoefficients

_COMPONENTS_DIR = Path(__file__).resolve().parents[2] / "components" / "tesseracts"


def _load_tesseract_api(name: str) -> Any | None:
    api_path = _COMPONENTS_DIR / name / "tesseract_api.py"
    if not api_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            f"_{name}_tesseract_api", api_path
        )
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_ct_api: Any | None = _load_tesseract_api("chargetransport")
_gyptis_api: Any | None = _load_tesseract_api("gyptis")

_DEFAULT_BACKGROUND_EPSILON: float = 3.4757**2
_DEFAULT_DOMAIN_COUNT: int = 3
_M3_TO_CM3: float = 1e-6
_DEFAULT_COEFFS: SorefBennettCoefficients = SorefBennettCoefficients()

_HAS_CT: bool = _ct_api is not None
_HAS_GYPTIS: bool = _gyptis_api is not None

_ct_tesseract: Any | None = None
_gyptis_tesseract: Any | None = None
_HAS_CT_CONTAINER: bool = False
_HAS_GYPTIS_CONTAINER: bool = False


def init_tesseract_containers() -> None:
    """Start tesseract Docker containers for ChargeTransport and gyptis.

    Must be called before optimization. Containers stay running until
    ``teardown_containers()`` is called. Startup is all-or-nothing so a
    container run can never silently use local component stubs.
    """
    global _ct_tesseract, _gyptis_tesseract, _HAS_CT_CONTAINER, _HAS_GYPTIS_CONTAINER
    try:
        from tesseract_core import Tesseract  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "tesseract_core is required for container pipeline runs"
        ) from exc

    try:
        _ct_tesseract = Tesseract.from_image("prismo_chargetransport:latest")
        _ct_tesseract.serve()
        _HAS_CT_CONTAINER = True
    except Exception as exc:
        teardown_containers()
        raise RuntimeError("Failed to start ChargeTransport container") from exc

    try:
        _gyptis_tesseract = Tesseract.from_image("prismo_gyptis:latest")
        _gyptis_tesseract.serve()
        _HAS_GYPTIS_CONTAINER = True
    except Exception as exc:
        teardown_containers()
        raise RuntimeError("Failed to start gyptis container") from exc


def teardown_containers() -> None:
    """Stop and remove running tesseract containers."""
    global _ct_tesseract, _gyptis_tesseract, _HAS_CT_CONTAINER, _HAS_GYPTIS_CONTAINER
    if _ct_tesseract is not None:
        _ct_tesseract.teardown()
        _ct_tesseract = None
    _HAS_CT_CONTAINER = False
    if _gyptis_tesseract is not None:
        _gyptis_tesseract.teardown()
        _gyptis_tesseract = None
    _HAS_GYPTIS_CONTAINER = False


def _shaped_like(arr: jax.Array) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(arr.shape, arr.dtype)


def _scalar_like(arr: jax.Array) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct((), arr.dtype)


# -- Density filter JAX wrapper --------------------------------------------------


@jax.custom_vjp
def _filter_jax(rho: jax.Array, H: jax.Array, H_sum: jax.Array) -> jax.Array:
    return (H @ rho) / H_sum


def _filter_jax_fwd(
    rho: jax.Array,
    H: jax.Array,
    H_sum: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    rho_tilde = (H @ rho) / H_sum
    return rho_tilde, (H, H_sum)


def _filter_jax_bwd(
    res: tuple[jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, None, None]:
    H, H_sum = res
    return H.T @ (g / H_sum), None, None


_filter_jax.defvjp(_filter_jax_fwd, _filter_jax_bwd)


# -- ChargeTransport external implementation -------------------------------------


def _ct_forward_impl(
    doping_np: np.ndarray,
    bias_voltage: float,
    mesh_ref: MeshRef | None,
) -> tuple[np.ndarray, np.ndarray]:
    if _ct_tesseract is not None:
        inputs: dict[str, Any] = {
            "doping": doping_np.tolist(),
            "bias_voltage": float(bias_voltage),
        }
        if mesh_ref is not None:
            inputs["mesh_ref"] = mesh_ref.model_dump()
        result = _ct_tesseract.apply(inputs)
        return (
            np.asarray(result["electrons"], dtype=doping_np.dtype),
            np.asarray(result["holes"], dtype=doping_np.dtype),
        )
    if _ct_api is None:
        return doping_np.copy(), doping_np.copy()
    inputs = _ct_api.InputSchema(
        doping=doping_np,
        bias_voltage=bias_voltage,
        mesh_ref=mesh_ref,
    )
    outputs = _ct_api.apply(inputs)
    return (
        np.asarray(outputs.electrons, dtype=doping_np.dtype),
        np.asarray(outputs.holes, dtype=doping_np.dtype),
    )


def _ct_vjp_impl(
    doping_np: np.ndarray,
    bias_voltage: float,
    mesh_ref: MeshRef | None,
    cot_n: np.ndarray,
    cot_p: np.ndarray,
) -> np.ndarray:
    if _ct_tesseract is not None:
        inputs: dict[str, Any] = {
            "doping": doping_np.tolist(),
            "bias_voltage": float(bias_voltage),
        }
        if mesh_ref is not None:
            inputs["mesh_ref"] = mesh_ref.model_dump()
        cotangent: dict[str, Any] = {
            "electrons": cot_n.tolist(),
            "holes": cot_p.tolist(),
        }
        vjp_result = _ct_tesseract.vector_jacobian_product(
            inputs,
            ["doping"],
            ["electrons", "holes"],
            cotangent,
        )
        return np.asarray(vjp_result["doping"], dtype=doping_np.dtype)
    if _ct_api is None:
        return cot_n + cot_p
    inputs = _ct_api.InputSchema(
        doping=doping_np,
        bias_voltage=bias_voltage,
        mesh_ref=mesh_ref,
    )
    ct_cotangent: dict[str, np.ndarray] = {
        "electrons": cot_n,
        "holes": cot_p,
    }
    vjp_result = _ct_api.vector_jacobian_product(
        inputs,
        {"doping"},
        {"electrons", "holes"},
        ct_cotangent,
    )
    return np.asarray(vjp_result["doping"], dtype=doping_np.dtype)


def _ct_call_impl(
    doping: jax.Array,
    bias_voltage: float,
    mesh_ref: MeshRef | None,
) -> tuple[jax.Array, jax.Array]:
    if not _HAS_CT and not _HAS_CT_CONTAINER:
        return doping, doping
    return jax.pure_callback(
        _ct_forward_impl,
        (_shaped_like(doping), _shaped_like(doping)),
        doping,
        bias_voltage,
        mesh_ref,
    )


# -- ChargeTransport JAX wrapper -------------------------------------------------


@jax.custom_vjp
def _ct_call_jax(
    doping: jax.Array,
    bias_voltage: float,
    mesh_ref: MeshRef | None = None,
) -> tuple[jax.Array, jax.Array]:
    return _ct_call_impl(doping, bias_voltage, mesh_ref)


def _ct_call_fwd(
    doping: jax.Array,
    bias_voltage: float,
    mesh_ref: MeshRef | None = None,
) -> tuple[
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, float, MeshRef | None],
]:
    return _ct_call_impl(doping, bias_voltage, mesh_ref), (
        doping,
        bias_voltage,
        mesh_ref,
    )


def _ct_call_bwd(
    res: tuple[jax.Array, float, MeshRef | None],
    g: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, None, None]:
    doping, bias_voltage, mesh_ref = res
    g_electrons, g_holes = g

    if not _HAS_CT and not _HAS_CT_CONTAINER:
        return g_electrons + g_holes, None, None

    vjp = jax.pure_callback(
        _ct_vjp_impl,
        _shaped_like(doping),
        doping,
        bias_voltage,
        mesh_ref,
        g_electrons,
        g_holes,
    )
    return vjp, None, None


_ct_call_jax.defvjp(_ct_call_fwd, _ct_call_bwd)


# -- gyptis external implementation ----------------------------------------------


def _gyptis_forward_impl(epsilon_np: np.ndarray) -> np.ndarray:
    out_dtype = epsilon_np.dtype
    if _gyptis_tesseract is not None:
        result = _gyptis_tesseract.apply({"epsilon": epsilon_np.tolist()})
        return np.asarray(result["neff_sq"], dtype=out_dtype)
    if _gyptis_api is None:
        return np.asarray(np.mean(epsilon_np), dtype=out_dtype)
    inputs = _gyptis_api.InputSchema(epsilon=epsilon_np)
    outputs = _gyptis_api.apply(inputs)
    return np.asarray(outputs.neff_sq, dtype=out_dtype)


def _gyptis_vjp_impl(
    epsilon_np: np.ndarray,
    cot_neff_sq: np.ndarray,
) -> np.ndarray:
    n = len(epsilon_np)
    out_dtype = epsilon_np.dtype
    if _gyptis_tesseract is not None:
        cotangent = {"neff_sq": float(cot_neff_sq)}
        vjp_result = _gyptis_tesseract.vector_jacobian_product(
            {"epsilon": epsilon_np.tolist()},
            ["epsilon"],
            ["neff_sq"],
            cotangent,
        )
        return np.asarray(vjp_result["epsilon"], dtype=out_dtype)
    if _gyptis_api is None:
        return np.full(n, float(cot_neff_sq) / n, dtype=out_dtype)
    inputs = _gyptis_api.InputSchema(epsilon=epsilon_np)
    vjp_result = _gyptis_api.vector_jacobian_product(
        inputs,
        {"epsilon"},
        {"neff_sq"},
        {"neff_sq": np.asarray(cot_neff_sq)},
    )
    return np.asarray(vjp_result["epsilon"], dtype=out_dtype)


def _build_domain_epsilon(
    delta_eps: jax.Array,
    background_epsilon: jax.Array,
    domain_count: int,
    active_domains: tuple[int, ...] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Map nodal silicon perturbation onto gyptis material domains."""
    if domain_count < 1:
        raise ValueError("domain_count must be positive")
    if active_domains is None:
        active_domains = (0,)
    if any(index < 0 or index >= domain_count for index in active_domains):
        raise ValueError("active_domains contains an invalid domain index")

    delta_mean = jnp.mean(delta_eps)
    epsilon_bg = jnp.full((domain_count,), background_epsilon, dtype=delta_eps.dtype)
    epsilon_pert = epsilon_bg.at[jnp.asarray(active_domains)].add(delta_mean)
    return epsilon_bg, epsilon_pert


def _gyptis_call_impl(epsilon: jax.Array) -> jax.Array:
    if not _HAS_GYPTIS and not _HAS_GYPTIS_CONTAINER:
        return jnp.mean(epsilon)
    return jax.pure_callback(
        _gyptis_forward_impl,
        _scalar_like(epsilon),
        epsilon,
    )


# -- gyptis JAX wrapper ----------------------------------------------------------


@jax.custom_vjp
def _gyptis_call_jax(epsilon: jax.Array) -> jax.Array:
    return _gyptis_call_impl(epsilon)


def _gyptis_call_fwd(
    epsilon: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array]]:
    return _gyptis_call_impl(epsilon), (epsilon,)


def _gyptis_call_bwd(
    res: tuple[jax.Array],
    g: jax.Array,
) -> tuple[jax.Array]:
    (epsilon,) = res

    if not _HAS_GYPTIS and not _HAS_GYPTIS_CONTAINER:
        n = epsilon.shape[0]
        return (jnp.full_like(epsilon, g / n),)

    vjp = jax.pure_callback(
        _gyptis_vjp_impl,
        _shaped_like(epsilon),
        epsilon,
        g,
    )
    return (vjp,)


_gyptis_call_jax.defvjp(_gyptis_call_fwd, _gyptis_call_bwd)


# 1 cm^-3 = 1e6 m^-3: ChargeTransport output -> Soref-Bennett input.
_CM3_TO_M3 = 1e6


# -- Soref-Bennett (pure JAX) ----------------------------------------------------


@jax.custom_vjp
def _signed_pow(x: jax.Array, p: float) -> jax.Array:
    """Odd extension of ``x**p``: ``sign(x) * |x|**p``.

    Soref-Bennett is calibrated for carrier injection (dn > 0); reverse
    bias depletes carriers (dn < 0), where a fractional ``x**p`` is
    undefined. The antisymmetric extension keeps depletion physically
    correct: removing carriers raises the refractive index (ticket 17).
    """
    return jnp.sign(x) * jnp.abs(x) ** p


def _signed_pow_fwd(
    x: jax.Array,
    p: float,
) -> tuple[jax.Array, tuple[jax.Array, float]]:
    return _signed_pow(x, p), (x, p)


def _signed_pow_bwd(
    res: tuple[jax.Array, float],
    g: jax.Array,
) -> tuple[jax.Array, None]:
    x, p = res
    grad_x = jnp.where(x != 0, p * jnp.abs(x) ** (p - 1.0), 0.0)
    return g * grad_x, None


_signed_pow.defvjp(_signed_pow_fwd, _signed_pow_bwd)


def _sb_jax(
    electrons: jax.Array,
    holes: jax.Array,
    eq_electrons: jax.Array,
    eq_holes: jax.Array,
    coeffs: SorefBennettCoefficients | None = None,
) -> tuple[jax.Array, jax.Array]:
    if coeffs is None:
        coeffs = _DEFAULT_COEFFS

    dn_e = (electrons - eq_electrons) * _M3_TO_CM3
    dn_h = (holes - eq_holes) * _M3_TO_CM3

    dn = -(
        coeffs.A_e * _signed_pow(dn_e, coeffs.B_e)
        + coeffs.A_h * _signed_pow(dn_h, coeffs.B_h)
    )
    dalpha = coeffs.C_e * _signed_pow(dn_e, coeffs.D_e) + coeffs.C_h * _signed_pow(
        dn_h, coeffs.D_h
    )
    depsilon = 2.0 * coeffs.background_index * dn

    return depsilon, dalpha


# -- Pipeline --------------------------------------------------------------------


def pipeline(
    rho: jax.Array,
    H: jax.Array | None = None,
    H_sum: jax.Array | None = None,
    mesh_ref: MeshRef | None = None,
    background_epsilon: float | None = None,
    domain_count: int = _DEFAULT_DOMAIN_COUNT,
    active_domains: tuple[int, ...] | None = None,
) -> jax.Array:
    """Rho -> Delta n_eff differentiable pipeline.

    Args:
        rho: Normalized doping density per node in [0,1], shape ``(n_nodes,)``.
        H: Dense filter matrix, shape ``(n_nodes, n_nodes)``. Skip filter if
            ``None``.
        H_sum: Pre-computed row sums of ``H``.
        mesh_ref: ``MeshRef`` forwarded to ChargeTransport calls.
        background_epsilon: Background Si relative permittivity
            (default: ``n_si^2 = 3.4757^2``).
        domain_count: Number of gyptis material domains.
        active_domains: Zero-based gyptis domains receiving the perturbation.

    Returns:
        ``Delta n_eff = n_eff(0 V) - n_eff(-5 V)``.
    """
    if background_epsilon is None:
        background_epsilon = _DEFAULT_BACKGROUND_EPSILON

    rho = jnp.asarray(rho)

    # 1. Density filter: rho -> rho_tilde
    if H is not None:
        if H_sum is None:
            H_sum = jnp.sum(H, axis=1)
        rho_tilde = _filter_jax(rho, H, H_sum)
    else:
        rho_tilde = rho

    # 2. Doping mapping: rho_tilde -> N = 10^(14 + 7*rho_tilde) [cm^-3]
    dtype = rho_tilde.dtype
    doping = jnp.power(
        jnp.asarray(10.0, dtype=dtype),
        jnp.asarray(14.0, dtype=dtype) + jnp.asarray(7.0, dtype=dtype) * rho_tilde,
    )

    # 3. ChargeTransport at equilibrium (0 V)
    n0, p0 = _ct_call_jax(doping, 0.0, mesh_ref)

    # 4. ChargeTransport at reverse bias (-5 V)
    n1, p1 = _ct_call_jax(doping, -5.0, mesh_ref)

    # CT reports carrier densities in cm^-3 (same unit system as the doping
    # input); Soref-Bennett consumes m^-3 per CarrierDensityField. Convert
    # at the component boundary (ticket 17).
    n0 = n0 * _CM3_TO_M3
    p0 = p0 * _CM3_TO_M3
    n1 = n1 * _CM3_TO_M3
    p1 = p1 * _CM3_TO_M3

    # 5. Soref-Bennett coupling (equilibrium-subtracted)
    delta_eps, _ = _sb_jax(n1, p1, n0, p0)

    # 6. gyptis eigenmode solves
    bg = jnp.asarray(background_epsilon, dtype=delta_eps.dtype)
    epsilon_bg, epsilon_pert = _build_domain_epsilon(
        delta_eps,
        bg,
        domain_count,
        active_domains,
    )

    neff_sq_0 = _gyptis_call_jax(epsilon_bg)
    neff_sq_1 = _gyptis_call_jax(epsilon_pert)

    neff_0 = jnp.sqrt(jnp.maximum(neff_sq_0, 0.0))
    neff_1 = jnp.sqrt(jnp.maximum(neff_sq_1, 0.0))

    return neff_0 - neff_1
