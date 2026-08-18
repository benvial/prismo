"""End-to-end differentiable pipeline: rho -> delta_n_eff.

Composes density filter -> doping mapping -> ChargeTransport.jl (0V, -5V)
-> Soref-Bennett coupling -> gyptis -> delta_n_eff into a single
JAX-differentiable function.

Ref: tickets 14 (pipeline composition), 15 (optimization loop).
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from prismo_shared.schemas import MeshRef, SorefBennettCoefficients

from prismo.differentiable_component import (
    DifferentiableComponent,
    invoke_tesseract,
)

jax.config.update("jax_enable_x64", True)

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


@dataclass
class _PhaseTiming:
    """Aggregated component-boundary timing for one optimizer callback."""

    calls: int = 0
    seconds: float = 0.0
    cold_calls: int = 0
    cold_seconds: float = 0.0


_background_neff_sq: dict[tuple[tuple[int, ...], str, bytes], np.ndarray] = {}
_timing_lock = RLock()
_active_phase_timing: dict[str, _PhaseTiming] | None = None
_seen_timing_phases: set[str] = set()


def clear_pipeline_runtime_state() -> None:
    """Clear lifecycle-owned background modes and timing state.

    A background eigenmode is valid only while its component lifecycle and
    immutable optical configuration remain unchanged. Container startup and
    teardown both begin a new lifecycle.
    """
    global _active_phase_timing
    with _timing_lock:
        _background_neff_sq.clear()
        _active_phase_timing = None
        _seen_timing_phases.clear()
    shutdown_worker = getattr(_ct_api, "shutdown", None)
    if callable(shutdown_worker):
        shutdown_worker()


def begin_pipeline_callback_timing() -> None:
    """Start collecting component timings for one optimizer callback."""
    global _active_phase_timing
    with _timing_lock:
        _active_phase_timing = {}


def finish_pipeline_callback_timing() -> dict[str, dict[str, float | int]]:
    """Return timings collected since ``begin_pipeline_callback_timing``."""
    global _active_phase_timing
    with _timing_lock:
        phases = _active_phase_timing or {}
        _active_phase_timing = None
        return {
            name: {
                "calls": timing.calls,
                "seconds": timing.seconds,
                "cold_calls": timing.cold_calls,
                "cold_seconds": timing.cold_seconds,
                "warm_seconds": timing.seconds - timing.cold_seconds,
            }
            for name, timing in phases.items()
        }


def _record_phase_timing(name: str, started_at: float) -> None:
    """Record a component-boundary call without exposing solver internals."""
    global _active_phase_timing
    elapsed = time.perf_counter() - started_at
    with _timing_lock:
        cold = name not in _seen_timing_phases
        _seen_timing_phases.add(name)
        if _active_phase_timing is None:
            return
        timing = _active_phase_timing.setdefault(name, _PhaseTiming())
        timing.calls += 1
        timing.seconds += elapsed
        timing.cold_calls += int(cold)
        if cold:
            timing.cold_seconds += elapsed


def init_tesseract_containers() -> None:
    """Start tesseract Docker containers for ChargeTransport and gyptis.

    Must be called before optimization. Containers stay running until
    ``teardown_containers()`` is called. Startup is all-or-nothing so a
    container run can never silently use local component stubs.
    """
    global _ct_tesseract, _gyptis_tesseract, _HAS_CT_CONTAINER, _HAS_GYPTIS_CONTAINER
    clear_pipeline_runtime_state()
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
    clear_pipeline_runtime_state()


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


def _ct_container_inputs(
    doping_np: np.ndarray,
    bias_voltage: float,
    mesh_ref: MeshRef | None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "doping": doping_np.tolist(),
        "bias_voltage": float(bias_voltage),
    }
    if mesh_ref is not None:
        inputs["mesh_ref"] = mesh_ref.model_dump()
    return inputs


def _ct_forward_impl(
    doping_np: np.ndarray,
    bias_voltage: float,
    mesh_ref: MeshRef | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    started_at = time.perf_counter()
    phase = f"ct_forward_{bias_voltage:g}V"

    def from_container(tess: Any) -> tuple[np.ndarray, np.ndarray]:
        result = tess.apply(_ct_container_inputs(doping_np, bias_voltage, mesh_ref))
        return (
            np.asarray(result["electrons"], dtype=doping_np.dtype),
            np.asarray(result["holes"], dtype=doping_np.dtype),
        )

    def from_local(api: Any) -> tuple[np.ndarray, np.ndarray]:
        outputs = api.apply(
            api.InputSchema(
                doping=doping_np, bias_voltage=bias_voltage, mesh_ref=mesh_ref
            )
        )
        return (
            np.asarray(outputs.electrons, dtype=doping_np.dtype),
            np.asarray(outputs.holes, dtype=doping_np.dtype),
        )

    try:
        return invoke_tesseract(
            _ct_tesseract,
            _ct_api,
            container_call=from_container,
            local_call=from_local,
        )
    finally:
        _record_phase_timing(phase, started_at)


def _ct_vjp_impl(
    doping_np: np.ndarray,
    cotangent: tuple[np.ndarray, np.ndarray],
    bias_voltage: float,
    mesh_ref: MeshRef | None = None,
) -> np.ndarray:
    cot_n, cot_p = cotangent
    started_at = time.perf_counter()

    def from_container(tess: Any) -> np.ndarray:
        vjp_result = tess.vector_jacobian_product(
            _ct_container_inputs(doping_np, bias_voltage, mesh_ref),
            ["doping"],
            ["electrons", "holes"],
            {"electrons": cot_n.tolist(), "holes": cot_p.tolist()},
        )
        return np.asarray(vjp_result["doping"], dtype=doping_np.dtype)

    def from_local(api: Any) -> np.ndarray:
        vjp_result = api.vector_jacobian_product(
            api.InputSchema(
                doping=doping_np, bias_voltage=bias_voltage, mesh_ref=mesh_ref
            ),
            {"doping"},
            {"electrons", "holes"},
            {"electrons": cot_n, "holes": cot_p},
        )
        return np.asarray(vjp_result["doping"], dtype=doping_np.dtype)

    try:
        return invoke_tesseract(
            _ct_tesseract,
            _ct_api,
            container_call=from_container,
            local_call=from_local,
        )
    finally:
        _record_phase_timing("ct_vjp", started_at)


# -- ChargeTransport JAX wrapper -------------------------------------------------


def _ct_out_struct(
    doping: jax.Array,
    bias_voltage: float,
    mesh_ref: MeshRef | None = None,
) -> tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]:
    return _shaped_like(doping), _shaped_like(doping)


def _ct_stub_forward(
    doping: jax.Array,
    bias_voltage: float,
    mesh_ref: MeshRef | None = None,
) -> tuple[jax.Array, jax.Array]:
    return doping, doping


def _ct_stub_vjp(
    doping: jax.Array,
    g: tuple[jax.Array, jax.Array],
    bias_voltage: float,
    mesh_ref: MeshRef | None = None,
) -> jax.Array:
    g_electrons, g_holes = g
    return g_electrons + g_holes


_ct_call_jax = DifferentiableComponent(
    forward=_ct_forward_impl,
    vjp=_ct_vjp_impl,
    out_struct=_ct_out_struct,
    stub_forward=_ct_stub_forward,
    stub_vjp=_ct_stub_vjp,
    available=lambda: _HAS_CT or _HAS_CT_CONTAINER,
)


# -- gyptis external implementation ----------------------------------------------


def _gyptis_forward_impl(
    epsilon_np: np.ndarray,
    *,
    phase: str = "gyptis_perturbed_forward",
) -> np.ndarray:
    out_dtype = epsilon_np.dtype
    started_at = time.perf_counter()

    def from_container(tess: Any) -> np.ndarray:
        result = tess.apply({"epsilon": epsilon_np.tolist()})
        return np.asarray(result["neff_sq"], dtype=out_dtype)

    def from_local(api: Any) -> np.ndarray:
        outputs = api.apply(api.InputSchema(epsilon=epsilon_np))
        return np.asarray(outputs.neff_sq, dtype=out_dtype)

    try:
        return invoke_tesseract(
            _gyptis_tesseract,
            _gyptis_api,
            container_call=from_container,
            local_call=from_local,
        )
    finally:
        _record_phase_timing(phase, started_at)


def _gyptis_background_forward_impl(epsilon_np: np.ndarray) -> np.ndarray:
    """Evaluate immutable background epsilon once per component lifecycle."""
    key = (epsilon_np.shape, epsilon_np.dtype.str, epsilon_np.tobytes())
    with _timing_lock:
        cached = _background_neff_sq.get(key)
    if cached is not None:
        started_at = time.perf_counter()
        _record_phase_timing("gyptis_background_cache", started_at)
        return cached.copy()

    result = _gyptis_forward_impl(epsilon_np, phase="gyptis_background_forward")
    with _timing_lock:
        _background_neff_sq[key] = result.copy()
    return result


def _gyptis_vjp_impl(
    epsilon_np: np.ndarray,
    cot_neff_sq: np.ndarray,
) -> np.ndarray:
    out_dtype = epsilon_np.dtype
    started_at = time.perf_counter()

    def from_container(tess: Any) -> np.ndarray:
        vjp_result = tess.vector_jacobian_product(
            {"epsilon": epsilon_np.tolist()},
            ["epsilon"],
            ["neff_sq"],
            {"neff_sq": float(cot_neff_sq)},
        )
        return np.asarray(vjp_result["epsilon"], dtype=out_dtype)

    def from_local(api: Any) -> np.ndarray:
        vjp_result = api.vector_jacobian_product(
            api.InputSchema(epsilon=epsilon_np),
            {"epsilon"},
            {"neff_sq"},
            {"neff_sq": np.asarray(cot_neff_sq)},
        )
        return np.asarray(vjp_result["epsilon"], dtype=out_dtype)

    try:
        return invoke_tesseract(
            _gyptis_tesseract,
            _gyptis_api,
            container_call=from_container,
            local_call=from_local,
        )
    finally:
        _record_phase_timing("gyptis_vjp", started_at)


def _gyptis_background_vjp_impl(
    epsilon_np: np.ndarray,
    cot_neff_sq: np.ndarray,
) -> np.ndarray:
    """Background permittivity is rho-independent: its cotangent is zero."""
    return np.zeros_like(epsilon_np)


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


# -- gyptis JAX wrappers ---------------------------------------------------------


def _gyptis_out_struct(epsilon: jax.Array) -> jax.ShapeDtypeStruct:
    return _scalar_like(epsilon)


def _gyptis_stub_forward(epsilon: jax.Array) -> jax.Array:
    return jnp.mean(epsilon)


def _gyptis_stub_vjp(epsilon: jax.Array, g: jax.Array) -> jax.Array:
    n = epsilon.shape[0]
    return jnp.full_like(epsilon, g / n)


def _gyptis_background_stub_vjp(epsilon: jax.Array, g: jax.Array) -> jax.Array:
    return jnp.zeros_like(epsilon)


def _gyptis_available() -> bool:
    return _HAS_GYPTIS or _HAS_GYPTIS_CONTAINER


_gyptis_call_jax = DifferentiableComponent(
    forward=_gyptis_forward_impl,
    vjp=_gyptis_vjp_impl,
    out_struct=_gyptis_out_struct,
    stub_forward=_gyptis_stub_forward,
    stub_vjp=_gyptis_stub_vjp,
    available=_gyptis_available,
)


# Background solve: rho-independent, so it contributes a zero cotangent to rho.
_gyptis_background_call_jax = DifferentiableComponent(
    forward=_gyptis_background_forward_impl,
    vjp=_gyptis_background_vjp_impl,
    out_struct=_gyptis_out_struct,
    stub_forward=_gyptis_stub_forward,
    stub_vjp=_gyptis_background_stub_vjp,
    available=_gyptis_available,
)


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
    polarity: jax.Array | None = None,
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
        polarity: Fixed per-node P/N polarity applied to the positive doping
            magnitude. ``-1`` denotes p-type and ``1`` denotes n-type.
        mesh_ref: ``MeshRef`` forwarded to ChargeTransport calls.
        background_epsilon: Background Si relative permittivity
            (default: ``n_si^2 = 3.4757^2``).
        domain_count: Number of gyptis material domains.
        active_domains: Zero-based gyptis domains receiving the perturbation.

    Returns:
        Smooth positive effective-index shift magnitude between 0 V and -5 V.
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
    if polarity is not None:
        doping = doping * jnp.asarray(polarity, dtype=dtype)

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

    # Background epsilon does not depend on rho. Cache its eigenmode while
    # keeping the perturbed solve and eigen-adjoint live for every rho.
    neff_sq_0 = _gyptis_background_call_jax(epsilon_bg)
    neff_sq_1 = _gyptis_call_jax(epsilon_pert)

    neff_0 = jnp.sqrt(jnp.maximum(neff_sq_0, 0.0))
    neff_1 = jnp.sqrt(jnp.maximum(neff_sq_1, 0.0))

    delta_neff = neff_1 - neff_0
    # Keep a positive, differentiable objective when the mode shift is zero.
    return jnp.hypot(delta_neff, jnp.asarray(1e-15, dtype=delta_neff.dtype))
