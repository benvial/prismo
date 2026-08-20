"""End-to-end differentiable pipeline: rho -> delta_n_eff.

Composes density filter -> doping mapping -> ChargeTransport.jl (0V, -5V)
-> Soref-Bennett coupling -> gyptis -> delta_n_eff into a single
JAX-differentiable function.

Ref: tickets 14 (pipeline composition), 15 (optimization loop).
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
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

# Signed net-doping map, zero-referenced and continuous through theta=0:
#   N(theta) = sign(theta) * DOPING_REFERENCE_CM3 * (10^(DOPING_LOG10_SPAN*|theta|) - 1)
# theta in [-1, 1] is the single signed design field. sign(theta) is the free
# P/N polarity, so the junction is exactly the zero-crossing (the optimizer moves
# it) and counterdoping is not representable. theta=0 -> 0 (net-intrinsic); a
# larger |theta| dopes harder, saturating at |N| ~ 1e19 cm^-3 at |theta|=1 -- the
# largest reverse-bias-stable concentration on the shared ChargeTransport mesh,
# within the B/P solid-solubility and Boltzmann-statistics window. Code = comment
# = glossary (CONTEXT.md).
DOPING_REFERENCE_CM3 = 1e14
DOPING_LOG10_SPAN = 5.0


@jax.custom_jvp
def doping_from_theta(theta: jax.Array) -> jax.Array:
    """Map the signed design field ``theta`` to signed net doping in ``cm^-3``.

    ``N(theta) = sign(theta) * 1e14 * (10^(span*|theta|) - 1)``. The map is
    zero-referenced (``N(0) = 0``) and antisymmetric, so a single ``theta``
    sign-crossing is a single P/N junction and no node can counterdope.
    """
    theta = jnp.asarray(theta)
    span = jnp.asarray(DOPING_LOG10_SPAN, dtype=theta.dtype)
    reference = jnp.asarray(DOPING_REFERENCE_CM3, dtype=theta.dtype)
    magnitude = reference * (jnp.power(10.0, span * jnp.abs(theta)) - 1.0)
    return jnp.sign(theta) * magnitude


@doping_from_theta.defjvp
def _doping_from_theta_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Analytic derivative ``dN/dtheta = 1e14 * ln10 * span * 10^(span*|theta|)``.

    The map is C^1 through the junction: the derivative is even and continuous,
    with the true slope ``1e14 * ln10 * span`` at ``theta=0``. Autodiff of the
    raw ``sign(theta) * |theta|`` form collapses to zero there (``sign(0)**2``),
    so the crossing -- exactly where the optimizer moves the junction -- is given
    its correct one-sided limit here.
    """
    (theta,) = primals
    (theta_dot,) = tangents
    theta = jnp.asarray(theta)
    span = jnp.asarray(DOPING_LOG10_SPAN, dtype=theta.dtype)
    reference = jnp.asarray(DOPING_REFERENCE_CM3, dtype=theta.dtype)
    derivative = (
        reference
        * jnp.log(jnp.asarray(10.0, dtype=theta.dtype))
        * span
        * jnp.power(10.0, span * jnp.abs(theta))
    )
    return doping_from_theta(theta), derivative * theta_dot


# Initial junction magnitude for the signed design field: moderate, non-degenerate
# doping the reverse-bias solve converges on, well inside the [-1, 1] bounds.
_JUNCTION_SEED_THETA = 0.3


def seed_signed_junction(coords: np.ndarray) -> jax.Array:
    """Seed a signed lateral P/N junction across the mesh in every run path.

    Nodes left of the median x seed p-type (``-0.3``), nodes to the right seed
    n-type (``+0.3``). sign(theta) is a free design variable, so the optimizer
    can move or dissolve this junction rather than being handed a fixed polarity.
    """
    midpoint = np.median(coords[:, 0])
    return jnp.where(
        coords[:, 0] <= midpoint, -_JUNCTION_SEED_THETA, _JUNCTION_SEED_THETA
    )


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


_DEFAULT_BACKGROUND_EPSILON: float = 3.4757**2
_M3_TO_CM3: float = 1e-6
_DEFAULT_COEFFS: SorefBennettCoefficients = SorefBennettCoefficients()


@dataclass
class _PhaseTiming:
    """Aggregated component-boundary timing for one optimizer callback."""

    calls: int = 0
    seconds: float = 0.0
    cold_calls: int = 0
    cold_seconds: float = 0.0


class PhaseTimer:
    """Per-component boundary timing owned by the components, not a global.

    Records elapsed time per phase label (e.g. ``ct_forward_0V``, ``ct_vjp``),
    tracking cold (first-use) versus warm cost. ``collect`` returns the tallies
    accumulated since the previous ``collect`` and resets them for the next
    optimizer callback, while the cold/warm phase memory persists for the
    component's lifetime.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._tallies: dict[str, _PhaseTiming] = {}
        self._seen: set[str] = set()

    def record(self, name: str, started_at: float) -> None:
        """Record one component-boundary call without exposing solver internals."""
        elapsed = time.perf_counter() - started_at
        with self._lock:
            cold = name not in self._seen
            self._seen.add(name)
            timing = self._tallies.setdefault(name, _PhaseTiming())
            timing.calls += 1
            timing.seconds += elapsed
            timing.cold_calls += int(cold)
            if cold:
                timing.cold_seconds += elapsed

    def collect(self) -> dict[str, dict[str, float | int]]:
        """Return and reset the tallies accumulated since the last collect."""
        with self._lock:
            result = {
                name: {
                    "calls": timing.calls,
                    "seconds": timing.seconds,
                    "cold_calls": timing.cold_calls,
                    "cold_seconds": timing.cold_seconds,
                    "warm_seconds": timing.seconds - timing.cold_seconds,
                }
                for name, timing in self._tallies.items()
            }
            self._tallies = {}
            return result


@dataclass(frozen=True)
class TimedComponent:
    """A differentiable component paired with the timer it records into.

    Callable exactly like the wrapped component and transparently delegating
    its attributes, so ``pipeline()`` composes it unchanged. The optimizer
    reads per-callback timings from ``timer`` after each callback instead of a
    global begin/finish protocol.
    """

    component: DifferentiableComponent
    timer: PhaseTimer

    def __call__(self, x: jax.Array, *static: Any) -> Any:
        """Evaluate the wrapped component."""
        return self.component(x, *static)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.component, name)


def _close_all(
    closers: tuple[Callable[[], None], ...] | list[Callable[[], None]],
) -> None:
    """Run resource closers in reverse (last-acquired first)."""
    for close in reversed(closers):
        close()


def init_tesseract_containers(
    mesh_dir: str | Path | None = None,
) -> PipelineComponents:
    """Start tesseract Docker containers and bundle the live components.

    Returns a :class:`PipelineComponents` the caller owns and passes to
    ``pipeline()``. Containers stay running until the bundle's ``close()`` is
    called (see ``teardown_containers``). Startup is all-or-nothing so a
    container run can never silently use local component stubs.

    Args:
        mesh_dir: Host directory holding the shared ``.msh`` file. It is
            bind-mounted read-only into the ChargeTransport container at
            :data:`_CT_MESH_MOUNT` so its Julia solver can load the real 2D
            grid instead of the 1D fallback. The bind mount is live, so a mesh
            written after the container starts is still visible.
    """
    try:
        from tesseract_core import Tesseract  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "tesseract_core is required for container pipeline runs"
        ) from exc

    ct_volumes: list[str] | None = None
    if mesh_dir is not None:
        host_mesh_dir = Path(mesh_dir).resolve()
        host_mesh_dir.mkdir(parents=True, exist_ok=True)
        ct_volumes = [f"{host_mesh_dir}:{_CT_MESH_MOUNT}:ro"]

    closers: list[Callable[[], None]] = []
    try:
        ct_tesseract = Tesseract.from_image(
            "prismo_chargetransport:latest", volumes=ct_volumes
        )
        ct_tesseract.serve()
        closers.append(ct_tesseract.teardown)
    except Exception as exc:
        _close_all(closers)
        raise RuntimeError("Failed to start ChargeTransport container") from exc

    try:
        gyptis_tesseract = Tesseract.from_image("prismo_gyptis:latest")
        gyptis_tesseract.serve()
        closers.append(gyptis_tesseract.teardown)
    except Exception as exc:
        _close_all(closers)
        raise RuntimeError("Failed to start gyptis container") from exc

    chargetransport = build_chargetransport_component(container=ct_tesseract)
    gyptis, gyptis_background = build_gyptis_components(container=gyptis_tesseract)
    return PipelineComponents(
        chargetransport=chargetransport,
        gyptis=gyptis,
        gyptis_background=gyptis_background,
        design_cell_centroids=partial(
            read_gyptis_design_cell_centroids, container=gyptis_tesseract
        ),
        design_cell_vertices=partial(
            read_gyptis_design_cell_vertices, container=gyptis_tesseract
        ),
        closers=tuple(closers),
    )


def teardown_containers(components: PipelineComponents) -> None:
    """Stop and remove the running tesseract containers owned by ``components``."""
    components.close()


def read_gyptis_design_cell_centroids(
    *, container: Any | None = None, local_api: Any | None = None
) -> np.ndarray:
    """Read gyptis design-cell centroids in ``design_epsilon`` field order.

    The Tesseract API's fixed endpoint set exposes this static geometry query
    through ``apply(operation="design_cell_centroids")``. A live gyptis/FEniCS
    backend is required; unlike a solve, there is no meaningful local stub for
    its mesh-dependent cells.
    """

    def from_container(tess: Any) -> np.ndarray:
        result = tess.apply({"operation": "design_cell_centroids"})
        return np.asarray(result["design_cell_centroids"], dtype=float)

    def from_local(api: Any) -> np.ndarray:
        outputs = api.apply(api.InputSchema(operation="design_cell_centroids"))
        return np.asarray(outputs.design_cell_centroids, dtype=float)

    centroids = invoke_tesseract(
        container,
        local_api,
        container_call=from_container,
        local_call=from_local,
    )
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError("gyptis design-cell centroids must have shape (n_design, 2)")
    return centroids


def read_gyptis_design_cell_vertices(
    *, container: Any | None = None, local_api: Any | None = None
) -> np.ndarray:
    """Read the ``(n_design, 3, 2)`` design-cell vertex coordinates.

    Each gyptis design cell is a triangle of the shared unified mesh; its three
    vertices are shared-mesh nodes. The host matches these coordinates to node
    indices to assemble the exact node->DG0-cell restriction operator (ticket
    05). Exposed through the ``write_mesh`` operation. Requires a gyptis backend.
    """

    def from_container(tess: Any) -> np.ndarray:
        result = tess.apply({"operation": "write_mesh"})
        return np.asarray(result["design_cell_vertices"], dtype=float)

    def from_local(api: Any) -> np.ndarray:
        outputs = api.apply(api.InputSchema(operation="write_mesh"))
        return np.asarray(outputs.design_cell_vertices, dtype=float)

    vertices = invoke_tesseract(
        container,
        local_api,
        container_call=from_container,
        local_call=from_local,
    )
    if vertices.ndim != 3 or vertices.shape[1:] != (3, 2):
        raise ValueError(
            "gyptis design-cell vertices must have shape (n_design, 3, 2)"
        )
    return vertices


def build_design_transfer(
    components: PipelineComponents,
    node_coords: np.ndarray,
) -> jax.Array:
    """Assemble the dense mesh-transfer matrix for the container gyptis path.

    Both solvers share one gmsh geometry (ticket 05), so the transfer is an exact
    local restriction: each gyptis design cell is a triangle of the shared mesh
    whose three vertices are shared-mesh nodes. Reads the design-cell vertices
    from the gyptis backend (in ``design_epsilon`` order) and matches them to
    ``node_coords`` (the same shared-mesh nodes the ChargeTransport solve and the
    density filter use), returning the ``(n_design_cells, n_nodes)`` matrix that
    feeds ``pipeline(design_transfer=...)`` directly.
    """
    from prismo.mesh_transfer import build_mesh_transfer_operator

    if components.design_cell_vertices is None:
        raise RuntimeError(
            "Container pipeline requires a gyptis backend exposing design-cell "
            "vertices to build the mesh-transfer operator"
        )
    vertices = components.design_cell_vertices()
    operator = build_mesh_transfer_operator(node_coords, vertices)
    return jnp.asarray(operator.dense())


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


# -- ChargeTransport component ---------------------------------------------------

# Read-only mountpoint where the shared mesh directory is bind-mounted into the
# ChargeTransport container (see ``init_tesseract_containers``). The container
# has no access to host paths, so a host ``mesh_ref.path`` is rewritten to this
# mount before the request crosses the container boundary.
_CT_MESH_MOUNT = "/tesseract/mesh"


def _container_mesh_ref(mesh_ref: MeshRef | None) -> MeshRef | None:
    """Rewrite a host ``mesh_ref`` path to its in-container mount location."""
    if mesh_ref is None:
        return None
    container_path = f"{_CT_MESH_MOUNT}/{Path(mesh_ref.path).name}"
    return mesh_ref.model_copy(update={"path": container_path})


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


def _ct_out_struct(
    doping: jax.Array,
    bias_voltage: float,
    mesh_ref: MeshRef | None = None,
) -> tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]:
    return _shaped_like(doping), _shaped_like(doping)


def build_chargetransport_component(
    container: Any | None = None,
    local_api: Any | None = None,
) -> TimedComponent:
    """Build the ChargeTransport component bound to one backend.

    ``container`` is a running Tesseract handle; ``local_api`` an in-process
    ``tesseract_api`` module. With neither, calling the component raises -- there
    is no physics-free identity fallback. The backend is captured here, not read
    from module globals. Returns a :class:`TimedComponent` owning the phase timer
    it records into.
    """
    timer = PhaseTimer()

    def forward(
        doping_np: np.ndarray,
        bias_voltage: float,
        mesh_ref: MeshRef | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        started_at = time.perf_counter()
        phase = f"ct_forward_{bias_voltage:g}V"

        def from_container(tess: Any) -> tuple[np.ndarray, np.ndarray]:
            result = tess.apply(
                _ct_container_inputs(
                    doping_np, bias_voltage, _container_mesh_ref(mesh_ref)
                )
            )
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
                container,
                local_api,
                container_call=from_container,
                local_call=from_local,
            )
        finally:
            timer.record(phase, started_at)

    def vjp(
        doping_np: np.ndarray,
        cotangent: tuple[np.ndarray, np.ndarray],
        bias_voltage: float,
        mesh_ref: MeshRef | None = None,
    ) -> np.ndarray:
        cot_n, cot_p = cotangent
        started_at = time.perf_counter()

        def from_container(tess: Any) -> np.ndarray:
            vjp_result = tess.vector_jacobian_product(
                _ct_container_inputs(
                    doping_np, bias_voltage, _container_mesh_ref(mesh_ref)
                ),
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
                container,
                local_api,
                container_call=from_container,
                local_call=from_local,
            )
        finally:
            timer.record("ct_vjp", started_at)

    component = DifferentiableComponent(
        forward=forward,
        vjp=vjp,
        out_struct=_ct_out_struct,
    )
    return TimedComponent(component, timer)


# -- gyptis component ------------------------------------------------------------


# The gyptis field-epsilon component takes the design-region permittivity field
# as its differentiated input and the constant core background as a static arg
# (so non-design core cells match the pipeline's background_epsilon).


def _gyptis_out_struct(
    design_epsilon: jax.Array, core_epsilon: float = _DEFAULT_BACKGROUND_EPSILON
) -> jax.ShapeDtypeStruct:
    return _scalar_like(design_epsilon)


def _gyptis_background_vjp_impl(
    design_epsilon_np: np.ndarray,
    cot_neff_sq: np.ndarray,
    core_epsilon: float = _DEFAULT_BACKGROUND_EPSILON,
) -> np.ndarray:
    """Background permittivity is rho-independent: its cotangent is zero."""
    return np.zeros_like(design_epsilon_np)


def build_gyptis_components(
    container: Any | None = None,
    local_api: Any | None = None,
) -> tuple[TimedComponent, TimedComponent]:
    """Build the perturbed and background gyptis components for one backend.

    Both share a background eigenmode cache owned by this call, so the
    rho-independent background solve runs once per component lifecycle, and one
    phase timer, so their combined boundary timings read from one place. The
    backend is captured here, not read from module globals.
    """
    background_cache: dict[tuple[tuple[int, ...], str, bytes, float], np.ndarray] = {}
    cache_lock = RLock()
    timer = PhaseTimer()

    def forward(
        design_epsilon_np: np.ndarray,
        core_epsilon: float = _DEFAULT_BACKGROUND_EPSILON,
        *,
        phase: str = "gyptis_perturbed_forward",
    ) -> np.ndarray:
        out_dtype = design_epsilon_np.dtype
        started_at = time.perf_counter()

        def from_container(tess: Any) -> np.ndarray:
            result = tess.apply(
                {
                    "design_epsilon": design_epsilon_np.tolist(),
                    "core_epsilon": float(core_epsilon),
                }
            )
            return np.asarray(result["neff_sq"], dtype=out_dtype)

        def from_local(api: Any) -> np.ndarray:
            outputs = api.apply(
                api.InputSchema(
                    design_epsilon=design_epsilon_np, core_epsilon=float(core_epsilon)
                )
            )
            return np.asarray(outputs.neff_sq, dtype=out_dtype)

        try:
            return invoke_tesseract(
                container,
                local_api,
                container_call=from_container,
                local_call=from_local,
            )
        finally:
            timer.record(phase, started_at)

    def background_forward(
        design_epsilon_np: np.ndarray,
        core_epsilon: float = _DEFAULT_BACKGROUND_EPSILON,
    ) -> np.ndarray:
        key = (
            design_epsilon_np.shape,
            design_epsilon_np.dtype.str,
            design_epsilon_np.tobytes(),
            float(core_epsilon),
        )
        with cache_lock:
            cached = background_cache.get(key)
        if cached is not None:
            started_at = time.perf_counter()
            timer.record("gyptis_background_cache", started_at)
            return cached.copy()

        result = forward(
            design_epsilon_np, core_epsilon, phase="gyptis_background_forward"
        )
        with cache_lock:
            background_cache[key] = result.copy()
        return result

    def vjp(
        design_epsilon_np: np.ndarray,
        cot_neff_sq: np.ndarray,
        core_epsilon: float = _DEFAULT_BACKGROUND_EPSILON,
    ) -> np.ndarray:
        out_dtype = design_epsilon_np.dtype
        started_at = time.perf_counter()

        def from_container(tess: Any) -> np.ndarray:
            vjp_result = tess.vector_jacobian_product(
                {
                    "design_epsilon": design_epsilon_np.tolist(),
                    "core_epsilon": float(core_epsilon),
                },
                ["design_epsilon"],
                ["neff_sq"],
                {"neff_sq": float(cot_neff_sq)},
            )
            return np.asarray(vjp_result["design_epsilon"], dtype=out_dtype)

        def from_local(api: Any) -> np.ndarray:
            vjp_result = api.vector_jacobian_product(
                api.InputSchema(
                    design_epsilon=design_epsilon_np, core_epsilon=float(core_epsilon)
                ),
                {"design_epsilon"},
                {"neff_sq"},
                {"neff_sq": np.asarray(cot_neff_sq)},
            )
            return np.asarray(vjp_result["design_epsilon"], dtype=out_dtype)

        try:
            return invoke_tesseract(
                container,
                local_api,
                container_call=from_container,
                local_call=from_local,
            )
        finally:
            timer.record("gyptis_vjp", started_at)

    perturbed = DifferentiableComponent(
        forward=forward,
        vjp=vjp,
        out_struct=_gyptis_out_struct,
    )
    # Background solve: rho-independent, so it contributes a zero cotangent.
    background = DifferentiableComponent(
        forward=background_forward,
        vjp=_gyptis_background_vjp_impl,
        out_struct=_gyptis_out_struct,
    )
    return TimedComponent(perturbed, timer), TimedComponent(background, timer)


def _build_design_epsilon(
    delta_eps: jax.Array,
    background_epsilon: jax.Array,
    design_transfer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Build the gyptis background and perturbed design-region permittivity fields.

    The nodal permittivity perturbation is carried onto the gyptis design cells
    by the mesh-transfer operator (ticket 04), preserving its spatial structure
    rather than collapsing it to a per-domain scalar mean. The perturbed field is
    ``background + transferred(delta_eps)``; the background field is a *uniform*
    ``background`` of the same length, so the background solve stays
    rho-independent and contributes an exact zero design gradient.

    Args:
        delta_eps: Nodal permittivity perturbation on the shared mesh.
        background_epsilon: Background silicon permittivity (constant).
        design_transfer: Dense ``(n_design_cells, n_nodes)`` mesh-transfer matrix
            -- ``build_mesh_transfer_operator(...).dense()`` from
            :mod:`prismo.mesh_transfer`. When ``None`` the transfer is the
            identity, so the design cells are the shared-mesh nodes themselves.

    Returns:
        ``(epsilon_bg, epsilon_pert)`` fields over the design cells.
    """
    if design_transfer is None:
        transferred = delta_eps
    else:
        transfer = jnp.asarray(design_transfer, dtype=delta_eps.dtype)
        transferred = transfer @ delta_eps

    epsilon_bg = jnp.full(transferred.shape, background_epsilon, dtype=delta_eps.dtype)
    epsilon_pert = epsilon_bg + transferred
    return epsilon_bg, epsilon_pert


# -- Components bundle -----------------------------------------------------------


@dataclass(frozen=True)
class PipelineComponents:
    """The differentiable components one ``pipeline()`` call composes.

    Carries all backend state -- container handles or local api modules and
    the per-lifecycle background eigenmode cache -- so ``pipeline()`` reads no
    module globals. The container lifecycle builds one on startup and releases
    it via ``close()``.

    ``design_cell_centroids`` is the static geometry query bound to the same
    gyptis backend as the solve components: a zero-arg reader returning the
    ``(n_design, 2)`` centroids in ``design_epsilon`` order, so the pipeline
    setup can build the mesh-transfer operator (ticket 08) through the seam
    without reaching for a raw container handle. ``None`` when no gyptis
    backend is bound.
    """

    chargetransport: Callable[..., Any]
    gyptis: Callable[..., Any]
    gyptis_background: Callable[..., Any]
    design_cell_centroids: Callable[[], np.ndarray] | None = None
    design_cell_vertices: Callable[[], np.ndarray] | None = None
    closers: tuple[Callable[[], None], ...] = field(default=())

    def close(self) -> None:
        """Release owned resources (containers, worker processes)."""
        _close_all(self.closers)

    def collect_phase_timing(self) -> dict[str, dict[str, float | int]]:
        """Merge each component's per-callback phase timings and reset them.

        Reads the ``PhaseTimer`` each timed component owns rather than a global
        begin/finish protocol. Components sharing a timer (the gyptis pair) are
        collected once; components without one (e.g. test fakes) contribute
        nothing.
        """
        merged: dict[str, dict[str, float | int]] = {}
        seen_timers: set[int] = set()
        for component in (
            self.chargetransport,
            self.gyptis,
            self.gyptis_background,
        ):
            timer = getattr(component, "timer", None)
            if timer is None or id(timer) in seen_timers:
                continue
            seen_timers.add(id(timer))
            merged.update(timer.collect())
        return merged


def build_default_components() -> PipelineComponents:
    """Build the default in-process components from the local tesseract apis.

    Loads each component's ``tesseract_api`` module if importable. There is no
    physics-free stub: a component whose tesseract_api has no live solver (no
    Julia, no gyptis/FEniCS) raises when called rather than fabricating carriers
    or an effective index. Used when ``pipeline()`` is called without an explicit
    bundle (no containers running).
    """
    ct_api = _load_tesseract_api("chargetransport")
    gyptis_api = _load_tesseract_api("gyptis")
    chargetransport = build_chargetransport_component(local_api=ct_api)
    gyptis, gyptis_background = build_gyptis_components(local_api=gyptis_api)

    closers: list[Callable[[], None]] = []
    shutdown_worker = getattr(ct_api, "shutdown", None)
    if callable(shutdown_worker):
        closers.append(shutdown_worker)
    design_cell_centroids = (
        partial(read_gyptis_design_cell_centroids, local_api=gyptis_api)
        if gyptis_api is not None
        else None
    )
    design_cell_vertices = (
        partial(read_gyptis_design_cell_vertices, local_api=gyptis_api)
        if gyptis_api is not None
        else None
    )
    return PipelineComponents(
        chargetransport=chargetransport,
        gyptis=gyptis,
        gyptis_background=gyptis_background,
        design_cell_centroids=design_cell_centroids,
        design_cell_vertices=design_cell_vertices,
        closers=tuple(closers),
    )


_DEFAULT_COMPONENTS: PipelineComponents = build_default_components()


def default_components() -> PipelineComponents:
    """The shared in-process components ``pipeline()`` uses without a bundle."""
    return _DEFAULT_COMPONENTS


# 1 cm^-3 = 1e6 m^-3: ChargeTransport output -> Soref-Bennett input.
_CM3_TO_M3 = 1e6

# Fixed reverse-bias operating point and free-space wavelength for the objective
# and the VπLπ modulation-efficiency headline (ticket 03). Δneff is evaluated
# between 0 V and REVERSE_BIAS_V; λ matches the gyptis solver's 1.55 µm C-band
# point (WAVELENGTH in the gyptis tesseract_api).
REVERSE_BIAS_V = -5.0
_WAVELENGTH_CM = 1.55e-4  # 1.55 µm in cm


def vpi_lpi_v_cm(delta_neff: float | jax.Array) -> float:
    """Half-wave voltage-length product VπLπ [V·cm] from signed Δneff.

    A carrier-depletion phase modulator accrues ``φ = (2π/λ)·Δneff·L``, so the
    length for a π shift at the fixed reverse bias is ``Lπ = λ/(2·Δneff)`` and
    ``VπLπ = |V_bias|·λ/(2·Δneff)``. This is the field-standard modulation-
    efficiency headline (smaller ``|VπLπ|`` is better); it is *reported*, not
    optimized -- signed Δneff is the optimized proxy. VπLπ carries Δneff's sign
    (a wrong-polarity optimum reads negative) and diverges as Δneff → 0.
    """
    dneff = float(delta_neff)
    if dneff == 0.0:
        return float("inf")
    return abs(REVERSE_BIAS_V) * _WAVELENGTH_CM / (2.0 * dneff)


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
    theta: jax.Array,
    H: jax.Array | None = None,
    H_sum: jax.Array | None = None,
    mesh_ref: MeshRef | None = None,
    background_epsilon: float | None = None,
    design_transfer: jax.Array | None = None,
    components: PipelineComponents | None = None,
) -> jax.Array:
    """Signed design field theta -> Delta n_eff differentiable pipeline.

    Args:
        theta: Signed design field per node in [-1, 1], shape ``(n_nodes,)``.
            Its sign is the free P/N polarity (junction = zero-crossing).
        H: Dense filter matrix, shape ``(n_nodes, n_nodes)``. Skip filter if
            ``None``.
        H_sum: Pre-computed row sums of ``H``.
        mesh_ref: ``MeshRef`` forwarded to ChargeTransport calls.
        background_epsilon: Background Si relative permittivity
            (default: ``n_si^2 = 3.4757^2``).
        design_transfer: Dense ``(n_design_cells, n_nodes)`` mesh-transfer matrix
            carrying the nodal perturbation onto the gyptis design cells (ticket
            04). ``None`` maps the perturbation node-for-node (identity).
        components: Live differentiable components to compose. Defaults to the
            in-process components built from the local tesseract apis.

    Returns:
        Signed effective-index shift ``Δneff = Re[neff(-5V)] - Re[neff(0)]``.
        Smooth and differentiable through zero; depletion (the physical bias
        response) makes it positive. Report ``VπLπ`` from it via
        :func:`vpi_lpi_v_cm`.
    """
    if background_epsilon is None:
        background_epsilon = _DEFAULT_BACKGROUND_EPSILON
    if components is None:
        components = _DEFAULT_COMPONENTS

    # 1. Density filter: theta -> theta_tilde (linear; regularizes junction width)
    if H is not None:
        if H_sum is None:
            H_sum = jnp.sum(H, axis=1)
        theta_tilde = _filter_jax(theta, H, H_sum)
    else:
        theta_tilde = theta

    # 2. Signed doping mapping: theta_tilde -> N(theta) [cm^-3]
    doping = doping_from_theta(theta_tilde)

    # 3. ChargeTransport at equilibrium (0 V)
    n0, p0 = components.chargetransport(doping, 0.0, mesh_ref)

    # 4. ChargeTransport at reverse bias (-5 V)
    n1, p1 = components.chargetransport(doping, REVERSE_BIAS_V, mesh_ref)

    # CT reports carrier densities in cm^-3 (same unit system as the doping
    # input); Soref-Bennett consumes m^-3 per CarrierDensityField. Convert
    # at the component boundary (ticket 17).
    n0 = n0 * _CM3_TO_M3
    p0 = p0 * _CM3_TO_M3
    n1 = n1 * _CM3_TO_M3
    p1 = p1 * _CM3_TO_M3

    # 5. Soref-Bennett coupling (equilibrium-subtracted)
    delta_eps, _ = _sb_jax(n1, p1, n0, p0)

    # 6. gyptis eigenmode solves on the design-region permittivity field. The
    # perturbation keeps its full spatial structure (no mean-collapse): a
    # topology change that redistributes carriers at fixed mean now moves neff.
    bg = jnp.asarray(background_epsilon, dtype=delta_eps.dtype)
    epsilon_bg, epsilon_pert = _build_design_epsilon(delta_eps, bg, design_transfer)

    # Background epsilon does not depend on rho. Cache its eigenmode while
    # keeping the perturbed solve and eigen-adjoint live for every rho.
    neff_sq_0 = components.gyptis_background(epsilon_bg, background_epsilon)
    neff_sq_1 = components.gyptis(epsilon_pert, background_epsilon)

    neff_0 = jnp.sqrt(jnp.maximum(neff_sq_0, 0.0))
    neff_1 = jnp.sqrt(jnp.maximum(neff_sq_1, 0.0))

    # Signed Δneff = Re[neff(-5V)] - Re[neff(0)]: the honest objective. Depletion
    # raises the index, so a physical optimum is positive; the sign is physically
    # determined, so the optimizer maximizes Δneff directly -- no epsilon fudge,
    # no magnitude folding that would reward a wrong-polarity mode shift.
    return neff_1 - neff_0
