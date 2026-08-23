"""End-to-end differentiable pipeline: signed design field theta -> objective.

Composes density filter -> doping mapping -> ChargeTransport.jl (0V, -5V)
-> Soref-Bennett coupling -> gyptis -> delta_n_eff into a single
JAX-differentiable function, optionally minus a weighted first-order modal
free-carrier loss (ticket 25, ADR 0004).

Ref: tickets 14 (pipeline composition), 15 (optimization loop), 25 (loss).
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from threading import RLock
from typing import Any, NamedTuple

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
#
# The reference is the *depletion-modulator* concentration, not a near-intrinsic
# one: at |N| ~ 3e15 the -5 V depletion width (~1.6 um) swamps the 500 nm x 220 nm
# rib, so the rib is fully depleted and the reverse-bias carrier field carries no
# bulk-doping signal at all (ticket 06). Centring the span on 1e17 keeps the seeded
# junction partially depleted -- the regime a real carrier-depletion modulator works
# in -- while the |theta|=1 rail still lands at ~1e19.
DOPING_REFERENCE_CM3 = 1e17
DOPING_LOG10_SPAN = 2.0


@jax.custom_jvp
def doping_from_theta(theta: jax.Array) -> jax.Array:
    """Map the signed design field ``theta`` to signed net doping in ``cm^-3``.

    ``N(theta) = sign(theta) * 1e17 * (10^(span*|theta|) - 1)``. The map is
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
    """Analytic derivative ``dN/dtheta = 1e17 * ln10 * span * 10^(span*|theta|)``.

    The map is C^1 through the junction: the derivative is even and continuous,
    with the true slope ``1e17 * ln10 * span`` at ``theta=0``. Autodiff of the
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


# Initial junction magnitude for the signed design field: |N| ~ 3e17 cm^-3 --
# the partially-depleted depletion-modulator operating point, non-degenerate,
# reverse-bias convergent, and well inside the [-1, 1] bounds.
_JUNCTION_SEED_THETA = 0.3


@dataclass(frozen=True)
class DesignNodes:
    """Which shared-mesh nodes carry a design variable, and how to place them back.

    The design field is defined on the *silicon* nodes (``slab`` +
    ``rib_silicon``) rather than on every node of the shared mesh. Only silicon
    nodes have physics attached to them: ChargeTransport gathers doping on the
    silicon subgrid and scatters carriers back from it, and every gyptis design
    cell is a rib triangle whose three vertices are silicon nodes. A variable on
    an oxide, substrate, clad or PML node therefore has an exactly-zero gradient
    unless the density filter happens to reach a silicon node, in which case it
    dopes silicon from outside the device -- a degree of freedom with no
    physical referent either way.

    Restricting the design set drops those variables from the MMA problem and
    shrinks the dense filter matrix quadratically (it is ``(n_design,
    n_design)``). Downstream contracts are unchanged: :meth:`scatter` places the
    filtered field back into full gmsh node order before the doping map, so
    ``mesh_ref``'s node ordering and the ``(n_design_cells, n_nodes)`` transfer
    still see a full-length nodal field. Non-design nodes scatter to ``theta =
    0``, i.e. net-intrinsic, which is what the oxide already meant.

    Attributes:
        indices: ``(n_design,)`` node indices into full gmsh node order.
        n_mesh_nodes: Number of nodes in the full shared mesh.
    """

    indices: np.ndarray
    n_mesh_nodes: int

    def __len__(self) -> int:
        """Number of design variables."""
        return int(self.indices.size)

    @classmethod
    def all_nodes(cls, n_mesh_nodes: int) -> DesignNodes:
        """Every mesh node is a design node (the no-silicon-groups fallback)."""
        return cls(
            indices=np.arange(n_mesh_nodes, dtype=np.intp), n_mesh_nodes=n_mesh_nodes
        )

    def scatter(self, design_field: jax.Array) -> jax.Array:
        """Place a design-node field into full node order, zero elsewhere."""
        full = jnp.zeros(self.n_mesh_nodes, dtype=design_field.dtype)
        return full.at[jnp.asarray(self.indices)].set(design_field)

    def scatter_numpy(self, design_field: np.ndarray) -> np.ndarray:
        """:meth:`scatter` for plotting and other non-traced callers."""
        full = np.zeros(self.n_mesh_nodes, dtype=float)
        full[self.indices] = np.asarray(design_field, dtype=float)
        return full


def seed_signed_junction(coords: np.ndarray) -> jax.Array:
    """Seed a signed lateral P/N junction across the mesh in every run path.

    Nodes left of the median x seed n-type (``+0.3``), nodes to the right seed
    p-type (``-0.3``). With the cathode on the right at -5 V, this is the
    reverse-bias orientation. sign(theta) remains free, so the optimizer can
    move, dissolve, or reverse this junction.
    """
    midpoint = np.median(coords[:, 0])
    return jnp.where(
        coords[:, 0] <= midpoint, _JUNCTION_SEED_THETA, -_JUNCTION_SEED_THETA
    )


# Junction seeds (ticket 25). The MMA optimum is local, so the run can start
# from more than the lateral junction: a vertical (P over N) and a U-shaped
# (N wrapped under and beside a P core) topology, the 2D cross-sections of the
# literature's higher-perimeter junctions. Every seed keeps n-type on the left
# slab edge (anode) and p-type on the right slab edge (cathode at -5 V), so
# both carrier populations reach their contact and the seed is reverse-biased.
SEED_KINDS: tuple[str, ...] = ("lateral", "vertical", "u")
# Width of the p-type column kept along the rib's right wall (vertical and U
# seeds) and of the n-type wall along the rib's left wall (U seed), as a
# fraction of the rib width: the p core must reach the p slab through the wall.
_SEED_WALL_FRACTION = 0.25


def _seed_rib_box(coords: np.ndarray) -> tuple[float, float, float, float, float]:
    """``(x_left, x_right, y_slab_top, y_top, x_centre)`` of the rib from the design nodes.

    The design nodes span slab + rib; the rib's nodes are those above the
    vertical midpoint of the whole set (the slab is thinner than the rib), and
    the slab top is the highest node outside the rib's x-span.
    """
    x, y = coords[:, 0], coords[:, 1]
    y_mid = 0.5 * (float(y.min()) + float(y.max()))
    in_rib = y > y_mid
    if not np.any(in_rib):
        # Flat node set (no rib): treat the whole set as the rib box.
        in_rib = np.ones_like(in_rib, dtype=bool)
    x_left, x_right = float(x[in_rib].min()), float(x[in_rib].max())
    y_top = float(y.max())
    outside = (x < x_left) | (x > x_right)
    y_slab_top = float(y[outside].max()) if np.any(outside) else float(y.min())
    return x_left, x_right, y_slab_top, y_top, float(np.median(x))


def seed_design_field(coords: np.ndarray, kind: str = "lateral") -> jax.Array:
    """Seed one of :data:`SEED_KINDS` on the design nodes at ``|theta| = 0.3``.

    ``lateral``: :func:`seed_signed_junction` -- n-type left of the median x,
    p-type right of it. ``vertical``: in the rib, n-type below the rib's
    mid-height and p-type above it; a p column along the rib's right wall
    joins the p top to the right slab, and the slab itself is lateral so both
    regions reach their contact. ``u``: n-type wraps under (lower third of the
    rib) and beside (left wall) a p core that, with the right wall and the
    right slab, is p-type. All three are reverse-biased with the cathode on
    the right.
    """
    if kind not in SEED_KINDS:
        raise ValueError(f"unknown seed {kind!r}; expected one of {SEED_KINDS}")
    if kind == "lateral":
        return seed_signed_junction(coords)

    x, y = coords[:, 0], coords[:, 1]
    x_left, x_right, y_slab_top, y_top, _x_centre = _seed_rib_box(coords)
    rib_width = x_right - x_left
    rib_height = y_top - y_slab_top
    wall = _SEED_WALL_FRACTION * rib_width
    p_column = x > x_right - wall  # p column on the rib's right wall + right slab

    if kind == "vertical":
        n_type = (y <= y_slab_top + 0.5 * rib_height) & ~p_column
    else:  # "u"
        floor = y <= y_slab_top + rib_height / 3.0
        left_wall = x <= x_left + wall  # rib left wall + left slab
        n_type = (floor & ~p_column) | left_wall
    return jnp.where(n_type, _JUNCTION_SEED_THETA, -_JUNCTION_SEED_THETA)


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


DEFAULT_BACKGROUND_EPSILON: float = 3.4757**2
_M3_TO_CM3: float = 1e-6
_DEFAULT_COEFFS: SorefBennettCoefficients = SorefBennettCoefficients()


def _close_all(
    closers: tuple[Callable[[], None], ...] | list[Callable[[], None]],
) -> None:
    """Run resource closers in reverse (last-acquired first)."""
    for close in reversed(closers):
        close()


def init_tesseract_containers(
    mesh_dir: str | Path | None = None,
    mesh_size: float | None = None,
    mode_index: int = 0,
    contact_offset: float | None = None,
    domain_width: float | None = None,
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
        mesh_size: Characteristic element size of the silicon (rib + slab) in
            micrometres, passed to the gyptis container as
            ``PRISMO_GYPTIS_MESH_SIZE``. gyptis authors the shared mesh, so this
            one knob sets the resolution both solvers see. ``None`` keeps the
            component's own default (0.04 µm).
        mode_index: Guided mode the gyptis solves target and track -- ``0`` the
            fundamental, ``k`` the ``k``-th guided mode in descending neff (see
            :func:`build_gyptis_components`).
        contact_offset: Gap from the rib edge to the near contact edge [µm],
            passed to the gyptis mesh author as ``PRISMO_GYPTIS_CONTACT_OFFSET``
            (ticket 25). ``None`` keeps the component's default (0.2 µm).
        domain_width: Physical box width [µm] (the slab spans it; the PML lies
            outside), passed as ``PRISMO_GYPTIS_WIDTH``. ``None`` keeps the
            default (2.0 µm).
    """
    try:
        from tesseract_core import Tesseract  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "tesseract_core is required for container pipeline runs"
        ) from exc

    ct_volumes: list[str] = []
    if mesh_dir is not None:
        host_mesh_dir = Path(mesh_dir).resolve()
        host_mesh_dir.mkdir(parents=True, exist_ok=True)
        ct_volumes.append(f"{host_mesh_dir}:{_CT_MESH_MOUNT}:ro")

    # Dev loop: bind-mount host Julia scripts over the image's baked
    # ``/tesseract/scripts`` so a solver/adjoint edit can be validated without a
    # ~15 min image rebuild. The sysimage still supplies the precompiled heavy
    # packages; ``worker.jl`` is ``include``d at process start, so the mounted
    # source takes effect (recompiling only the edited methods).
    ct_scripts_dir = os.environ.get("PRISMO_CT_SCRIPTS_DIR")
    if ct_scripts_dir:
        ct_volumes.append(f"{Path(ct_scripts_dir).resolve()}:/tesseract/scripts:ro")

    # The Julia-side solve budget (``PRISMO_CT_SOLVE_BUDGET_S``, checked inside
    # the continuation loops) and the Python request-timeout backstop
    # (``PRISMO_CT_JULIA_TIMEOUT_S``) both live in the ChargeTransport
    # container; forward the host's overrides so a refinement study can stretch
    # them without an image rebuild.
    ct_env = {
        name: os.environ[name]
        for name in ("PRISMO_CT_JULIA_TIMEOUT_S", "PRISMO_CT_SOLVE_BUDGET_S")
        if name in os.environ
    }

    # Dev loop (ticket 21): ``PRISMO_DEV_MOUNTS=1`` bind-mounts the host
    # ``tesseract_api.py`` and ``prismo_shared`` over both images so a Python
    # component edit costs a container restart, not a 4-5 GB image rebuild.
    # See docs/agents/debugging.md.
    dev_mounts = dev_mounts_requested()
    gyptis_volumes: list[str] = []
    gyptis_env: dict[str, str] = {}
    if dev_mounts:
        ct_dev_volumes, dev_env = _dev_mount_volumes("chargetransport")
        gyptis_volumes, _ = _dev_mount_volumes("gyptis")
        ct_volumes.extend(ct_dev_volumes)
        ct_env.update(dev_env)
        gyptis_env.update(dev_env)
        print(
            "      PRISMO_DEV_MOUNTS=1: running the HOST tesseract_api.py and "
            "prismo_shared in both containers (image copies are shadowed)",
            flush=True,
        )

    closers: list[Callable[[], None]] = []
    try:
        ct_tesseract = Tesseract.from_image(
            "prismo_chargetransport:latest",
            volumes=ct_volumes or None,
            environment=ct_env or None,
        )
        ct_tesseract.serve()
        closers.append(ct_tesseract.teardown)
    except Exception as exc:
        _close_all(closers)
        raise RuntimeError("Failed to start ChargeTransport container") from exc

    if mesh_size is not None:
        gyptis_env["PRISMO_GYPTIS_MESH_SIZE"] = repr(float(mesh_size))
    if contact_offset is not None:
        gyptis_env["PRISMO_GYPTIS_CONTACT_OFFSET"] = repr(float(contact_offset))
    if domain_width is not None:
        gyptis_env["PRISMO_GYPTIS_WIDTH"] = repr(float(domain_width))
    try:
        gyptis_tesseract = Tesseract.from_image(
            "prismo_gyptis:latest",
            volumes=gyptis_volumes or None,
            environment=gyptis_env or None,
        )
        gyptis_tesseract.serve()
        closers.append(gyptis_tesseract.teardown)
    except Exception as exc:
        _close_all(closers)
        raise RuntimeError("Failed to start gyptis container") from exc

    chargetransport = build_chargetransport_component(container=ct_tesseract)
    gyptis, gyptis_background = build_gyptis_components(
        container=gyptis_tesseract, mode_index=mode_index
    )
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
        write_mesh=partial(write_gyptis_mesh, container=gyptis_tesseract),
        mode_field=partial(
            read_gyptis_mode_field, container=gyptis_tesseract, mode_index=mode_index
        ),
        reset_chargetransport=partial(
            reset_chargetransport_worker, container=ct_tesseract
        ),
        closers=tuple(closers),
    )


def teardown_containers(components: PipelineComponents) -> None:
    """Stop and remove the running tesseract containers owned by ``components``."""
    components.close()


# -- Dev mounts (ticket 21) ------------------------------------------------------

# In-container paths the dev mounts shadow. ``tesseract_api.py`` sits at the
# image's fixed API path; ``prismo_shared`` cannot be mounted over its
# ``site-packages`` copy (the two images run different Python versions, so the
# path differs) and is instead mounted under a dev root that ``PYTHONPATH``
# puts ahead of ``site-packages``.
_DEV_API_MOUNT = "/tesseract/tesseract_api.py"
_DEV_PYTHONPATH_ROOT = "/prismo_dev"
_SHARED_CODE_DIR = _COMPONENTS_DIR.parent / "shared_code"


def dev_mounts_requested() -> bool:
    """``PRISMO_DEV_MOUNTS`` is set to a truthy value (``1``/``true``/``yes``)."""
    return os.environ.get("PRISMO_DEV_MOUNTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _dev_mount_volumes(component: str) -> tuple[list[str], dict[str, str]]:
    """Bind mounts + env that run the host ``tesseract_api.py``/``prismo_shared``.

    Returns ``(volumes, environment)`` for one component's container. The dev
    ``PYTHONPATH`` is merged into the caller's env rather than replacing it.
    """
    api_path = (_COMPONENTS_DIR / component / "tesseract_api.py").resolve()
    shared_path = (_SHARED_CODE_DIR / "prismo_shared").resolve()
    if not api_path.is_file():
        raise RuntimeError(f"PRISMO_DEV_MOUNTS: {api_path} does not exist")
    if not shared_path.is_dir():
        raise RuntimeError(f"PRISMO_DEV_MOUNTS: {shared_path} does not exist")
    volumes = [
        f"{api_path}:{_DEV_API_MOUNT}:ro",
        f"{shared_path}:{_DEV_PYTHONPATH_ROOT}/prismo_shared:ro",
    ]
    return volumes, {"PYTHONPATH": _DEV_PYTHONPATH_ROOT}


def reset_chargetransport_worker(
    *, container: Any | None = None, local_api: Any | None = None
) -> None:
    """Drop the ChargeTransport worker's warm solutions (ticket 20).

    The next solve is then a function of the doping alone -- the cold
    continuation from near-intrinsic equilibrium and the bias ramp -- rather
    than of the Newton starting points the previous designs left behind.
    Carried as the ``reset`` operation of the component's ``apply`` endpoint.
    """

    def from_container(tess: Any) -> None:
        tess.apply({"operation": "reset"})

    def from_local(api: Any) -> None:
        api.apply(api.InputSchema(operation="reset"))

    invoke_tesseract(
        container,
        local_api,
        container_call=from_container,
        local_call=from_local,
    )


def _gyptis_query(
    payload: dict[str, Any],
    outputs: tuple[str, ...],
    *,
    container: Any | None,
    local_api: Any | None,
) -> tuple[Any, ...]:
    """Run one read-only gyptis ``apply()`` against whichever backend is bound.

    The Tesseract API's fixed endpoint set carries these static geometry and
    field queries as ``operation`` values on ``apply``. Returns the named
    outputs in order, ``None`` for any the backend omitted, so each caller
    validates the payload it needs. A live gyptis/FEniCS backend is required:
    unlike a solve, there is no meaningful local stub for its mesh-dependent
    cells.
    """

    def from_container(tess: Any) -> tuple[Any, ...]:
        result = tess.apply(payload)
        return tuple(result.get(name) for name in outputs)

    def from_local(api: Any) -> tuple[Any, ...]:
        result = api.apply(api.InputSchema(**payload))
        return tuple(getattr(result, name, None) for name in outputs)

    return invoke_tesseract(
        container,
        local_api,
        container_call=from_container,
        local_call=from_local,
    )


def _as_design_cell_vertices(raw: Any) -> np.ndarray:
    """Validate a design-cell vertex payload as ``(n_design, 3, 2)`` floats."""
    vertices = np.asarray(raw, dtype=float)
    if vertices.ndim != 3 or vertices.shape[1:] != (3, 2):
        raise ValueError("gyptis design-cell vertices must have shape (n_design, 3, 2)")
    return vertices


def read_gyptis_design_cell_centroids(
    *, container: Any | None = None, local_api: Any | None = None
) -> np.ndarray:
    """Read gyptis design-cell centroids in ``design_epsilon`` field order."""
    (raw,) = _gyptis_query(
        {"operation": "design_cell_centroids"},
        ("design_cell_centroids",),
        container=container,
        local_api=local_api,
    )
    if raw is None:
        raise RuntimeError("gyptis design_cell_centroids returned no centroids")
    centroids = np.asarray(raw, dtype=float)
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
    05). Exposed through the ``write_mesh`` operation.
    """
    (raw,) = _gyptis_query(
        {"operation": "write_mesh"},
        ("design_cell_vertices",),
        container=container,
        local_api=local_api,
    )
    if raw is None:
        raise RuntimeError("gyptis write_mesh returned no design-cell vertices")
    return _as_design_cell_vertices(raw)


def _gyptis_mode_payload(mode_index: int) -> dict[str, int]:
    """The ``mode_index`` entry of a gyptis payload, omitted for the fundamental.

    The fundamental is the component's default, so leaving the key out keeps
    a default run's payload identical to what an image predating the field
    accepts; only a higher-order target sends it (and needs the field).
    """
    index = int(mode_index)
    if index < 0:
        raise ValueError("mode_index must be a non-negative integer")
    return {"mode_index": index} if index else {}


def read_gyptis_mode_field(
    design_epsilon: np.ndarray,
    core_epsilon: float = DEFAULT_BACKGROUND_EPSILON,
    *,
    container: Any | None = None,
    local_api: Any | None = None,
    mode_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the tracked mode's ``|E|`` profile at the gyptis mesh vertices.

    Solves once at ``design_epsilon`` and returns ``(abs_e, coords)`` -- the
    peak-normalized magnitude per vertex and the matching ``(n_vertices, 2)``
    vertex coordinates in micrometres. This is the headline optical-mode figure's
    data source (ticket 07); it is a read-only query, so it does not advance the
    component's tracked mode branch.

    ``core_epsilon`` must be the same background the solve components were given:
    it is part of the geometry key the component tracks the mode branch by, so a
    mismatched value would both solve a different device and miss the tracked
    branch, falling back to the mode selection ticket 13 replaced. Likewise
    ``mode_index`` must match the solve components' target: it is part of the
    same branch key, so the figure shows the mode that was optimized.
    """
    abs_e_raw, coords_raw = _gyptis_query(
        {
            "operation": "mode_field",
            "design_epsilon": np.asarray(design_epsilon, dtype=float),
            "core_epsilon": float(core_epsilon),
            **_gyptis_mode_payload(mode_index),
        },
        ("mode_abs_e", "mode_coordinates"),
        container=container,
        local_api=local_api,
    )
    if abs_e_raw is None or coords_raw is None:
        raise RuntimeError("gyptis mode_field returned no field payload")
    abs_e = np.asarray(abs_e_raw, dtype=float)
    coords = np.asarray(coords_raw, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("gyptis mode coordinates must have shape (n_vertices, 2)")
    if abs_e.ndim != 1 or abs_e.shape[0] != coords.shape[0]:
        raise ValueError(
            "gyptis mode field must carry one magnitude per mesh vertex "
            f"({abs_e.shape} values for {coords.shape[0]} vertices)"
        )
    return abs_e, coords


def write_gyptis_mesh(
    mesh_path: str | Path, *, container: Any | None = None, local_api: Any | None = None
) -> np.ndarray:
    """Persist gyptis' unified mesh on host and return its design-cell vertices."""
    mesh_text, vertices_raw = _gyptis_query(
        {"operation": "write_mesh"},
        ("mesh_text", "design_cell_vertices"),
        container=container,
        local_api=local_api,
    )
    if mesh_text is None or vertices_raw is None:
        raise RuntimeError("gyptis write_mesh returned no mesh payload")
    if not str(mesh_text).startswith("$MeshFormat"):
        raise RuntimeError("gyptis write_mesh returned invalid Gmsh text")
    vertices = _as_design_cell_vertices(vertices_raw)
    path = Path(mesh_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(mesh_text))
    return vertices


def build_design_transfer(
    components: PipelineComponents,
    node_coords: np.ndarray,
    design_cell_vertices: np.ndarray | None = None,
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

    if design_cell_vertices is None and components.design_cell_vertices is None:
        raise RuntimeError(
            "Container pipeline requires a gyptis backend exposing design-cell "
            "vertices to build the mesh-transfer operator"
        )
    vertices = (
        components.design_cell_vertices()
        if design_cell_vertices is None
        else np.asarray(design_cell_vertices, dtype=float)
    )
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
) -> DifferentiableComponent:
    """Build the ChargeTransport component bound to one backend.

    ``container`` is a running Tesseract handle; ``local_api`` an in-process
    ``tesseract_api`` module. With neither, calling the component raises -- there
    is no physics-free identity fallback. The backend is captured here, not read
    from module globals.
    """

    def forward(
        doping_np: np.ndarray,
        bias_voltage: float,
        mesh_ref: MeshRef | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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

        return invoke_tesseract(
            container,
            local_api,
            container_call=from_container,
            local_call=from_local,
        )

    def vjp(
        doping_np: np.ndarray,
        cotangent: tuple[np.ndarray, np.ndarray],
        bias_voltage: float,
        mesh_ref: MeshRef | None = None,
    ) -> np.ndarray:
        cot_n, cot_p = cotangent

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

        return invoke_tesseract(
            container,
            local_api,
            container_call=from_container,
            local_call=from_local,
        )

    return DifferentiableComponent(
        forward=forward,
        vjp=vjp,
        out_struct=_ct_out_struct,
    )


# -- gyptis component ------------------------------------------------------------


# The gyptis field-epsilon component takes the design-region permittivity field
# as its differentiated input and the constant core background as a static arg
# (so non-design core cells match the pipeline's background_epsilon).


def _gyptis_out_struct(
    design_epsilon: jax.Array, core_epsilon: float = DEFAULT_BACKGROUND_EPSILON
) -> jax.ShapeDtypeStruct:
    return _scalar_like(design_epsilon)


def _gyptis_background_vjp_impl(
    design_epsilon_np: np.ndarray,
    cot_neff_sq: np.ndarray,
    core_epsilon: float = DEFAULT_BACKGROUND_EPSILON,
) -> np.ndarray:
    """Background permittivity is rho-independent: its cotangent is zero."""
    return np.zeros_like(design_epsilon_np)


def build_gyptis_components(
    container: Any | None = None,
    local_api: Any | None = None,
    mode_index: int = 0,
) -> tuple[DifferentiableComponent, DifferentiableComponent]:
    """Build the perturbed and background gyptis components for one backend.

    Both share a background eigenmode cache owned by this call, so the
    rho-independent background solve runs once per component lifecycle. The
    backend is captured here, not read from module globals.

    ``mode_index`` selects the guided mode every solve of this bundle targets:
    ``0`` the fundamental (largest neff in the guided window), ``k`` the
    ``k``-th guided mode in descending neff. The component ranks the modes on
    the first solve of a geometry and tracks the chosen branch by nearest
    eigenvalue thereafter, so Δneff is the shift of *that* mode throughout an
    optimization. The index is baked into the bundle rather than threaded
    through ``pipeline()``: one run optimizes one mode.
    """
    background_cache: dict[
        tuple[tuple[int, ...], str, bytes, float, int], np.ndarray
    ] = {}
    cache_lock = RLock()
    mode_payload = _gyptis_mode_payload(mode_index)

    def forward(
        design_epsilon_np: np.ndarray,
        core_epsilon: float = DEFAULT_BACKGROUND_EPSILON,
    ) -> np.ndarray:
        out_dtype = design_epsilon_np.dtype

        def from_container(tess: Any) -> np.ndarray:
            result = tess.apply(
                {
                    "design_epsilon": design_epsilon_np.tolist(),
                    "core_epsilon": float(core_epsilon),
                    **mode_payload,
                }
            )
            return np.asarray(result["neff_sq"], dtype=out_dtype)

        def from_local(api: Any) -> np.ndarray:
            outputs = api.apply(
                api.InputSchema(
                    design_epsilon=design_epsilon_np,
                    core_epsilon=float(core_epsilon),
                    **mode_payload,
                )
            )
            return np.asarray(outputs.neff_sq, dtype=out_dtype)

        return invoke_tesseract(
            container,
            local_api,
            container_call=from_container,
            local_call=from_local,
        )

    def background_forward(
        design_epsilon_np: np.ndarray,
        core_epsilon: float = DEFAULT_BACKGROUND_EPSILON,
    ) -> np.ndarray:
        key = (
            design_epsilon_np.shape,
            design_epsilon_np.dtype.str,
            design_epsilon_np.tobytes(),
            float(core_epsilon),
            int(mode_index),
        )
        with cache_lock:
            cached = background_cache.get(key)
        if cached is not None:
            return cached.copy()

        result = forward(design_epsilon_np, core_epsilon)
        with cache_lock:
            background_cache[key] = result.copy()
        return result

    def vjp(
        design_epsilon_np: np.ndarray,
        cot_neff_sq: np.ndarray,
        core_epsilon: float = DEFAULT_BACKGROUND_EPSILON,
    ) -> np.ndarray:
        out_dtype = design_epsilon_np.dtype

        def from_container(tess: Any) -> np.ndarray:
            vjp_result = tess.vector_jacobian_product(
                {
                    "design_epsilon": design_epsilon_np.tolist(),
                    "core_epsilon": float(core_epsilon),
                    **mode_payload,
                },
                ["design_epsilon"],
                ["neff_sq"],
                {"neff_sq": float(cot_neff_sq)},
            )
            return np.asarray(vjp_result["design_epsilon"], dtype=out_dtype)

        def from_local(api: Any) -> np.ndarray:
            vjp_result = api.vector_jacobian_product(
                api.InputSchema(
                    design_epsilon=design_epsilon_np,
                    core_epsilon=float(core_epsilon),
                    **mode_payload,
                ),
                {"design_epsilon"},
                {"neff_sq"},
                {"neff_sq": np.asarray(cot_neff_sq)},
            )
            return np.asarray(vjp_result["design_epsilon"], dtype=out_dtype)

        return invoke_tesseract(
            container,
            local_api,
            container_call=from_container,
            local_call=from_local,
        )

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
    return perturbed, background


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
    without reaching for a raw container handle. ``mode_field`` is the matching
    read-only query for the tracked mode's ``|E|`` profile, used by the headline
    mode figure (ticket 07). Both are ``None`` when no gyptis backend is bound.
    """

    chargetransport: Callable[..., Any]
    gyptis: Callable[..., Any]
    gyptis_background: Callable[..., Any]
    design_cell_centroids: Callable[[], np.ndarray] | None = None
    design_cell_vertices: Callable[[], np.ndarray] | None = None
    write_mesh: Callable[[str | Path], np.ndarray] | None = None
    mode_field: (
        Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]] | None
    ) = None
    # Drops the ChargeTransport worker's warm solutions so the next solve is
    # cold (ticket 20). ``None`` when no ChargeTransport backend is bound.
    reset_chargetransport: Callable[[], None] | None = None
    closers: tuple[Callable[[], None], ...] = field(default=())

    def close(self) -> None:
        """Release owned resources (containers, worker processes)."""
        _close_all(self.closers)


def build_default_components(mode_index: int = 0) -> PipelineComponents:
    """Build the default in-process components from the local tesseract apis.

    Loads each component's ``tesseract_api`` module if importable. There is no
    physics-free stub: a component whose tesseract_api has no live solver (no
    Julia, no gyptis/FEniCS) raises when called rather than fabricating carriers
    or an effective index. Used when ``pipeline()`` is called without an explicit
    bundle (no containers running). ``mode_index`` is the guided mode the
    gyptis solves target (see :func:`build_gyptis_components`); the shared
    module-level bundle is built for the fundamental.
    """
    ct_api = _load_tesseract_api("chargetransport")
    gyptis_api = _load_tesseract_api("gyptis")
    chargetransport = build_chargetransport_component(local_api=ct_api)
    gyptis, gyptis_background = build_gyptis_components(
        local_api=gyptis_api, mode_index=mode_index
    )

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
        write_mesh=(
            partial(write_gyptis_mesh, local_api=gyptis_api)
            if gyptis_api is not None
            else None
        ),
        mode_field=(
            partial(read_gyptis_mode_field, local_api=gyptis_api, mode_index=mode_index)
            if gyptis_api is not None
            else None
        ),
        reset_chargetransport=(
            partial(reset_chargetransport_worker, local_api=ct_api)
            if ct_api is not None
            else None
        ),
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


# -- Free-carrier loss (ticket 25) ------------------------------------------------

# Power attenuation 1 cm^-1 (neper) = 10*log10(e) dB/cm.
NEPER_TO_DB: float = 10.0 / float(np.log(10.0))


def free_carrier_absorption_cm(
    electrons_cm3: jax.Array,
    holes_cm3: jax.Array,
    coeffs: SorefBennettCoefficients | None = None,
) -> jax.Array:
    """Absolute Soref-Bennett free-carrier absorption per node [cm^-1].

    ``alpha = C_e * N_e^D_e + C_h * N_h^D_h`` from the *absolute* carrier
    densities (cm^-3), not the equilibrium-subtracted perturbation the
    permittivity uses: the insertion loss of a doped waveguide is set by every
    carrier the mode sees. At 1e19 cm^-3 this is ~85 cm^-1 (electrons) and
    ~60 cm^-1 (holes), i.e. hundreds of dB/cm.
    """
    if coeffs is None:
        coeffs = _DEFAULT_COEFFS
    return coeffs.C_e * _signed_pow(electrons_cm3, coeffs.D_e) + coeffs.C_h * _signed_pow(
        holes_cm3, coeffs.D_h
    )


def modal_loss_db_cm(
    alpha_cells_cm: jax.Array,
    mode_overlap: jax.Array,
    neff_background: jax.Array | float,
    background_index: float = _DEFAULT_COEFFS.background_index,
) -> jax.Array:
    """Modal free-carrier loss [dB/cm] from the per-cell absorption, first order.

    With ``w_cell = d(neff^2)/d(eps_cell)`` the mode's sensitivity to the
    design-cell permittivity (the Hellmann-Feynman adjoint gyptis already
    computes), an imaginary permittivity ``Im(eps) = n_si*alpha*lambda/(2*pi)``
    in each cell shifts ``Im(neff^2)`` by ``sum(w*Im(eps))``, and the modal power
    loss ``2*k0*Im(neff)`` is ``(n_si/neff) * sum(w_cell * alpha_cell)``; the
    wavelength cancels. For a uniform core this is the textbook
    confinement-weighted loss ``Gamma * alpha * n_si/neff``. The weights are
    those of the *unperturbed* (background) mode, frozen: the carrier-induced
    permittivity shift is ~1e-3 and does not reshape the mode.
    """
    alpha_mode_cm = (
        background_index / neff_background
    ) * jnp.sum(mode_overlap * alpha_cells_cm)
    return NEPER_TO_DB * alpha_mode_cm


def loss_figure_of_merit_v_db(
    delta_neff: float | jax.Array, modal_loss_db_cm_value: float | jax.Array
) -> float:
    """``VπLπ x alpha`` [V·dB]: the literature's efficiency-loss figure of merit.

    Smaller is better; good depletion modulators reach ~10-30 V·dB. Like
    VπLπ it is *reported*, not optimized, and inherits VπLπ's sign.
    """
    return vpi_lpi_v_cm(delta_neff) * float(modal_loss_db_cm_value)


def read_mode_overlap(
    components: PipelineComponents,
    n_design_cells: int,
    background_epsilon: float = DEFAULT_BACKGROUND_EPSILON,
) -> np.ndarray:
    """The mode-overlap weights ``d(neff^2)/d(eps_cell)`` at the uniform background.

    One eigensolve plus one eigen-adjoint of the bound gyptis component on the
    rho-independent background permittivity. The result feeds
    ``pipeline(mode_overlap=...)`` as a constant: the loss term is first order
    in the carrier perturbation, so the weights are frozen at the unperturbed
    mode rather than re-derived (which would need second-order eigen-
    derivatives the adjoint does not provide).
    """
    eps_bg = jnp.full((int(n_design_cells),), float(background_epsilon))

    def neff_sq(eps: jax.Array) -> jax.Array:
        return jnp.reshape(components.gyptis(eps, float(background_epsilon)), ())

    weights = np.asarray(jax.grad(neff_sq)(eps_bg), dtype=float)
    if weights.shape != (int(n_design_cells),) or not np.all(np.isfinite(weights)):
        raise RuntimeError("gyptis returned no usable mode-overlap weights")
    return weights


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


class PipelineTerms(NamedTuple):
    """The two physical terms one pipeline evaluation yields (ticket 25).

    ``delta_neff`` is the signed effective-index shift the optimizer has always
    maximized; ``modal_loss_db_cm`` is the first-order modal free-carrier loss
    of the unbiased device (``nan`` when no mode-overlap weights were given).
    Both are JAX scalars and differentiable; ``pipeline()`` combines them.
    """

    delta_neff: jax.Array
    modal_loss_db_cm: jax.Array


def _pre_eigensolve(
    theta: jax.Array,
    H: jax.Array | None,
    H_sum: jax.Array | None,
    mesh_ref: MeshRef | None,
    background_epsilon: float,
    design_transfer: jax.Array | None,
    design_nodes: DesignNodes | None,
    components: PipelineComponents,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Filter -> doping -> both ChargeTransport solves -> Soref-Bennett -> transfer.

    Returns ``(epsilon_bg, epsilon_pert, n0_cm3, p0_cm3)``: the two design-cell
    permittivity fields the eigensolves consume and the equilibrium carriers
    (cm^-3, full node order) the loss term reads.
    """
    # 1. Density filter: theta -> theta_tilde (linear; regularizes junction width).
    # Filtering happens on the design nodes, so the filter never averages a
    # silicon node against an oxide variable that carries no physics.
    if H is not None:
        if H_sum is None:
            H_sum = jnp.sum(H, axis=1)
        theta_tilde = _filter_jax(theta, H, H_sum)
    else:
        theta_tilde = theta

    # 1b. Back to full gmsh node order for the solvers, which key off the shared
    # mesh's node set. Non-design nodes take theta = 0 (net-intrinsic): they are
    # oxide, and ChargeTransport reads doping on the silicon subgrid only.
    if design_nodes is not None:
        theta_tilde = design_nodes.scatter(theta_tilde)

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

    # 6. Design-region permittivity field. The perturbation keeps its full
    # spatial structure (no mean-collapse): a topology change that redistributes
    # carriers at fixed mean now moves neff.
    bg = jnp.asarray(background_epsilon, dtype=delta_eps.dtype)
    epsilon_bg, epsilon_pert = _build_design_epsilon(delta_eps, bg, design_transfer)
    return epsilon_bg, epsilon_pert, n0 * _M3_TO_CM3, p0 * _M3_TO_CM3


def carrier_fields(
    theta: jax.Array,
    H: jax.Array | None = None,
    H_sum: jax.Array | None = None,
    mesh_ref: MeshRef | None = None,
    design_nodes: DesignNodes | None = None,
    components: PipelineComponents | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The carrier densities behind one design: ``(n0, p0, n1, p1)`` [cm^-3].

    Filter -> doping -> ChargeTransport at 0 V and at the reverse bias, in
    full node order, for the headline depletion figure (swept carriers under
    the mode). Not differentiated; the same two solves the pipeline runs.
    """
    if components is None:
        components = _DEFAULT_COMPONENTS
    if H is not None:
        if H_sum is None:
            H_sum = jnp.sum(H, axis=1)
        theta_tilde = _filter_jax(theta, H, H_sum)
    else:
        theta_tilde = theta
    if design_nodes is not None:
        theta_tilde = design_nodes.scatter(theta_tilde)
    doping = doping_from_theta(theta_tilde)
    n0, p0 = components.chargetransport(doping, 0.0, mesh_ref)
    n1, p1 = components.chargetransport(doping, REVERSE_BIAS_V, mesh_ref)
    return n0, p0, n1, p1


def design_epsilon_from_theta(
    theta: jax.Array,
    H: jax.Array | None = None,
    H_sum: jax.Array | None = None,
    mesh_ref: MeshRef | None = None,
    background_epsilon: float | None = None,
    design_transfer: jax.Array | None = None,
    design_nodes: DesignNodes | None = None,
    components: PipelineComponents | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Everything ``pipeline()`` does up to the eigensolves.

    Runs filter -> signed doping -> both ChargeTransport solves -> Soref-Bennett
    -> mesh transfer, and returns the ``(epsilon_bg, epsilon_pert)`` design-cell
    permittivity fields the two gyptis solves consume. Split out of
    :func:`pipeline` so the same permittivity the objective was evaluated on can
    be handed to the mode-field query for the headline figure (ticket 07),
    rather than reconstructing the chain a second way.

    Arguments match :func:`pipeline`.
    """
    if background_epsilon is None:
        background_epsilon = DEFAULT_BACKGROUND_EPSILON
    if components is None:
        components = _DEFAULT_COMPONENTS
    epsilon_bg, epsilon_pert, _n0, _p0 = _pre_eigensolve(
        theta, H, H_sum, mesh_ref, background_epsilon, design_transfer,
        design_nodes, components,
    )
    return epsilon_bg, epsilon_pert


def pipeline_with_terms(
    theta: jax.Array,
    H: jax.Array | None = None,
    H_sum: jax.Array | None = None,
    mesh_ref: MeshRef | None = None,
    background_epsilon: float | None = None,
    design_transfer: jax.Array | None = None,
    design_nodes: DesignNodes | None = None,
    components: PipelineComponents | None = None,
    loss_weight: float = 0.0,
    mode_overlap: jax.Array | np.ndarray | None = None,
) -> tuple[jax.Array, PipelineTerms]:
    """:func:`pipeline` returning ``(objective, terms)`` for ``has_aux`` callers.

    The objective is ``delta_neff - loss_weight * modal_loss_db_cm`` (ticket
    25); with ``loss_weight == 0`` it is exactly ``delta_neff`` and the loss is
    only reported (``nan`` when no ``mode_overlap`` is bound). See
    :func:`pipeline` for the arguments.
    """
    loss_weight = float(loss_weight)
    if loss_weight < 0.0:
        raise ValueError("loss_weight must be non-negative")
    if loss_weight > 0.0 and mode_overlap is None:
        raise ValueError(
            "a positive loss_weight needs mode_overlap weights (read_mode_overlap)"
        )
    if background_epsilon is None:
        background_epsilon = DEFAULT_BACKGROUND_EPSILON
    if components is None:
        components = _DEFAULT_COMPONENTS

    epsilon_bg, epsilon_pert, n0_cm3, p0_cm3 = _pre_eigensolve(
        theta, H, H_sum, mesh_ref, background_epsilon, design_transfer,
        design_nodes, components,
    )

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
    delta_neff = neff_1 - neff_0

    if mode_overlap is None:
        loss = jnp.asarray(jnp.nan, dtype=delta_neff.dtype)
    else:
        weights = jnp.asarray(mode_overlap, dtype=delta_neff.dtype)
        if weights.shape != epsilon_bg.shape:
            raise ValueError(
                f"mode_overlap has shape {weights.shape}; expected one weight per "
                f"design cell {epsilon_bg.shape}"
            )
        # Loss of the unbiased device: the absolute 0 V carriers, carried onto
        # the design cells by the same transfer as the permittivity, summed
        # against the background mode's sensitivity weights.
        alpha_nodes = free_carrier_absorption_cm(n0_cm3, p0_cm3)
        if design_transfer is None:
            alpha_cells = alpha_nodes
        else:
            alpha_cells = jnp.asarray(design_transfer, dtype=alpha_nodes.dtype) @ alpha_nodes
        loss = modal_loss_db_cm(alpha_cells, weights, neff_0)

    objective = delta_neff if loss_weight == 0.0 else delta_neff - loss_weight * loss
    return objective, PipelineTerms(delta_neff=delta_neff, modal_loss_db_cm=loss)


def pipeline(
    theta: jax.Array,
    H: jax.Array | None = None,
    H_sum: jax.Array | None = None,
    mesh_ref: MeshRef | None = None,
    background_epsilon: float | None = None,
    design_transfer: jax.Array | None = None,
    design_nodes: DesignNodes | None = None,
    components: PipelineComponents | None = None,
    loss_weight: float = 0.0,
    mode_overlap: jax.Array | np.ndarray | None = None,
) -> jax.Array:
    """Signed design field theta -> objective (Δneff, optionally loss-penalized).

    Args:
        theta: Signed design field per design node in [-1, 1], shape
            ``(n_design,)`` -- the silicon nodes when ``design_nodes`` is given,
            otherwise every mesh node. Its sign is the free P/N polarity
            (junction = zero-crossing).
        H: Dense filter matrix, shape ``(n_design, n_design)``. Skip filter if
            ``None``.
        H_sum: Pre-computed row sums of ``H``.
        mesh_ref: ``MeshRef`` forwarded to ChargeTransport calls.
        background_epsilon: Background Si relative permittivity
            (default: ``n_si^2 = 3.4757^2``).
        design_transfer: Dense ``(n_design_cells, n_nodes)`` mesh-transfer matrix
            carrying the nodal perturbation onto the gyptis design cells (ticket
            04). ``None`` maps the perturbation node-for-node (identity).
        design_nodes: Which shared-mesh nodes ``theta`` addresses. The filtered
            field is scattered back to full node order before the doping map, so
            everything downstream still sees ``(n_nodes,)``. ``None`` means
            ``theta`` already spans every mesh node.
        components: Live differentiable components to compose. Defaults to the
            in-process components built from the local tesseract apis.
        loss_weight: Weight ``w`` of the modal free-carrier loss in the
            objective ``Δneff - w * alpha_mode`` [neff per dB/cm] (ticket 25).
            ``0`` (default) optimizes Δneff alone.
        mode_overlap: ``(n_design_cells,)`` mode-overlap weights from
            :func:`read_mode_overlap`. Required when ``loss_weight > 0``;
            without them the loss is not evaluated.

    Returns:
        Signed effective-index shift ``Δneff = Re[neff(-5V)] - Re[neff(0)]``,
        minus ``loss_weight`` times the modal loss in dB/cm. Smooth and
        differentiable through zero; depletion (the physical bias response)
        makes Δneff positive. Report ``VπLπ`` from it via :func:`vpi_lpi_v_cm`
        and the two terms via :func:`pipeline_with_terms`.
    """
    objective, _terms = pipeline_with_terms(
        theta,
        H=H,
        H_sum=H_sum,
        mesh_ref=mesh_ref,
        background_epsilon=background_epsilon,
        design_transfer=design_transfer,
        design_nodes=design_nodes,
        components=components,
        loss_weight=loss_weight,
        mode_overlap=mode_overlap,
    )
    return objective
