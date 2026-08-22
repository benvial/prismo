"""Paper-ready visualization outputs for the PRISMO pipeline.

Ref: rethink ticket 07 — the headline figure set. ``generate_outputs`` emits
four: the MMA convergence curve carrying both the optimized signed Δneff and
the reported VπLπ, the adjoint-vs-finite-difference gradient validation
(ticket 06), the initial-vs-optimized signed θ field, and the optical mode |E|
on the rib cross-section. Panels that serve none of those were dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri as mtri
from matplotlib.colors import SymLogNorm

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

_OUTPUT_DIR = Path("outputs")

# Dynamic range at which the convergence figure's VpiLpi axis switches from
# linear to symlog. VpiLpi goes as 1/delta_n_eff, so a run that starts near zero
# spans many decades and a linear axis flattens everything after the first
# iteration; a run that starts already converged spans almost nothing, and
# symlog's locator would place no ticks across it. Two decades separates the two.
_VPI_SYMLOG_SPAN: float = 100.0


def vpi_axis_is_symlog(magnitudes: list[float]) -> bool:
    """Whether the convergence figure's VpiLpi axis should be symlog, not linear.

    symlog exists to compress a wide span. VpiLpi goes as ``1/delta_n_eff``, so a
    run that starts near zero spans many decades and a linear axis flattens
    every iteration after the first. A run that starts already converged spans
    almost nothing, and there symlog's locator places no ticks at all, leaving a
    labelled axis with no numbers on it. Switch at two decades.
    """
    return max(magnitudes) > _VPI_SYMLOG_SPAN * min(magnitudes)


def _ensure_output_dir(output_dir: str | Path | None = None) -> Path:
    d = Path(output_dir) if output_dir is not None else _OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pcolormesh_field(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    triangles: np.ndarray | None = None,
    **kwargs: object,
) -> matplotlib.collections.Collection:
    """Render a nodal field at the mesh's own resolution.

    With ``triangles`` -- ``(n_triangles, 3)`` node indices, e.g. the silicon
    triangulation of the shared mesh -- the field is drawn on exactly those
    elements with Gouraud shading (the linear FE representation), so nothing
    is painted outside the device. Without them, an irregular point set is
    drawn on its Delaunay triangulation: full nodal resolution, unlike the
    coarse ``sqrt(n)``-pixel resample this function used to do, which turned a
    555-node mesh into ~23x23 blocks and smeared silicon values across the
    oxide. A regular grid still uses ``pcolormesh`` directly (the synthetic
    fallback grid).
    """
    if triangles is not None and len(triangles):
        return ax.tripcolor(
            mtri.Triangulation(x, y, triangles), values, shading="gouraud", **kwargs,
        )

    x_values = np.unique(x)
    y_values = np.unique(y)
    if x_values.size * y_values.size != values.size:
        return ax.tripcolor(
            mtri.Triangulation(x, y), values, shading="gouraud", **kwargs,
        )

    order = np.lexsort((x, y))
    field = np.asarray(values)[order].reshape(y_values.size, x_values.size)
    return ax.pcolormesh(x_values, y_values, field, shading="nearest", **kwargs)


def plot_convergence(
    history: list[dict],
    output_dir: str | Path | None = None,
    ftol_rel: float | None = None,
) -> Path:
    """Convergence plot: signed delta_n_eff and reported VpiLpi vs iteration.

    Args:
        history: Per-iteration records from ``optimize_doping``.
        output_dir: Directory to write ``convergence.pdf``.
        ftol_rel: MMA relative tolerance for the marker annotation.

    Returns:
        Path to the saved figure.
    """
    out = _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax_vpi = None

    if not history:
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            transform=ax.transAxes, fontsize=14,
        )
    else:
        from prismo.pipeline import vpi_lpi_v_cm

        iters = [h["iteration"] for h in history]
        values = [h["delta_n_eff"] for h in history]

        ax.plot(
            iters, values, "o-", markersize=3, color="#2c7bb6",
            label=r"$\Delta n_{\mathrm{eff}}$ (optimized)",
        )
        ax.axhline(
            y=values[-1], color="#d7191c", linestyle="--", alpha=0.6,
            label="final value",
        )

        if ftol_rel is not None and len(values) >= 2:
            threshold = abs(values[-1]) * ftol_rel
            idx = None
            for i in range(1, len(values)):
                if abs(values[i] - values[i - 1]) < threshold:
                    idx = i
                    break
            if idx is not None:
                ax.axvline(
                    x=iters[idx], color="#fdae61", linestyle=":", alpha=0.8,
                    label=f"ftol_rel={ftol_rel:.0e}",
                )

        # VpiLpi is the field-standard headline but is *reported*, not optimized
        # (ticket 03). It goes as 1/delta_n_eff, so an early iteration near zero
        # is orders of magnitude above the converged value and on a linear axis
        # would flatten the whole curve; an iteration exactly at zero is infinite
        # and cannot be drawn at all. So it rides a twin *symlog* axis -- signed,
        # since a wrong-polarity optimum reads negative -- with the finite points
        # only.
        vpi = ((i, vpi_lpi_v_cm(v)) for i, v in zip(iters, values, strict=True))
        finite = [(i, v) for i, v in vpi if np.isfinite(v)]
        if finite:
            magnitudes = [abs(v) for _, v in finite if v != 0.0]
            ax_vpi = ax.twinx()
            ax_vpi.plot(
                [i for i, _ in finite], [v for _, v in finite],
                "s--", markersize=3, color="#1a9641", alpha=0.8,
                label=r"$V_\pi L_\pi$ (reported)",
            )
            if magnitudes and vpi_axis_is_symlog(magnitudes):
                ax_vpi.set_yscale("symlog", linthresh=min(magnitudes))
            ax_vpi.set_ylabel(r"$V_\pi L_\pi$ [V$\cdot$cm]", color="#1a9641")
            ax_vpi.tick_params(axis="y", labelcolor="#1a9641")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\Delta n_{\mathrm{eff}}$")
    ax.set_title("MMA Convergence")
    handles, labels = ax.get_legend_handles_labels()
    if ax_vpi is not None:
        extra_handles, extra_labels = ax_vpi.get_legend_handles_labels()
        handles, labels = handles + extra_handles, labels + extra_labels
    ax.legend(handles, labels, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out / "convergence.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_doping_field(
    rho_initial: np.ndarray,
    rho_opt: np.ndarray,
    mesh_coords: np.ndarray,
    geometry: object | None = None,
    output_dir: str | Path | None = None,
    triangles: np.ndarray | None = None,
) -> Path:
    """Side-by-side doping field: initial vs optimized.

    Args:
        rho_initial: Initial design vector ``(n_nodes,)``.
        rho_opt: Optimized design vector ``(n_nodes,)``.
        mesh_coords: Node coordinates ``(n_nodes, 2)`` in micrometres -- the
            unit both mesh authors emit (ticket 15), and the unit the axes are
            labelled in, so no conversion happens here.
        geometry: ``RibWaveguideGeometry`` for overlay (optional).
        output_dir: Directory to write ``doping_field.pdf``.
        triangles: ``(n_triangles, 3)`` silicon triangulation of the shared
            mesh. With it the field is drawn on the device's own elements and
            the oxide stays blank; without it the whole point set is drawn on
            a Delaunay triangulation.

    Returns:
        Path to the saved figure.
    """
    out = _ensure_output_dir(output_dir)
    x, y = mesh_coords[:, 0], mesh_coords[:, 1]

    # Constrained layout, not tight_layout: the shared colorbar's axes are the
    # kind tight_layout cannot handle, and it repositioned the two panels over
    # the bar (the bar landed on top of the optimized panel).
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    # Signed design field theta in [-1, 1]: diverging map centred on the theta=0
    # junction so the p-side (theta<0) is not clipped.
    mesh_kw: dict = dict(vmin=-1.0, vmax=1.0, cmap="RdBu_r", triangles=triangles)

    _pcolormesh_field(ax0, x, y, rho_initial, **mesh_kw)
    ax0.set_title(r"Initial $\theta$")
    ax0.set_xlabel("x [µm]")
    ax0.set_ylabel("y [µm]")
    ax0.set_aspect("equal")

    sc1 = _pcolormesh_field(ax1, x, y, rho_opt, **mesh_kw)
    ax1.set_title(r"Optimized $\theta$")
    ax1.set_xlabel("x [µm]")
    ax1.set_ylabel("y [µm]")
    ax1.set_aspect("equal")

    for ax in (ax0, ax1):
        # Distinguish the undrawn surroundings from theta = 0 (white on the
        # diverging map) when only the silicon elements are painted.
        ax.set_facecolor("0.92")

    if geometry is not None:
        _overlay_geometry(ax0, geometry)
        _overlay_geometry(ax1, geometry)

    cbar = fig.colorbar(sc1, ax=[ax0, ax1], shrink=0.7, label=r"$\theta$ (signed)")
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    path = out / "doping_field.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_live_doping_field(
    doping: np.ndarray,
    mesh_coords: np.ndarray,
    iteration: int,
    geometry: object | None = None,
    output_dir: str | Path | None = None,
    name: str | None = None,
    triangles: np.ndarray | None = None,
) -> Path:
    """Write the optimizer's current signed doping-field image.

    One image per optimizer callback, named by ``name``; callers pass the
    iteration number so a long run leaves a snapshot per iteration rather than
    one file overwritten in place. ``mesh_coords`` is in micrometres, as
    everywhere else in the app. ``triangles`` is the silicon triangulation, as
    in :func:`plot_doping_field`.
    """
    out = _ensure_output_dir(output_dir)
    doping = np.asarray(doping, dtype=float)
    x, y = mesh_coords[:, 0], mesh_coords[:, 1]
    limit = float(np.max(np.abs(doping))) if doping.size else 1.0
    limit = max(limit, 1e14)

    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = _pcolormesh_field(
        ax,
        x,
        y,
        doping,
        triangles=triangles,
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=1e14, vmin=-limit, vmax=limit, base=10),
    )
    ax.set_facecolor("0.92")
    ax.set_title(f"Net doping — optimizer iteration {iteration}")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal")
    if geometry is not None:
        _overlay_geometry(ax, geometry)
    fig.colorbar(mesh, ax=ax, label=r"Net doping [cm$^{-3}$]")
    fig.tight_layout()

    name = name or "doping_field_live"

    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


@dataclass(frozen=True)
class OverlayGeometry:
    """The overlay frame of :func:`_overlay_geometry`, as plain values.

    ``RibWaveguideGeometry`` satisfies the same duck-typed contract but can
    only describe the mesh the *local* author builds (y from 0 at the
    substrate bottom, 0.5 µm substrate). The gyptis author centres its layer
    stack on y = 0 with a 0.35 µm substrate, so drawing the local frame on
    container figures put the rib outline off the device entirely (ticket 16).
    The container path builds one of these from the gyptis mesh itself instead.

    ``substrate_top`` is the y of the substrate/slab interface -- the value the
    local geometry exposes as ``substrate_thickness``, which only equals it in
    a frame whose substrate starts at y = 0.
    """

    rib_left: float
    rib_right: float
    slab_top: float
    rib_top: float
    substrate_top: float
    half_width: float
    contact_offset: float
    contact_width: float

    @property
    def substrate_thickness(self) -> float:
        """Alias for the duck-typed attribute ``_overlay_geometry`` reads."""
        return self.substrate_top


def _overlay_geometry(ax: plt.Axes, geometry: object) -> None:
    """Draw waveguide geometry overlay on an axis.

    The geometry's dimensions are already micrometres (ticket 15), the same
    unit as the node coordinates and the axes, so nothing is rescaled here.
    """
    # Geometry is duck-typed to keep plotting independent from mesh module.
    geom = geometry  # type: ignore[assignment]
    rib_l = geom.rib_left  # type: ignore[attr-defined]
    rib_r = geom.rib_right  # type: ignore[attr-defined]
    slab_top = geom.slab_top  # type: ignore[attr-defined]
    rib_top = geom.rib_top  # type: ignore[attr-defined]
    sub_top = geom.substrate_thickness  # type: ignore[attr-defined]
    hw = geom.half_width  # type: ignore[attr-defined]
    ct_off = geom.contact_offset  # type: ignore[attr-defined]
    ct_w = geom.contact_width  # type: ignore[attr-defined]

    ax.plot([rib_l, rib_r], [slab_top, slab_top], "w--", linewidth=0.8, alpha=0.7)
    ax.plot([rib_l, rib_r], [rib_top, rib_top], "w--", linewidth=0.8, alpha=0.7)
    ax.plot([rib_l, rib_l], [slab_top, rib_top], "w--", linewidth=0.8, alpha=0.7)
    ax.plot([rib_r, rib_r], [slab_top, rib_top], "w--", linewidth=0.8, alpha=0.7)

    ct_l_start = rib_l - ct_off - ct_w
    ct_l_end = rib_l - ct_off
    ct_r_start = rib_r + ct_off
    ct_r_end = rib_r + ct_off + ct_w

    for cs, ce in [(ct_l_start, ct_l_end), (ct_r_start, ct_r_end)]:
        ax.fill_between(
            [cs, ce], sub_top, slab_top, color="gray", alpha=0.3, edgecolor="none",
        )

    ax.fill_between([-hw, rib_l], slab_top, rib_top, color="gray", alpha=0.1, edgecolor="none")
    ax.fill_between([rib_r, hw], slab_top, rib_top, color="gray", alpha=0.1, edgecolor="none")

    props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray")
    ax.text(
        0.05, 0.95, "Si core", transform=ax.transAxes, fontsize=9,
        verticalalignment="top", bbox=props,
    )


@dataclass(frozen=True)
class ModeField:
    """The tracked optical mode's ``|E|`` profile on the shared mesh.

    ``abs_e`` is the peak-normalized magnitude at each mesh vertex and
    ``coords_um`` the matching ``(n_vertices, 2)`` vertex coordinates. Unlike the
    doping-field plots -- which take SI node coordinates and scale them -- these
    come straight off the gyptis mesh and are already in **micrometres**, so the
    plot uses them unscaled. ``rib_bounds`` is the optional
    ``(x_min, x_max, y_min, y_max)`` rectangle of the silicon rib interior, in
    the same micrometre frame, drawn as an outline so the mode can be read
    against the guiding core.
    """

    abs_e: np.ndarray
    coords_um: np.ndarray
    rib_bounds: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords_um, dtype=float)
        abs_e = np.asarray(self.abs_e, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("mode coordinates must have shape (n_vertices, 2)")
        if abs_e.ndim != 1 or abs_e.shape[0] != coords.shape[0]:
            raise ValueError(
                "the mode field needs one magnitude per vertex "
                f"({abs_e.shape} values for {coords.shape[0]} vertices)"
            )
        object.__setattr__(self, "abs_e", abs_e)
        object.__setattr__(self, "coords_um", coords)


def plot_mode_field(
    mode: ModeField,
    output_dir: str | Path | None = None,
) -> Path:
    """Optical mode ``|E|`` on the rib cross-section.

    The eigensolver's mode is what turns the doping design into a phase shift,
    so the headline set shows it directly: whichever mode the solve tracked, on
    the same cross-section as the doping figure (ticket 07).

    Args:
        mode: The tracked mode's normalized ``|E|`` and vertex coordinates.
        output_dir: Directory to write ``mode_field.pdf``.

    Returns:
        Path to the saved figure.
    """
    out = _ensure_output_dir(output_dir)
    x, y = mode.coords_um[:, 0], mode.coords_um[:, 1]

    # Rendered on the solver's own triangulation rather than resampled onto a
    # regular grid: the mesh is refined inside the rib, which is exactly where
    # the mode lives, and a uniform resample throws that refinement away.
    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.tripcolor(
        mtri.Triangulation(x, y),
        mode.abs_e,
        shading="gouraud",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(r"Optical mode $|E|$ (tracked)")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal")

    if mode.rib_bounds is not None:
        x0, x1, y0, y1 = mode.rib_bounds
        ax.plot(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            "w--", linewidth=0.8, alpha=0.8, label="Si rib",
        )
        ax.legend(loc="upper right", framealpha=0.6)

    fig.colorbar(mesh, ax=ax, label=r"$|E|$ (normalized)")
    fig.tight_layout()

    path = out / "mode_field.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# Default acceptance bar for the composed cross-boundary gradient (ticket 06).
# Central differences of a smooth pipeline are O(h^2), but the *composed* adjoint
# is limited by the least-exact link in the chain: the CT discrete adjoint reads
# carrier density back through an internal 1e-8 finite difference (audit #7) and
# the gyptis eigen-adjoint carries its solver tolerance. 1% relative agreement at
# the best step is the "gradients do real work" bar -- tight enough to catch a
# sign error, a missing adjoint term, or a unit mismatch across the boundary,
# loose enough not to chase the CT readout's own FD floor.
GRADIENT_VALIDATION_TOLERANCE = 1e-2


@dataclass(frozen=True)
class GradientValidationResult:
    """Outcome of an adjoint-vs-finite-difference gradient check (ticket 06).

    ``best_rel_errors[i]`` is the smallest relative error over the feasible
    finite-difference steps for direction ``i`` -- the point where central FD is
    closest to the adjoint's directional derivative before rounding or the CT
    readout's own FD floor takes over. ``worst_rel_error`` is the largest of
    those per-direction minima; the gradient passes when it stays at or below
    ``tolerance``.
    """

    n_directions: int
    best_rel_errors: list[float]
    worst_rel_error: float
    tolerance: float
    passed: bool
    figure_path: Path


def _sample_directions(
    rho: jax.Array,
    directions: list[jax.Array] | None,
    n_directions: int,
) -> list[jax.Array]:
    if directions is not None:
        return directions
    rng = np.random.default_rng(0)
    sampled = []
    for _ in range(n_directions):
        d = jnp.asarray(rng.standard_normal(rho.shape), dtype=rho.dtype)
        d = d / jnp.linalg.norm(d)
        sampled.append(d)
    return sampled


# Nodes this close to the ±1 rails count as pinned during gradient validation.
# An optimized design rails many nodes exactly at the bounds, so a dense random
# direction leaves the central stencil no feasible step at all; projecting the
# direction onto the interior nodes keeps the check meaningful (every unpinned
# variable is still validated) instead of raising after a finished run.
_RAIL_CLEARANCE = 1e-3


def _interior_direction(rho: jax.Array, direction: jax.Array) -> jax.Array | None:
    """Project a direction off rail-pinned components and renormalize.

    Returns ``None`` when every component with direction weight is pinned --
    there is nothing left for a central stencil to probe.
    """
    rho_np = np.asarray(rho)
    dir_np = np.asarray(direction)
    projected = np.where(np.abs(rho_np) < 1.0 - _RAIL_CLEARANCE, dir_np, 0.0)
    norm = float(np.linalg.norm(projected))
    if norm == 0.0:
        return None
    return jnp.asarray(projected / norm, dtype=np.asarray(direction).dtype)


def _feasible_steps(
    rho: jax.Array, direction: jax.Array, step_sizes: np.ndarray
) -> np.ndarray:
    """Steps that keep ``rho ± h·direction`` inside the signed box ``[-1, 1]``.

    The central stencil probes *both* sides, so a component ``i`` needs
    ``rho_i + h·d_i <= 1`` and ``rho_i - h·d_i >= -1`` (and the mirror pair for a
    negative ``d_i``). Both collapse to ``h·|d_i| <= 1 - |rho_i|``, so the whole
    box constraint is ``h <= min_i (1 - |rho_i|) / |d_i|`` over the nonzero
    components -- one symmetric bound instead of the per-sign upper-only check.
    """
    rho_np = np.asarray(rho)
    dir_np = np.asarray(direction)
    nonzero = dir_np != 0.0
    max_step = np.min(
        (1.0 - np.abs(rho_np[nonzero])) / np.abs(dir_np[nonzero]), initial=np.inf
    )
    return step_sizes[step_sizes < max_step]


def _gradient_validation_curves(
    pipeline_fn: Callable[..., jax.Array],
    rho: jax.Array,
    directions: list[jax.Array],
    step_sizes: np.ndarray,
) -> list[tuple[int, np.ndarray, list[float]]]:
    """Central-FD relative-error curve per direction against the JAX adjoint.

    Returns ``(index, feasible_steps, errors)`` for every direction that has at
    least one feasible step. The adjoint's directional derivative is
    ``grad · direction``; the finite-difference estimate is the central
    difference at each step.
    """
    grad_exact = jax.grad(pipeline_fn)(rho)
    curves: list[tuple[int, np.ndarray, list[float]]] = []
    for i, direction in enumerate(directions):
        feasible_steps = _feasible_steps(rho, direction, step_sizes)
        if len(feasible_steps) == 0:
            # A design railed at ±1 leaves no room for the central stencil in
            # a dense random direction; validate the interior subspace instead.
            interior = _interior_direction(rho, direction)
            if interior is None:
                continue
            direction = interior
            feasible_steps = _feasible_steps(rho, direction, step_sizes)
            if len(feasible_steps) == 0:
                continue
        exact_val = jnp.dot(grad_exact, direction)
        denom = max(float(abs(exact_val)), 1e-30)
        errors = []
        for h in feasible_steps:
            f_plus = pipeline_fn(rho + h * direction)
            f_minus = pipeline_fn(rho - h * direction)
            fd_val = (f_plus - f_minus) / (2.0 * h)
            errors.append(float(abs(fd_val - exact_val)) / denom)
        curves.append((i, feasible_steps, errors))
    return curves


def _render_gradient_validation(
    curves: list[tuple[int, np.ndarray, list[float]]],
    step_sizes: np.ndarray,
    n_directions: int,
    tolerance: float | None,
    out: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, feasible_steps, errors in curves:
        label = f"direction {i+1}" if n_directions > 1 else "FD"
        ax.loglog(feasible_steps, errors, "o-", markersize=3, label=label)

    # Central differences of a smooth pipeline converge at second order (see
    # GRADIENT_VALIDATION_TOLERANCE above), so the guide line is h^2, not h.
    ax.plot(step_sizes, step_sizes**2, "k--", alpha=0.4, label=r"$O(h^2)$")
    if tolerance is not None:
        ax.axhline(
            tolerance, color="crimson", ls=":", lw=1.5, label=f"tol = {tolerance:g}"
        )
    ax.set_xlabel("Step size h")
    ax.set_ylabel("Relative error")
    ax.set_title("Gradient Validation (central FD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out / "gradient_validation.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _gradient_validation_setup(
    pipeline_fn: Callable[..., jax.Array],
    rho: jax.Array,
    directions: list[jax.Array] | None,
    n_directions: int,
    step_sizes: np.ndarray | None,
    output_dir: str | Path | None,
) -> tuple[Path, list[tuple[int, np.ndarray, list[float]]], np.ndarray, int]:
    """Shared setup for the plotting and pass/fail entry points.

    Resolves the output dir, samples directions, defaults the steps, and computes
    the per-direction relative-error curves, raising if no direction has a
    feasible step. Returns ``(out, curves, step_sizes, n_directions)``.
    """
    out = _ensure_output_dir(output_dir)
    rho = jnp.asarray(rho)
    directions = _sample_directions(rho, directions, n_directions)
    n_directions = len(directions)
    if step_sizes is None:
        step_sizes = np.logspace(-6, -1, 20)

    curves = _gradient_validation_curves(pipeline_fn, rho, directions, step_sizes)
    if not curves:
        raise ValueError("Gradient validation has no feasible finite-difference steps")
    return out, curves, step_sizes, n_directions


def plot_gradient_validation(
    pipeline_fn: Callable[..., jax.Array],
    rho: jax.Array,
    directions: list[jax.Array] | None = None,
    n_directions: int = 3,
    step_sizes: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    tolerance: float | None = None,
) -> Path:
    """Gradient validation: relative error vs FD step size.

    Args:
        pipeline_fn: Callable ``(rho) -> scalar`` differentiable by JAX.
        rho: Reference design vector.
        directions: Random unit perturbation directions. Generated if
            ``None``.
        n_directions: Number of random directions (ignored if
            ``directions`` given).
        step_sizes: Central-difference steps. Defaults to 20 logarithmic
            steps from ``1e-6`` to ``1e-1``.
        output_dir: Directory to write ``gradient_validation.pdf``.
        tolerance: If given, draw the acceptance bar as a horizontal line on
            the figure. Use :func:`validate_gradient` for the pass/fail result.

    Returns:
        Path to the saved figure.
    """
    out, curves, step_sizes, n_directions = _gradient_validation_setup(
        pipeline_fn, rho, directions, n_directions, step_sizes, output_dir
    )
    return _render_gradient_validation(
        curves, step_sizes, n_directions, tolerance, out
    )


def validate_gradient(
    pipeline_fn: Callable[..., jax.Array],
    rho: jax.Array,
    directions: list[jax.Array] | None = None,
    n_directions: int = 3,
    step_sizes: np.ndarray | None = None,
    tolerance: float = GRADIENT_VALIDATION_TOLERANCE,
    output_dir: str | Path | None = None,
) -> GradientValidationResult:
    """Check the adjoint against central FD and gate on a stated tolerance.

    The proof behind ticket 06's headline: the composed ``rho -> Delta n_eff``
    gradient is exercised through whatever ``pipeline_fn`` closes over -- pass a
    ``pipeline`` bound to the live ChargeTransport and gyptis components to prove
    the cross-boundary adjoint (CT included, audit #7), not a stub. The figure is
    always written; ``passed`` records whether the worst per-direction best error
    stayed at or below ``tolerance`` (it does not raise, so a failing check still
    produces the diagnostic figure).

    Args:
        pipeline_fn: Callable ``(rho) -> scalar`` differentiable by JAX.
        rho: Reference design vector to validate the gradient at.
        directions: Perturbation directions. Random unit vectors if ``None``.
        n_directions: Number of random directions (ignored if ``directions``).
        step_sizes: Central-difference steps. Defaults to 20 logarithmic steps
            from ``1e-6`` to ``1e-1``.
        tolerance: Acceptance bar on the worst per-direction best relative error.
        output_dir: Directory to write ``gradient_validation.pdf``.

    Returns:
        A :class:`GradientValidationResult` with the per-direction best errors,
        the worst of them, the pass/fail verdict, and the figure path.
    """
    out, curves, step_sizes, n_directions = _gradient_validation_setup(
        pipeline_fn, rho, directions, n_directions, step_sizes, output_dir
    )

    best_rel_errors = [min(errors) for _, _, errors in curves]
    worst_rel_error = max(best_rel_errors)
    path = _render_gradient_validation(
        curves, step_sizes, n_directions, tolerance, out
    )
    return GradientValidationResult(
        n_directions=len(curves),
        best_rel_errors=best_rel_errors,
        worst_rel_error=worst_rel_error,
        tolerance=tolerance,
        passed=worst_rel_error <= tolerance,
        figure_path=path,
    )


def generate_outputs(
    rho_initial: np.ndarray,
    rho_opt: np.ndarray,
    history: list[dict],
    mesh_coords: np.ndarray,
    geometry: object | None = None,
    pipeline_fn: Callable[..., jax.Array] | None = None,
    ftol_rel: float | None = None,
    gradient_validation_directions: int = 3,
    gradient_validation_steps: np.ndarray | None = None,
    gradient_validation_rho: np.ndarray | None = None,
    mode_field: ModeField | None = None,
    output_dir: str | Path | None = None,
    mesh_triangles: np.ndarray | None = None,
) -> list[Path]:
    """Generate the headline output figures (ticket 07).

    Args:
        rho_initial: Initial design vector.
        rho_opt: Optimized design vector.
        history: Optimization history from ``optimize_doping``.
        mesh_coords: Node coordinates ``(n_nodes, 2)`` in micrometres.
        mesh_triangles: ``(n_triangles, 3)`` silicon triangulation of the
            shared mesh, so the doping figure draws the device's own elements
            rather than a Delaunay fill of the whole domain.
        geometry: ``RibWaveguideGeometry`` instance.
        pipeline_fn: Differentiable pipeline function for gradient
            validation. Skipped if ``None``.
        ftol_rel: MMA relative tolerance for the convergence marker.
        gradient_validation_directions: Number of finite-difference directions.
        gradient_validation_steps: Central-difference steps for gradient validation.
        gradient_validation_rho: Reference design for gradient validation.
            Defaults to ``rho_opt``.
        mode_field: Tracked optical mode ``|E|`` from the gyptis backend.
            Skipped if ``None`` (no gyptis backend bound).
        output_dir: Output directory (default: ``outputs/``).

    Returns:
        List of paths to the generated plot files.
    """
    out = _ensure_output_dir(output_dir)
    paths: list[Path] = []

    conv = plot_convergence(history, output_dir=out, ftol_rel=ftol_rel)
    paths.append(conv)

    doping = plot_doping_field(
        rho_initial,
        rho_opt,
        mesh_coords,
        geometry,
        output_dir=out,
        triangles=mesh_triangles,
    )
    paths.append(doping)

    if pipeline_fn is not None:
        rho_jax = jnp.asarray(
            rho_opt if gradient_validation_rho is None else gradient_validation_rho,
        )
        # Post-hoc diagnostic on a finished multi-minute optimization: a
        # failure here (e.g. a fully railed design with no feasible FD step)
        # must cost this one figure, not every figure of the run.
        try:
            gv = plot_gradient_validation(
                pipeline_fn,
                rho_jax,
                n_directions=gradient_validation_directions,
                step_sizes=gradient_validation_steps,
                output_dir=out,
            )
        except Exception as exc:
            print(f"WARNING: gradient validation figure skipped ({exc})", flush=True)
        else:
            paths.append(gv)

    if mode_field is not None:
        paths.append(plot_mode_field(mode_field, output_dir=out))

    return paths
