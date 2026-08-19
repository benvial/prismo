"""Paper-ready visualization outputs for the PRISMO pipeline.

Ref: ticket 16 — produce convergence plots, doping-field visualizations,
delta_n_eff breakdown, and gradient-validation plots for the hackathon
submission.
"""

from __future__ import annotations

from collections.abc import Callable
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


def _ensure_output_dir(output_dir: str | Path | None = None) -> Path:
    d = Path(output_dir) if output_dir is not None else _OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pcolormesh_field(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    **kwargs: object,
) -> matplotlib.collections.QuadMesh:
    """Render nodal values with ``pcolormesh``, resampling irregular meshes."""
    x_values = np.unique(x)
    y_values = np.unique(y)
    if x_values.size * y_values.size != values.size:
        triangulation = mtri.Triangulation(x, y)
        interpolator = mtri.LinearTriInterpolator(triangulation, values)
        n_x = min(400, max(2, int(np.sqrt(values.size))))
        x_grid = np.linspace(x.min(), x.max(), n_x)
        x_span = x.max() - x.min()
        y_span = y.max() - y.min()
        n_y = max(2, round(n_x * y_span / x_span)) if x_span else n_x
        y_grid = np.linspace(y.min(), y.max(), n_y)
        xx, yy = np.meshgrid(x_grid, y_grid)
        return ax.pcolormesh(
            x_grid, y_grid, interpolator(xx, yy), shading="nearest", **kwargs,
        )

    order = np.lexsort((x, y))
    field = np.asarray(values)[order].reshape(y_values.size, x_values.size)
    return ax.pcolormesh(x_values, y_values, field, shading="nearest", **kwargs)


def plot_convergence(
    history: list[dict],
    output_dir: str | Path | None = None,
    ftol_rel: float | None = None,
) -> Path:
    """Convergence plot: delta_n_eff vs iteration.

    Args:
        history: Per-iteration records from ``optimize_doping``.
        output_dir: Directory to write ``convergence.pdf``.
        ftol_rel: MMA relative tolerance for the marker annotation.

    Returns:
        Path to the saved figure.
    """
    out = _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6, 4))

    if not history:
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            transform=ax.transAxes, fontsize=14,
        )
    else:
        iters = [h["iteration"] for h in history]
        values = [h["delta_n_eff"] for h in history]

        ax.plot(
            iters, values, "o-", markersize=3, color="#2c7bb6",
            label=r"$\Delta n_{\mathrm{eff}}$",
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

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\Delta n_{\mathrm{eff}}$")
    ax.set_title("MMA Convergence")
    ax.legend()
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
) -> Path:
    """Side-by-side doping field: initial vs optimized.

    Args:
        rho_initial: Initial design vector ``(n_nodes,)``.
        rho_opt: Optimized design vector ``(n_nodes,)``.
        mesh_coords: Node coordinates ``(n_nodes, 2)`` in meters.
        geometry: ``RibWaveguideGeometry`` for overlay (optional).
        output_dir: Directory to write ``doping_field.pdf``.

    Returns:
        Path to the saved figure.
    """
    out = _ensure_output_dir(output_dir)
    x, y = mesh_coords[:, 0] * 1e6, mesh_coords[:, 1] * 1e6

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

    # Signed design field theta in [-1, 1]: diverging map centred on the theta=0
    # junction so the p-side (theta<0) is not clipped.
    mesh_kw: dict = dict(vmin=-1.0, vmax=1.0, cmap="RdBu_r")

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

    if geometry is not None:
        _overlay_geometry(ax0, geometry)
        _overlay_geometry(ax1, geometry)

    cbar = fig.colorbar(sc1, ax=[ax0, ax1], shrink=0.7, label=r"$\theta$ (signed)")
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    fig.tight_layout()
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
) -> Path:
    """Update the optimizer's current signed doping-field image.

    The image is atomically replaced at every optimizer callback, so it can be
    opened while a long container-backed solve is still running.
    """
    out = _ensure_output_dir(output_dir)
    doping = np.asarray(doping, dtype=float)
    x, y = mesh_coords[:, 0] * 1e6, mesh_coords[:, 1] * 1e6
    limit = float(np.max(np.abs(doping))) if doping.size else 1.0
    limit = max(limit, 1e14)

    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = _pcolormesh_field(
        ax,
        x,
        y,
        doping,
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=1e14, vmin=-limit, vmax=limit, base=10),
    )
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
    temporary_path = out / f".{name}.tmp.png"
    fig.savefig(temporary_path)
    plt.close(fig)
    temporary_path.replace(path)
    return path


def _overlay_geometry(ax: plt.Axes, geometry: object) -> None:
    """Draw waveguide geometry overlay on an axis.

    Coordinates are converted from meters to µm.
    """
    # Geometry is duck-typed to keep plotting independent from mesh module.
    geom = geometry  # type: ignore[assignment]
    rib_l = geom.rib_left * 1e6  # type: ignore[attr-defined]
    rib_r = geom.rib_right * 1e6  # type: ignore[attr-defined]
    slab_top = geom.slab_top * 1e6  # type: ignore[attr-defined]
    rib_top = geom.rib_top * 1e6  # type: ignore[attr-defined]
    sub_top = geom.substrate_thickness * 1e6  # type: ignore[attr-defined]
    hw = geom.half_width * 1e6  # type: ignore[attr-defined]
    ct_off = geom.contact_offset * 1e6  # type: ignore[attr-defined]
    ct_w = geom.contact_width * 1e6  # type: ignore[attr-defined]

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


def plot_delta_neff_breakdown(
    delta_n: float,
    delta_alpha: float,
    output_dir: str | Path | None = None,
) -> Path:
    """Stacked bar chart: delta_n_eff from delta_n vs delta_alpha.

    Args:
        delta_n: Effective-index change from real-index perturbation.
        delta_alpha: Effective-index change from absorption perturbation.
        output_dir: Directory to write ``breakdown.pdf``.

    Returns:
        Path to the saved figure.
    """
    out = _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(4, 4))

    categories = [r"$\Delta n$", r"$\Delta\alpha$"]
    values = [abs(delta_n), abs(delta_alpha)]
    colors = ["#2c7bb6", "#d7191c"]

    bars = ax.bar(categories, values, color=colors, width=0.5)
    ax.set_ylabel(r"$|\Delta n_{\mathrm{eff}}|$ contribution")
    ax.set_title(r"$\Delta n_{\mathrm{eff}}$ Breakdown")

    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.2e}", ha="center", va="bottom", fontsize=10,
        )

    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = out / "breakdown.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gradient_validation(
    pipeline_fn: Callable[..., jax.Array],
    rho: jax.Array,
    directions: list[jax.Array] | None = None,
    n_directions: int = 3,
    step_sizes: np.ndarray | None = None,
    output_dir: str | Path | None = None,
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

    Returns:
        Path to the saved figure.
    """
    out = _ensure_output_dir(output_dir)
    rho = jnp.asarray(rho)
    grad_exact = jax.grad(pipeline_fn)(rho)

    if directions is None:
        rng = np.random.default_rng(0)
        directions = []
        for _ in range(n_directions):
            d = jnp.asarray(rng.standard_normal(rho.shape), dtype=rho.dtype)
            d = d / jnp.linalg.norm(d)
            directions.append(d)
        n_directions = len(directions)

    if step_sizes is None:
        step_sizes = np.logspace(-6, -1, 20)

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, direction in enumerate(directions):
        positive = np.asarray(direction) > 0.0
        negative = np.asarray(direction) < 0.0
        rho_np = np.asarray(rho)
        # Keep the perturbed field inside the signed design bounds [-1, 1] so
        # every FD sample stays feasible for the optimizer's box.
        max_step = min(
            np.min((1.0 - rho_np[positive]) / np.asarray(direction)[positive], initial=np.inf),
            np.min((rho_np[negative] + 1.0) / -np.asarray(direction)[negative], initial=np.inf),
        )
        feasible_steps = step_sizes[step_sizes < max_step]
        if len(feasible_steps) == 0:
            continue
        errors = []
        for h in feasible_steps:
            f_plus = pipeline_fn(rho + h * direction)
            f_minus = pipeline_fn(rho - h * direction)
            fd_val = (f_plus - f_minus) / (2.0 * h)
            exact_val = jnp.dot(grad_exact, direction)
            denom = max(float(abs(exact_val)), 1e-30)
            errors.append(float(abs(fd_val - exact_val)) / denom)
        label = f"direction {i+1}" if n_directions > 1 else "FD"
        ax.loglog(feasible_steps, errors, "o-", markersize=3, label=label)

    if not ax.lines:
        raise ValueError("Gradient validation has no feasible finite-difference steps")

    ax.plot(step_sizes, step_sizes, "k--", alpha=0.4, label=r"$O(h)$")
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
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Generate all paper-ready output plots.

    Args:
        rho_initial: Initial design vector.
        rho_opt: Optimized design vector.
        history: Optimization history from ``optimize_doping``.
        mesh_coords: Node coordinates ``(n_nodes, 2)``.
        geometry: ``RibWaveguideGeometry`` instance.
        pipeline_fn: Differentiable pipeline function for gradient
            validation. Skipped if ``None``.
        ftol_rel: MMA relative tolerance for the convergence marker.
        gradient_validation_directions: Number of finite-difference directions.
        gradient_validation_steps: Central-difference steps for gradient validation.
        gradient_validation_rho: Reference design for gradient validation.
            Defaults to ``rho_opt``.
        output_dir: Output directory (default: ``outputs/``).

    Returns:
        List of paths to the generated plot files.
    """
    out = _ensure_output_dir(output_dir)
    paths: list[Path] = []

    conv = plot_convergence(history, output_dir=out, ftol_rel=ftol_rel)
    paths.append(conv)

    doping = plot_doping_field(
        rho_initial, rho_opt, mesh_coords, geometry, output_dir=out,
    )
    paths.append(doping)

    if pipeline_fn is not None:
        import jax.numpy as jnp

        rho_jax = jnp.asarray(
            rho_opt if gradient_validation_rho is None else gradient_validation_rho,
        )
        gv = plot_gradient_validation(
            pipeline_fn,
            rho_jax,
            n_directions=gradient_validation_directions,
            step_sizes=gradient_validation_steps,
            output_dir=out,
        )
        paths.append(gv)

    breakdown = plot_delta_neff_breakdown(
        float(history[-1]["delta_n_eff"]) if history else 0.0,
        0.0,
        output_dir=out,
    )
    paths.append(breakdown)

    return paths
