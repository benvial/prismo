"""Paper-ready visualization outputs for the PRISMO pipeline.

The headline figure set. ``generate_outputs`` emits four: the MMA convergence
curve carrying both the optimized signed Δneff and the reported VπLπ, the
adjoint-vs-finite-difference gradient validation, the initial-vs-optimized
signed θ field, and the optical mode ``|E|`` on the rib cross-section. Panels that serve none of those were dropped. A
loss-aware run adds the loss history, the Δneff-vs-loss trade-off
plane, and an animation of the doping field over the optimizer's evaluations.
A run that swept the bias adds the initial-vs-optimized figures of merit
against reverse bias.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri as mtri
from matplotlib.colors import SymLogNorm

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

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
            mtri.Triangulation(x, y, triangles),
            values,
            shading="gouraud",
            **kwargs,
        )

    x_values = np.unique(x)
    y_values = np.unique(y)
    if x_values.size * y_values.size != values.size:
        return ax.tripcolor(
            mtri.Triangulation(x, y),
            values,
            shading="gouraud",
            **kwargs,
        )

    order = np.lexsort((x, y))
    field = np.asarray(values)[order].reshape(y_values.size, x_values.size)
    return ax.pcolormesh(x_values, y_values, field, shading="nearest", **kwargs)


# Relative warm/cold discrepancy above which the reported optimum is flagged
# . The ChargeTransport Newton solves converge to 1e-10 and the
# eigensolve is deterministic, so a warm and a cold solve of the same design
# agree to ~1e-8 relative when the steady state is unique; anything above this
# means the warm value depended on the solve history, not on the design.
COLD_REEVALUATION_RTOL = 1e-4


@dataclass(frozen=True)
class ColdReevaluation:
    """Warm-vs-cold Δneff of the reported design.

    ``warm_delta_neff`` is the value the optimizer saw (solved from the
    neighbouring designs' Newton starting points); ``cold_delta_neff`` is the
    same design solved after the ChargeTransport worker dropped every warm
    solution. The headline (and VπLπ) is the cold value.
    """

    warm_delta_neff: float
    cold_delta_neff: float
    tolerance: float = COLD_REEVALUATION_RTOL

    @property
    def rel_discrepancy(self) -> float:
        """``|cold - warm| / max(|cold|, |warm|)`` (0 when both are zero)."""
        scale = max(abs(self.cold_delta_neff), abs(self.warm_delta_neff))
        if scale == 0.0:
            return 0.0
        return abs(self.cold_delta_neff - self.warm_delta_neff) / scale

    @property
    def passed(self) -> bool:
        """Whether the discrepancy is within the tolerance."""
        return self.rel_discrepancy <= self.tolerance


def history_loss_series(history: list[dict]) -> tuple[list[int], list[float]]:
    """``(iterations, modal_loss_db_cm)`` for the entries that evaluated a loss.

    The optimizer records ``modal_loss_db_cm`` as ``None`` when no mode-overlap
    weights were bound; a history with no finite loss yields two
    empty lists and the convergence figure draws no loss panel.
    """
    iters: list[int] = []
    losses: list[float] = []
    for entry in history:
        loss = entry.get("modal_loss_db_cm")
        if loss is None or not np.isfinite(loss):
            continue
        iters.append(int(entry["iteration"]))
        losses.append(float(loss))
    return iters, losses


def _history_fom_series(history: list[dict]) -> list[tuple[int, float]]:
    """``(iteration, VπLπ·alpha)`` for every entry with a finite figure of merit."""
    from prismo.pipeline import loss_figure_of_merit_v_db

    by_iter = {int(h["iteration"]): h for h in history}
    iters, losses = history_loss_series(history)
    fom = [
        (i, loss_figure_of_merit_v_db(by_iter[i]["delta_n_eff"], loss))
        for i, loss in zip(iters, losses, strict=True)
    ]
    return [(i, v) for i, v in fom if np.isfinite(v)]


def _legend_below(ax: Any, handles: list[Any], labels: list[str]) -> None:
    """Legend in a row under the x-axis, clear of both y-axes' curves."""
    if not labels:
        return
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(len(labels), 3),
        frameon=False,
        handlelength=2.0,
    )


def plot_loss_convergence(
    history: list[dict],
    output_dir: str | Path | None = None,
) -> Path | None:
    """Modal loss [dB/cm] and the VπLπ·alpha figure of merit [V·dB] vs iteration.

    The companion of :func:`plot_convergence` for the loss axis:
    the free-carrier loss of the unbiased device at every evaluation, and the
    literature's efficiency-loss figure of merit on a twin axis. Returns
    ``None`` (writes nothing) when the history carries no evaluated loss.
    """
    iters, losses = history_loss_series(history)
    if not iters:
        return None
    out = _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        iters,
        losses,
        "o-",
        markersize=3,
        color="#d7191c",
        label=r"modal loss $\alpha$ (0 V)",
    )
    ax.set_ylabel(r"$\alpha$ [dB/cm]", color="#d7191c")
    ax.tick_params(axis="y", labelcolor="#d7191c")
    handles, labels = ax.get_legend_handles_labels()
    finite = _history_fom_series(history)
    if finite:
        ax_fom = ax.twinx()
        ax_fom.plot(
            [i for i, _ in finite],
            [v for _, v in finite],
            "s--",
            markersize=3,
            color="#7b3294",
            alpha=0.8,
            label=r"$V_\pi L_\pi \cdot \alpha$ (reported)",
        )
        magnitudes = [abs(v) for _, v in finite if v != 0.0]
        if magnitudes and vpi_axis_is_symlog(magnitudes):
            ax_fom.set_yscale("symlog", linthresh=min(magnitudes))
        ax_fom.set_ylabel(r"$V_\pi L_\pi \cdot \alpha$ [V$\cdot$dB]", color="#7b3294")
        ax_fom.tick_params(axis="y", labelcolor="#7b3294")
        extra_handles, extra_labels = ax_fom.get_legend_handles_labels()
        handles, labels = handles + extra_handles, labels + extra_labels
    _legend_below(ax, handles, labels)
    ax.set_xlabel("Iteration")
    ax.set_title("Modal loss and figure of merit")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "loss_convergence.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# Iso-figure-of-merit guide curves on the trade-off figure, V·dB. The
# literature's good depletion modulators sit at 10-30 V·dB.
_TRADEOFF_ISO_FOM_V_DB: tuple[float, ...] = (10.0, 30.0, 100.0, 300.0)


def iso_fom_delta_neff(loss_db_cm: np.ndarray, fom_v_db: float) -> np.ndarray:
    """Δneff along the curve ``VπLπ·alpha = fom``: ``|V|·λ·alpha / (2·fom)``."""
    from prismo.pipeline import _WAVELENGTH_CM, REVERSE_BIAS_V

    return (
        abs(REVERSE_BIAS_V) * _WAVELENGTH_CM * np.asarray(loss_db_cm) / (2.0 * fom_v_db)
    )


def plot_tradeoff(
    history: list[dict],
    output_dir: str | Path | None = None,
) -> Path | None:
    """The optimizer's path in the (modal loss, Δneff) plane.

    Every evaluated design is a point, coloured by iteration and joined in
    order, against iso-``VπLπ·alpha`` hyperbolae: a design moving up-left
    trades loss for efficiency, and crossing below the 30 V·dB curve puts it
    in the literature's band. Returns ``None`` when no loss was evaluated.
    """
    iters, losses = history_loss_series(history)
    if not iters:
        return None
    by_iter = {int(h["iteration"]): h for h in history}
    dneff = np.asarray([float(by_iter[i]["delta_n_eff"]) for i in iters])
    loss = np.asarray(losses)
    out = _ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    loss_max = float(loss.max()) * 1.15 if loss.max() > 0 else 1.0
    loss_grid = np.linspace(0.0, loss_max, 200)
    # The frame is set by the trajectory; the guide curves are clipped to it
    # and each label sits where its curve leaves the frame.
    y_top = float(dneff.max()) * 1.15 if dneff.max() > 0 else 1.0
    y_bottom = min(0.0, float(dneff.min()) * 1.15)
    for fom in _TRADEOFF_ISO_FOM_V_DB:
        curve = iso_fom_delta_neff(loss_grid, fom)
        inside = curve <= y_top
        if not np.any(inside):
            continue
        ax.plot(loss_grid, curve, color="0.6", linestyle=":", linewidth=1)
        last = int(np.flatnonzero(inside)[-1])
        ax.annotate(
            f"{fom:g} V·dB",
            xy=(loss_grid[last], curve[last]),
            xytext=(-2, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8,
            color="0.4",
        )
    ax.plot(loss, dneff, "-", color="0.3", linewidth=1, alpha=0.6, zorder=2)
    sc = ax.scatter(loss, dneff, c=iters, cmap="viridis", s=22, zorder=3)
    ax.plot(
        [loss[0]],
        [dneff[0]],
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor="#1a9641",
        linestyle="none",
        label="seed",
        zorder=4,
    )
    ax.plot(
        [loss[-1]],
        [dneff[-1]],
        marker="*",
        markersize=13,
        color="#d7191c",
        linestyle="none",
        label="last evaluation",
        zorder=4,
    )
    fig.colorbar(sc, ax=ax, label="Iteration")
    ax.set_xlim(0.0, loss_max)
    if y_top > y_bottom:
        ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel(r"modal loss $\alpha$ (0 V) [dB/cm]")
    ax.set_ylabel(r"$\Delta n_{\mathrm{eff}}$")
    ax.set_title("Efficiency-loss trade-off")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "tradeoff.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# The two designs the bias sweep compares, and their colours: the seed the run
# started from and the design MMA reported.
_SWEEP_STYLE: dict[str, dict] = {
    "initial": {
        "color": "#2c7bb6",
        "marker": "o",
        "linestyle": "--",
        "label": "initial",
    },
    "optimized": {
        "color": "#d7191c",
        "marker": "s",
        "linestyle": "-",
        "label": "optimized",
    },
}


def _sweep_arrays(points: Sequence[Any]) -> dict[str, np.ndarray]:
    """The four series of one bias sweep as arrays, keyed by quantity."""
    return {
        "bias_v": np.asarray([float(pt.bias_v) for pt in points]),
        "delta_neff": np.asarray([float(pt.delta_neff) for pt in points]),
        "modal_loss_db_cm": np.asarray([float(pt.modal_loss_db_cm) for pt in points]),
        "fom_v_db": np.asarray([float(pt.fom_v_db) for pt in points]),
    }


def plot_bias_sweep(
    initial: Sequence[Any] | None,
    optimized: Sequence[Any] | None,
    output_dir: str | Path | None = None,
) -> Path | None:
    """The three reported figures of merit vs reverse bias, initial vs optimized.

    Three panels sharing the bias axis -- Δneff, the modal free-carrier loss
    alpha, and the VπLπ·alpha efficiency-loss figure of merit -- each carrying
    one curve per design, so what the optimizer bought is visible across the
    whole operating range rather than only at the single bias it optimized. The
    bias axis runs left to right from 0 V into reverse bias.

    Δneff is zero at 0 V by construction (it is measured against that
    equilibrium), so VπLπ·alpha is infinite there and that point is dropped
    from the third panel rather than drawn off-scale.

    Args:
        initial: :class:`~prismo.pipeline.BiasPoint` sequence for the seed
            design; ``None`` or empty draws no initial curve.
        optimized: The same for the reported design.
        output_dir: Directory to write ``bias_sweep.pdf``.

    Returns:
        Path to the saved figure, or ``None`` when neither design swept.
    """
    sweeps = [
        (key, _sweep_arrays(points))
        for key, points in (("initial", initial), ("optimized", optimized))
        if points
    ]
    if not sweeps:
        return None
    out = _ensure_output_dir(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    panels = (
        ("delta_neff", r"$\Delta n_{\mathrm{eff}}$", "Index shift"),
        ("modal_loss_db_cm", r"$\alpha$ [dB/cm]", "Modal free-carrier loss"),
        ("fom_v_db", r"$V_\pi L_\pi \cdot \alpha$ [V$\cdot$dB]", "Figure of merit"),
    )
    for ax, (key, ylabel, title) in zip(axes, panels, strict=True):
        for name, series in sweeps:
            values = series[key]
            # A design whose physics never reported a loss carries nan across
            # the panel; the finite mask also drops the infinite 0 V figure of
            # merit, which would otherwise set the whole panel's scale.
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            ax.plot(
                series["bias_v"][finite],
                values[finite],
                markersize=4,
                **_SWEEP_STYLE[name],
            )
        ax.set_xlabel("Bias [V]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    # Reverse bias is negative: invert the axis so deepening bias reads left to
    # right, the way the sweep is run.
    axes[0].invert_xaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        axes[0].legend(handles, labels, loc="best")
    fig.tight_layout()
    path = out / "bias_sweep.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_convergence(
    history: list[dict],
    output_dir: str | Path | None = None,
    ftol_rel: float | None = None,
    cold_reevaluation: ColdReevaluation | None = None,
) -> Path:
    """Convergence plot: signed delta_n_eff and reported VpiLpi vs iteration.

    The loss axis of a loss-aware run is its own figure,
    :func:`plot_loss_convergence`.

    Args:
        history: Per-iteration records from ``optimize_doping``.
        output_dir: Directory to write ``convergence.pdf``.
        ftol_rel: MMA relative tolerance for the marker annotation.
        cold_reevaluation: Warm/cold Δneff of the reported design; drawn as a
            marker at the last iteration, with a warning annotation when the
            discrepancy exceeds its tolerance.

    Returns:
        Path to the saved figure.
    """
    out = _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax_vpi = None

    if not history:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
    else:
        from prismo.pipeline import vpi_lpi_v_cm

        iters = [h["iteration"] for h in history]
        values = [h["delta_n_eff"] for h in history]

        ax.plot(
            iters,
            values,
            "o-",
            markersize=3,
            color="#2c7bb6",
            label=r"$\Delta n_{\mathrm{eff}}$ (optimized)",
        )
        ax.axhline(
            y=values[-1],
            color="#d7191c",
            linestyle="--",
            alpha=0.6,
            label="final value",
        )

        if cold_reevaluation is not None:
            ax.plot(
                [iters[-1]],
                [cold_reevaluation.cold_delta_neff],
                marker="*",
                markersize=11,
                linestyle="none",
                color="#7b3294",
                label=r"$\Delta n_{\mathrm{eff}}$ (cold re-evaluation)",
                zorder=5,
            )
            if not cold_reevaluation.passed:
                ax.annotate(
                    "WARNING: warm/cold mismatch "
                    f"{cold_reevaluation.rel_discrepancy:.1e} "
                    f"> {cold_reevaluation.tolerance:.0e}",
                    xy=(0.02, 0.97),
                    xycoords="axes fraction",
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#d7191c",
                    bbox={"boxstyle": "round", "fc": "white", "ec": "#d7191c"},
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
                    x=iters[idx],
                    color="#fdae61",
                    linestyle=":",
                    alpha=0.8,
                    label=f"ftol_rel={ftol_rel:.0e}",
                )

        # VpiLpi is the field-standard headline but is *reported*, not optimized
        # . It goes as 1/delta_n_eff, so an early iteration near zero
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
                [i for i, _ in finite],
                [v for _, v in finite],
                "s--",
                markersize=3,
                color="#1a9641",
                alpha=0.8,
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
    # Below the axes: ``loc="best"`` only dodges the primary axis' artists and
    # lands on the twin-axis curve.
    _legend_below(ax, handles, labels)
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
            unit both mesh authors emit, and the unit the axes are
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


def _zoom_to_silicon(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    triangles: np.ndarray | None,
    pad: float = 0.15,
) -> None:
    """Crop the vertical axis to the painted silicon (plus ``pad`` µm).

    The shared mesh is mostly oxide and PML; the device is a thin strip, so a
    full-domain frame spends most of its pixels on blank surroundings.
    """
    if triangles is None or not len(triangles):
        return
    nodes = np.unique(np.asarray(triangles).ravel())
    ax.set_xlim(float(x[nodes].min()) - pad, float(x[nodes].max()) + pad)
    ax.set_ylim(float(y[nodes].min()) - pad, float(y[nodes].max()) + pad)


# Animation writers tried per requested format; a missing encoder skips that
# format rather than failing the run (ffmpeg is a system tool, Pillow a dep).
_ANIMATION_WRITERS: dict[str, str] = {"gif": "pillow", "mp4": "ffmpeg"}
_DOPING_SYMLOG_LINTHRESH = 1e14


def animate_doping_evolution(
    doping_frames: Sequence[np.ndarray],
    history: list[dict],
    mesh_coords: np.ndarray,
    geometry: object | None = None,
    output_dir: str | Path | None = None,
    triangles: np.ndarray | None = None,
    fps: int = 6,
    formats: Sequence[str] = ("gif", "mp4"),
    name: str = "doping_evolution",
) -> list[Path]:
    """Animate the net doping the solvers saw at every optimizer evaluation.

    One frame per evaluated design, on a colour scale fixed across the whole
    run (symlog, like the live frames), titled with the iteration, its Δneff
    and -- when evaluated -- its modal loss; a trial the optimizer rejected
    (objective below the running best) is labelled as such, so the move-limit
    halvings read as what they are rather than as progress.

    Args:
        doping_frames: Full-node net doping ``(n_nodes,)`` [cm^-3] per
            evaluation, in history order -- the filtered, scattered, mapped
            field, i.e. what ChargeTransport solved on.
        history: The matching per-evaluation records (same length).
        mesh_coords: ``(n_nodes, 2)`` node coordinates in micrometres.
        geometry: Overlay frame (see :func:`_overlay_geometry`); optional.
        output_dir: Directory for ``<name>.gif`` / ``<name>.mp4``.
        triangles: Silicon triangulation, as in :func:`plot_doping_field`.
        fps: Frames per second.
        formats: Which of ``gif`` (Pillow) and ``mp4`` (ffmpeg) to write;
            a format whose encoder is unavailable is skipped with a note.
        name: Output file stem.

    Returns:
        Paths written (empty when there are no frames or no usable writer).
    """
    from matplotlib import animation

    frames = [np.asarray(f, dtype=float) for f in doping_frames]
    if not frames:
        return []
    if len(frames) != len(history):
        raise ValueError(
            f"{len(frames)} doping frames but {len(history)} history entries; "
            "the animation needs one record per evaluated design"
        )
    out = _ensure_output_dir(output_dir)
    x, y = mesh_coords[:, 0], mesh_coords[:, 1]
    limit = max(float(max(np.max(np.abs(f)) for f in frames)), _DOPING_SYMLOG_LINTHRESH)
    norm = SymLogNorm(
        linthresh=_DOPING_SYMLOG_LINTHRESH, vmin=-limit, vmax=limit, base=10
    )

    # Running best of what the optimizer maximized: a frame below it is a
    # rejected trial (the optimizer only accepts improving steps).
    objectives = [float(h.get("objective", h["delta_n_eff"])) for h in history]
    best_so_far = -np.inf
    rejected: list[bool] = []
    for value in objectives:
        rejected.append(value < best_so_far)
        best_so_far = max(best_so_far, value)

    fig, ax = plt.subplots(figsize=(8.0, 3.6), layout="constrained")
    mesh = _pcolormesh_field(
        ax, x, y, frames[0], triangles=triangles, cmap="RdBu_r", norm=norm
    )
    fig.colorbar(mesh, ax=ax, label=r"Net doping [cm$^{-3}$]", shrink=0.9)
    ax.set_facecolor("0.92")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal")
    if geometry is not None:
        _overlay_geometry(ax, geometry)
    _zoom_to_silicon(ax, x, y, triangles)
    title = ax.set_title("", fontsize=11)
    # Axis limits are fixed by the first draw so the overlay and frame sizes
    # stay put; later frames only swap the painted field.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    def _caption(k: int) -> str:
        h = history[k]
        head = f"iteration {int(h['iteration'])}"
        if rejected[k]:
            head += "  (rejected trial)"
        detail = f"Δneff = {float(h['delta_n_eff']):+.3e}"
        loss = h.get("modal_loss_db_cm")
        if loss is not None and np.isfinite(loss):
            detail += f"   loss = {float(loss):.3g} dB/cm"
        return f"{head}\n{detail}"

    def _draw(k: int) -> tuple:
        nonlocal mesh
        mesh.remove()
        mesh = _pcolormesh_field(
            ax, x, y, frames[k], triangles=triangles, cmap="RdBu_r", norm=norm
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        title.set_text(_caption(k))
        title.set_color("#d7191c" if rejected[k] else "black")
        return (mesh, title)

    _draw(0)
    anim = animation.FuncAnimation(fig, _draw, frames=len(frames), blit=False)
    written: list[Path] = []
    for fmt in formats:
        writer_name = _ANIMATION_WRITERS.get(fmt)
        if writer_name is None:
            raise ValueError(f"unknown animation format {fmt!r}; expected gif or mp4")
        if not animation.writers.is_available(writer_name):
            print(
                f"      Animation {fmt} skipped: {writer_name} writer unavailable",
                flush=True,
            )
            continue
        path = out / f"{name}.{fmt}"
        anim.save(path, writer=animation.writers[writer_name](fps=fps), dpi=110)
        written.append(path)
    plt.close(fig)
    return written


@dataclass(frozen=True)
class OverlayGeometry:
    """The overlay frame of :func:`_overlay_geometry`, as plain values.

    ``RibWaveguideGeometry`` satisfies the same duck-typed contract but can
    only describe the mesh the *local* author builds (y from 0 at the
    substrate bottom, 0.5 µm substrate). The gyptis author centres its layer
    stack on y = 0 with a 0.35 µm substrate, so drawing the local frame on
    container figures put the rib outline off the device entirely.
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

    The geometry's dimensions are already micrometres, the same
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
            [cs, ce],
            sub_top,
            slab_top,
            color="gray",
            alpha=0.3,
            edgecolor="none",
        )

    ax.fill_between(
        [-hw, rib_l], slab_top, rib_top, color="gray", alpha=0.1, edgecolor="none"
    )
    ax.fill_between(
        [rib_r, hw], slab_top, rib_top, color="gray", alpha=0.1, edgecolor="none"
    )

    props = dict(
        boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"
    )
    ax.text(
        0.05,
        0.95,
        "Si core",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
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
    against the guiding core. ``mode_index`` is the guided-mode order the solve
    targeted (0 = fundamental), shown in the figure title.
    """

    abs_e: np.ndarray
    coords_um: np.ndarray
    rib_bounds: tuple[float, float, float, float] | None = None
    mode_index: int = 0

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
    the same cross-section as the doping figure.

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
    label = "fundamental" if mode.mode_index == 0 else f"index {mode.mode_index}"
    ax.set_title(rf"Optical mode $|E|$ (tracked, {label})")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal")

    if mode.rib_bounds is not None:
        x0, x1, y0, y1 = mode.rib_bounds
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            "w--",
            linewidth=0.8,
            alpha=0.8,
            label="Si rib",
        )
        ax.legend(loc="upper right", framealpha=0.6)

    fig.colorbar(mesh, ax=ax, label=r"$|E|$ (normalized)")
    fig.tight_layout()

    path = out / "mode_field.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_depletion_field(
    swept_carriers: np.ndarray,
    mesh_coords: np.ndarray,
    mode_field: ModeField | None = None,
    geometry: object | None = None,
    output_dir: str | Path | None = None,
    triangles: np.ndarray | None = None,
) -> Path:
    """Where the modulation happens: swept carriers under the optical mode.

    ``swept_carriers`` is ``(n + p)(V_bias) - (n + p)(0 V)`` per node [cm^-3]
    -- negative where reverse bias depletes the junction, which is the only
    place the design changes the index. Drawn on the silicon elements with the
    tracked mode's ``|E|`` contours on top, so the figure shows at a glance
    how much of the depleted region the mode actually sees (and how much
    doping sits in the mode for nothing but loss).

    Args:
        swept_carriers: ``(n_nodes,)`` change in total carrier density.
        mesh_coords: ``(n_nodes, 2)`` node coordinates in micrometres.
        mode_field: The tracked mode for the contour overlay; ``None`` draws
            the carriers alone.
        geometry: Overlay frame (see :func:`_overlay_geometry`); optional.
        output_dir: Directory to write ``depletion_field.pdf``.
        triangles: Silicon triangulation, as in :func:`plot_doping_field`.

    Returns:
        Path to the saved figure.
    """
    out = _ensure_output_dir(output_dir)
    swept = np.asarray(swept_carriers, dtype=float)
    x, y = mesh_coords[:, 0], mesh_coords[:, 1]
    limit = max(
        float(np.max(np.abs(swept))) if swept.size else 0.0, _DOPING_SYMLOG_LINTHRESH
    )

    fig, ax = plt.subplots(figsize=(8.0, 3.6), layout="constrained")
    mesh = _pcolormesh_field(
        ax,
        x,
        y,
        swept,
        triangles=triangles,
        cmap="PuOr",
        norm=SymLogNorm(
            linthresh=_DOPING_SYMLOG_LINTHRESH, vmin=-limit, vmax=limit, base=10
        ),
    )
    ax.set_facecolor("0.92")
    if mode_field is not None:
        mx, my = mode_field.coords_um[:, 0], mode_field.coords_um[:, 1]
        contours = ax.tricontour(
            mtri.Triangulation(mx, my),
            mode_field.abs_e,
            levels=[0.2, 0.4, 0.6, 0.8],
            colors="k",
            linewidths=0.8,
            alpha=0.8,
        )
        ax.clabel(contours, fmt="%.1f", fontsize=7)
    if geometry is not None:
        _overlay_geometry(ax, geometry)
    _zoom_to_silicon(ax, x, y, triangles, pad=0.3)
    ax.set_title(r"Swept carriers $\Delta(n+p)$ under the mode $|E|$")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal")
    fig.colorbar(
        mesh,
        ax=ax,
        shrink=0.9,
        label=r"$(n+p)(V_{bias}) - (n+p)(0\,\mathrm{V})$ [cm$^{-3}$]",
    )

    path = out / "depletion_field.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# Default acceptance bar for the composed cross-boundary gradient.
# Central differences of a smooth pipeline are O(h^2), but the *composed* adjoint
# is limited by the least-exact link in the chain: the CT discrete adjoint reads
# carrier density back through an internal 1e-8 finite difference and
# the gyptis eigen-adjoint carries its solver tolerance. 1% relative agreement at
# the best step is the "gradients do real work" bar -- tight enough to catch a
# sign error, a missing adjoint term, or a unit mismatch across the boundary,
# loose enough not to chase the CT readout's own FD floor.
GRADIENT_VALIDATION_TOLERANCE = 1e-2


@dataclass(frozen=True)
class GradientValidationResult:
    """Outcome of an adjoint-vs-finite-difference gradient check.

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
    before_evaluation: Callable[[], None] | None = None,
) -> list[tuple[int, np.ndarray, list[float]]]:
    """Central-FD relative-error curve per direction against the JAX adjoint.

    Returns ``(index, feasible_steps, errors)`` for every direction that has at
    least one feasible step. The adjoint's directional derivative is
    ``grad · direction``; the finite-difference estimate is the central
    difference at each step. ``before_evaluation`` runs before every pipeline
    evaluation (the adjoint one included) -- the cold-start hook,
    which resets the ChargeTransport worker so no FD sample inherits the
    previous sample's Newton starting point.
    """

    def _prepare() -> None:
        if before_evaluation is not None:
            before_evaluation()

    _prepare()
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
            _prepare()
            f_plus = pipeline_fn(rho + h * direction)
            _prepare()
            f_minus = pipeline_fn(rho - h * direction)
            fd_val = (f_plus - f_minus) / (2.0 * h)
            errors.append(float(abs(fd_val - exact_val)) / denom)
        curves.append((i, feasible_steps, errors))
    return curves


def _observed_orders(
    feasible_steps: np.ndarray, errors: list[float]
) -> list[tuple[float, float]]:
    """Local log-log slopes of a relative-error curve, one per adjacent pair.

    Each entry is ``(h_right, slope)`` for the interval ending at ``h_right``.
    On the truncation branch of a central difference the slope approaches +2;
    on the cancellation branch, where the evaluation noise floor divided by the
    step dominates, it approaches -1. Reporting the measured slope is what makes
    the figure a convergence test rather than an eyeball comparison against a
    guide line drawn with an arbitrary constant.
    """
    orders: list[tuple[float, float]] = []
    for left in range(len(feasible_steps) - 1):
        h0, h1 = float(feasible_steps[left]), float(feasible_steps[left + 1])
        e0, e1 = errors[left], errors[left + 1]
        if h0 <= 0.0 or h1 <= 0.0 or e0 <= 0.0 or e1 <= 0.0:
            continue
        orders.append((h1, math.log(e1 / e0) / math.log(h1 / h0)))
    return orders


def _steepest_observed_order(
    curves: list[tuple[int, np.ndarray, list[float]]],
) -> float | None:
    """The largest local slope seen on any direction's curve.

    The truncation branch is the steep, rising part of the V; its slope is the
    observed order of the finite difference. ``None`` when no curve has two
    usable points.
    """
    slopes = [
        slope
        for _, feasible_steps, errors in curves
        for _, slope in _observed_orders(feasible_steps, errors)
    ]
    return max(slopes) if slopes else None


def _render_gradient_validation(
    curves: list[tuple[int, np.ndarray, list[float]]],
    step_sizes: np.ndarray,
    n_directions: int,
    tolerance: float | None,
    out: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    anchors: list[tuple[float, float]] = []
    for i, feasible_steps, errors in curves:
        label = f"direction {i + 1}" if n_directions > 1 else "FD"
        ax.loglog(feasible_steps, errors, "o-", markersize=3, label=label)
        best = int(np.argmin(errors))
        anchors.append((float(feasible_steps[best]), errors[best]))

    # Central differences of a smooth pipeline converge at second order (see
    # GRADIENT_VALIDATION_TOLERANCE above), so the guide line is h^2, not h. It
    # is anchored to the best point of the first curve rather than drawn with a
    # unit constant: only its slope carries meaning, and an arbitrary offset
    # invites the reader to compare the wrong thing.
    if anchors:
        h_anchor, e_anchor = anchors[0]
        ax.plot(
            step_sizes,
            e_anchor * (step_sizes / h_anchor) ** 2,
            "k--",
            alpha=0.4,
            label=r"$O(h^2)$",
        )
    order = _steepest_observed_order(curves)
    if order is not None:
        ax.text(
            0.03,
            0.03,
            f"steepest observed order: {order:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.75,
        )
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
    before_evaluation: Callable[[], None] | None = None,
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

    curves = _gradient_validation_curves(
        pipeline_fn, rho, directions, step_sizes, before_evaluation
    )
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
    return _render_gradient_validation(curves, step_sizes, n_directions, tolerance, out)


def validate_gradient(
    pipeline_fn: Callable[..., jax.Array],
    rho: jax.Array,
    directions: list[jax.Array] | None = None,
    n_directions: int = 3,
    step_sizes: np.ndarray | None = None,
    tolerance: float = GRADIENT_VALIDATION_TOLERANCE,
    output_dir: str | Path | None = None,
    before_evaluation: Callable[[], None] | None = None,
) -> GradientValidationResult:
    """Check the adjoint against central FD and gate on a stated tolerance.

    The proof behind the headline: the composed ``rho -> Delta n_eff``
    gradient is exercised through whatever ``pipeline_fn`` closes over -- pass a
    ``pipeline`` bound to the live ChargeTransport and gyptis components to prove
    the cross-boundary adjoint (CT included), not a stub. The figure is
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
        before_evaluation: Hook run before every pipeline evaluation (the
            cold-start option: reset the ChargeTransport worker so the FD
            reference is not polluted by warm-start path dependence).

    Returns:
        A :class:`GradientValidationResult` with the per-direction best errors,
        the worst of them, the pass/fail verdict, and the figure path.
    """
    out, curves, step_sizes, n_directions = _gradient_validation_setup(
        pipeline_fn,
        rho,
        directions,
        n_directions,
        step_sizes,
        output_dir,
        before_evaluation,
    )

    best_rel_errors = [min(errors) for _, _, errors in curves]
    worst_rel_error = max(best_rel_errors)
    path = _render_gradient_validation(curves, step_sizes, n_directions, tolerance, out)
    return GradientValidationResult(
        n_directions=len(curves),
        best_rel_errors=best_rel_errors,
        worst_rel_error=worst_rel_error,
        tolerance=tolerance,
        passed=worst_rel_error <= tolerance,
        figure_path=path,
    )


# ---------------------------------------------------------------------------
# Objective line scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveLineScan:
    """``f(rho + t*d)`` sampled along one direction at uniform spacing.

    The smoothness probe: the optimizer's trials around an earlier
    optimum sat ~2e-3 relative below the iterate regardless of how
    small the box became, and this scan separates a genuine kink (the residual
    of a quadratic fit is structured and shrinks with the spacing) from an
    evaluation noise floor (the residual is white and does not). All relative
    quantities are relative to ``|f(rho)|`` at the centre sample.

    ``noise_estimate`` is the white-noise amplitude implied by the second
    differences of the samples, ``std(diff(values, 2)) / sqrt(6)``, relative to
    ``|f(rho)|``: second differences of a smooth function at fine spacing are
    ~0, so whatever is left is the per-evaluation scatter.
    """

    offsets: np.ndarray
    values: np.ndarray
    fit_coefficients: np.ndarray
    residuals: np.ndarray
    rms_rel_residual: float
    max_rel_residual: float
    noise_estimate: float
    adjoint_slope: float
    fitted_slope: float
    figure_path: Path | None
    json_path: Path | None


def feasible_offsets(
    rho: jax.Array, direction: jax.Array, offsets: np.ndarray
) -> np.ndarray:
    """The subset of signed ``offsets`` ``t`` with ``rho + t*direction`` in ``[-1, 1]``.

    Unlike :func:`_feasible_steps` (which bounds a symmetric central stencil)
    each side is bounded on its own, so a design pinned at +1 in the direction's
    sense still keeps its whole negative half-line.
    """
    rho_np = np.asarray(rho, dtype=float)
    dir_np = np.asarray(direction, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    pos = dir_np > 0.0
    neg = dir_np < 0.0
    t_max = min(
        np.min((1.0 - rho_np[pos]) / dir_np[pos], initial=np.inf),
        np.min((-1.0 - rho_np[neg]) / dir_np[neg], initial=np.inf),
    )
    t_min = max(
        np.max((-1.0 - rho_np[pos]) / dir_np[pos], initial=-np.inf),
        np.max((1.0 - rho_np[neg]) / dir_np[neg], initial=-np.inf),
    )
    return offsets[(offsets >= t_min) & (offsets <= t_max)]


def probe_direction(
    pipeline_fn: Callable[..., jax.Array],
    rho: jax.Array,
    kind: str,
    before_evaluation: Callable[[], None] | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Unit direction for a line scan through ``rho``, plus the gradient used.

    ``"gradient"`` is the normalized adjoint gradient at ``rho`` (the direction
    the optimizer would step); ``"random"`` is the first seeded unit vector
    :func:`validate_gradient` also samples. Either way the direction is
    projected off rail-pinned variables so the scan can move both ways inside
    the ``[-1, 1]`` box. Returns ``(direction, gradient)`` -- the gradient is
    ``None`` for a random direction -- so :func:`scan_objective_line` can take
    its adjoint slope from the same adjoint solve instead of running a second
    one. Raises ``ValueError`` for an unknown ``kind`` and ``RuntimeError`` when
    there is nothing to scan (zero gradient, or every weighted variable pinned).
    """
    rho = jnp.asarray(rho)
    gradient = None
    if kind == "gradient":
        if before_evaluation is not None:
            before_evaluation()
        gradient = jax.grad(pipeline_fn)(rho)
        d = gradient
        if float(jnp.linalg.norm(d)) == 0.0:
            raise RuntimeError(
                "the gradient is identically zero at the centre design; scan a "
                "random direction instead (--direction random)"
            )
    elif kind == "random":
        d = _sample_directions(rho, None, 1)[0]
    else:
        raise ValueError("direction must be 'gradient' or 'random'")
    interior = _interior_direction(rho, d / jnp.linalg.norm(d))
    if interior is None:
        raise RuntimeError("every variable with direction weight is rail-pinned")
    return interior, gradient


def _render_objective_line_scan(
    values: np.ndarray,
    offsets: np.ndarray,
    fit: np.ndarray,
    residuals: np.ndarray,
    out: Path,
) -> Path:
    fig, (ax_f, ax_r) = plt.subplots(
        2, 1, figsize=(6, 6), sharex=True, height_ratios=[2, 1]
    )
    ax_f.plot(offsets, values, "o", markersize=4, label="f(θ₀ + t·d)")
    ax_f.plot(offsets, np.polyval(fit, offsets), "-", alpha=0.7, label="quadratic fit")
    ax_f.set_ylabel(r"$\Delta n_{\rm eff}$")
    ax_f.set_title("Objective line scan")
    ax_f.legend()
    ax_f.grid(True, alpha=0.3)
    ax_r.plot(offsets, residuals, "o-", markersize=3, color="crimson")
    ax_r.axhline(0.0, color="k", lw=0.8)
    ax_r.set_xlabel("t (along the unit direction d)")
    ax_r.set_ylabel("residual")
    ax_r.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "objective_line_scan.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def scan_objective_line(
    pipeline_fn: Callable[..., jax.Array],
    rho: jax.Array,
    direction: jax.Array,
    offsets: np.ndarray,
    output_dir: str | Path | None = None,
    before_evaluation: Callable[[], None] | None = None,
    gradient: jax.Array | None = None,
) -> ObjectiveLineScan:
    """Sample ``pipeline_fn`` along ``direction`` and fit a quadratic.

    Args:
        pipeline_fn: Callable ``(rho) -> scalar`` differentiable by JAX.
        rho: Centre design.
        direction: Unit direction ``d``.
        offsets: Signed steps ``t``; every ``rho + t*d`` is evaluated as given
            (filter with :func:`feasible_offsets` first to respect the box).
        output_dir: When given, write ``objective_line_scan.pdf`` and
            ``objective_line_scan.json`` there.
        before_evaluation: Hook run before the gradient and before every
            sample.
        gradient: The adjoint gradient at ``rho`` if the caller already has it
            (e.g. from :func:`probe_direction`); computed here when ``None``.

    Returns:
        An :class:`ObjectiveLineScan`.
    """
    rho = jnp.asarray(rho)
    direction = jnp.asarray(direction, dtype=rho.dtype)
    offsets = np.asarray(offsets, dtype=float)
    if offsets.size < 3:
        raise ValueError("an objective line scan needs at least 3 offsets")

    def _prepare() -> None:
        if before_evaluation is not None:
            before_evaluation()

    if gradient is None:
        _prepare()
        gradient = jax.grad(pipeline_fn)(rho)
    adjoint_slope = float(jnp.dot(jnp.asarray(gradient), direction))

    values = np.empty(offsets.shape, dtype=float)
    for i, t in enumerate(offsets):
        _prepare()
        values[i] = float(pipeline_fn(rho + t * direction))

    fit = np.polyfit(offsets, values, 2)
    residuals = values - np.polyval(fit, offsets)
    centre = int(np.argmin(np.abs(offsets)))
    scale = max(abs(float(values[centre])), 1e-30)
    rms_rel = float(np.sqrt(np.mean(residuals**2))) / scale
    max_rel = float(np.max(np.abs(residuals))) / scale
    noise = float(np.std(np.diff(values, n=2)) / np.sqrt(6.0)) / scale

    figure_path = json_path = None
    if output_dir is not None:
        out = _ensure_output_dir(output_dir)
        figure_path = _render_objective_line_scan(values, offsets, fit, residuals, out)
        json_path = out / "objective_line_scan.json"
        json_path.write_text(
            json.dumps(
                {
                    "offsets": offsets.tolist(),
                    "values": values.tolist(),
                    "fit_coefficients": fit.tolist(),
                    "residuals": residuals.tolist(),
                    "rms_rel_residual": rms_rel,
                    "max_rel_residual": max_rel,
                    "noise_estimate": noise,
                    "adjoint_slope": adjoint_slope,
                    "fitted_slope": float(fit[1]),
                },
                indent=2,
            )
        )

    return ObjectiveLineScan(
        offsets=offsets,
        values=values,
        fit_coefficients=fit,
        residuals=residuals,
        rms_rel_residual=rms_rel,
        max_rel_residual=max_rel,
        noise_estimate=noise,
        adjoint_slope=adjoint_slope,
        fitted_slope=float(fit[1]),
        figure_path=figure_path,
        json_path=json_path,
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
    cold_reevaluation: ColdReevaluation | None = None,
    design_history: Sequence[np.ndarray] | None = None,
    animation_history: list[dict] | None = None,
    animation_formats: Sequence[str] = ("gif", "mp4"),
    swept_carriers: np.ndarray | None = None,
    bias_sweep_initial: Sequence[Any] | None = None,
    bias_sweep_optimized: Sequence[Any] | None = None,
) -> list[Path]:
    """Generate the headline output figures.

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
        cold_reevaluation: Warm/cold Δneff of the reported design, annotated
            on the convergence figure. ``None`` draws nothing.
        design_history: Full-node net doping [cm^-3] per evaluated design, in
            history order; animated into ``doping_evolution.{gif,mp4}`` (see
            :func:`animate_doping_evolution`). ``None`` skips the animation.
        animation_history: The records matching ``design_history`` one-to-one;
            defaults to ``history`` (use it when only some records carry a
            design).
        animation_formats: Encoders to try for the animation.
        swept_carriers: ``(n + p)(V_bias) - (n + p)(0 V)`` per node at the
            optimized design, for :func:`plot_depletion_field`; ``None``
            skips that figure.
        bias_sweep_initial: ``BiasPoint`` sequence from
            :func:`~prismo.pipeline.bias_sweep` for the seed design, for
            :func:`plot_bias_sweep`.
        bias_sweep_optimized: The same for the reported design. The figure is
            skipped when both are empty.

    Returns:
        List of paths to the generated plot files.
    """
    out = _ensure_output_dir(output_dir)
    paths: list[Path] = []

    conv = plot_convergence(
        history,
        output_dir=out,
        ftol_rel=ftol_rel,
        cold_reevaluation=cold_reevaluation,
    )
    paths.append(conv)
    # Loss-aware run: the loss history and the trade-off plane.
    for loss_figure in (plot_loss_convergence, plot_tradeoff):
        loss_path = loss_figure(history, output_dir=out)
        if loss_path is not None:
            paths.append(loss_path)

    sweep = plot_bias_sweep(bias_sweep_initial, bias_sweep_optimized, output_dir=out)
    if sweep is not None:
        paths.append(sweep)

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

    if swept_carriers is not None:
        paths.append(
            plot_depletion_field(
                swept_carriers,
                mesh_coords,
                mode_field=mode_field,
                geometry=geometry,
                output_dir=out,
                triangles=mesh_triangles,
            )
        )

    if design_history:
        try:
            paths.extend(
                animate_doping_evolution(
                    design_history,
                    history if animation_history is None else animation_history,
                    mesh_coords,
                    geometry=geometry,
                    output_dir=out,
                    triangles=mesh_triangles,
                    formats=animation_formats,
                )
            )
        except Exception as exc:
            print(f"WARNING: doping animation skipped ({exc})", flush=True)

    return paths
