"""Move-limited MMA optimization loop for the PRISMO pipeline.

Wraps the JAX-differentiable pipeline in NLopt's MMA, one fresh MMA subproblem
per outer step inside a move-limit box.
Design variables: the signed design field theta in [-1, 1], one entry per
design node (the silicon nodes of the shared mesh; see ``pipeline.DesignNodes``).
Objective: maximize signed delta_n_eff, optionally minus a weighted modal
free-carrier loss.
"""

from __future__ import annotations

import json
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import nlopt
import numpy as np
from prismo_shared.schemas import MeshRef

from prismo.density_filter import assemble_filter_matrix
from prismo.pipeline import (
    DesignNodes,
    PipelineComponents,
    default_components,
    pipeline_with_terms,
)

_HistoryEntry = dict[str, Any]

# Below this the objective is treated as numerically indistinguishable from
# zero and relative measures (ftol, gradient scaling) fall back to absolute ones
# rather than amplifying noise.
_OBJECTIVE_SCALE_FLOOR = 1e-30

# Per-iteration move limit on theta. Every outer step runs MMA
# inside the box ``x ± move_limit`` intersected with ``[-1, 1]``, so no design
# variable moves more than this in one iteration whatever MMA's asymptotes
# would do. With NLopt's unconstrained MMA the trust region grew by 1.2x per
# consistent iteration while the conservativeness decayed 10x, and from
# iteration ~8 every step moved every node 0.9 sigma: 13-16 junction flips per
# iteration, max |N| from 3e17 to the 9.9e18 rail in ten iterations, and an
# unsolvable design at iteration 11.
DEFAULT_MOVE_LIMIT = 0.05

# A failed physics solve means "step too large": the move limit is halved and
# the step re-proposed from the same iterate. Likewise a step that does not
# improve the objective. This bounds the consecutive halvings before the loop
# declares the iterate stalled (0.05 / 2^8 ~ 2e-4 on theta, well below any
# step the physics can resolve).
DEFAULT_MAX_MOVE_HALVINGS = 8

# NLopt's MMA sizes its subproblem move from the gradient relative to its
# initial conservativeness ``rho = 1``: the step is ~sigma for ``|g|·sigma >>
# 1`` and ~``|g|·sigma``-proportional below. The physical gradient (~1e-4 per
# node) would produce a first step of ~1e-7 and the design would never move, so
# the objective handed to each fresh MMA instance is scaled so that the
# steepest variable sees ``|g|·move_limit`` equal to this target -- a
# normalised gradient-weighted step that reaches ~70 % of the box for the
# steepest node and proportionally less for the rest. The maximiser of a
# positively-scaled objective is unchanged; history and printing stay physical.
_MMA_GRADIENT_SCALE_TARGET = 10.0

# One trial evaluation per outer MMA step: the instance evaluates the (cached)
# iterate, proposes one point inside the box, and returns. Accept/shrink is
# decided here, not by MMA's inner conservativeness loop, so every physics
# solve is one optimizer iteration.
_TRIALS_PER_STEP = 1


class _TrialFailed(Exception):
    """Internal: the physics solve at the proposed trial point raised.

    NLopt has no way to reject a trial point, and feeding MMA a fabricated
    penalty poisons its asymptote update (observed driving a run from
    Delta_neff=+3.4e-4 to a wrong-polarity -3.8e-4). The step is instead
    treated as too large: the MMA instance is abandoned, the move limit halved,
    and a new step proposed from the same iterate.
    """


class OptimizationCancelled(Exception):
    """Raised when the user interrupts the optimization loop."""


def optimize_doping(
    initial_rho: np.ndarray | None = None,
    n_nodes: int | None = None,
    H: jax.Array | None = None,
    H_sum: jax.Array | None = None,
    mesh_coords: np.ndarray | None = None,
    mesh_path: str | Path | None = None,
    r_min: float = 0.05,
    *,
    max_iter: int = 200,
    ftol_rel: float = 1e-3,
    move_limit: float = DEFAULT_MOVE_LIMIT,
    max_move_halvings: int = DEFAULT_MAX_MOVE_HALVINGS,
    checkpoint_path: str | Path | None = None,
    use_jit: bool = True,
    on_iteration: Callable[[int, np.ndarray], None] | None = None,
    design_transfer: jax.Array | None = None,
    design_nodes: DesignNodes | None = None,
    mesh_ref: MeshRef | None = None,
    components: PipelineComponents | None = None,
    loss_weight: float = 0.0,
    mode_overlap: np.ndarray | jax.Array | None = None,
) -> tuple[np.ndarray, list[_HistoryEntry]]:
    """Run the move-limited MMA optimization loop.

    Args:
        initial_rho: Starting signed design field ``(n_design,)`` in ``[-1, 1]``
            -- one entry per design node, which on a real mesh is the silicon
            nodes rather than every mesh node. Defaults to a uniform 0.25
            fallback; the real run seeds a signed junction (``main.py``). The
            name is retained for call-site compatibility.
        n_nodes: Number of design variables (derived from ``initial_rho`` if
            given, else required when ``H`` / ``mesh_coords`` / ``mesh_path``
            is provided).
        H: Dense filter matrix ``(n_design, n_design)``. Built from
            ``mesh_coords`` or ``mesh_path`` if omitted.
        H_sum: Pre-computed row sums of ``H``.
        mesh_coords: ``(n_design, 2)`` design-node coordinates for building the
            filter matrix.
        mesh_path: Path to a ``.msh`` file for building the filter matrix
            (requires ``gmsh``).
        r_min: Filter radius in micrometres -- the unit the shared mesh's
            coordinates are authored in (default 0.05, i.e. 50 nm).
        max_iter: Maximum number of physics evaluations (optimizer iterations).
        ftol_rel: Relative tolerance on the objective: the loop stops once an
            accepted full-move-limit step improves the objective by less than
            this fraction.
        move_limit: Per-iteration move limit on theta: no design variable
            moves more than this in one iteration.
        max_move_halvings: Consecutive halvings of the move limit (after failed
            or non-improving steps) before the iterate is declared stalled.
        checkpoint_path: Where to write ``{"rho_opt", "history", ...}`` after
            every evaluation, so a killed run still yields a figure. ``None``
            writes no checkpoint.
        use_jit: JIT-compile combined objective and gradient computation.
        on_iteration: Optional callback receiving an iteration number and its
            candidate design field immediately before its solver evaluation.
        design_transfer: Dense ``(n_design_cells, n_nodes)`` mesh-transfer
            matrix carrying the nodal perturbation onto the gyptis design
            cells. ``None`` maps the perturbation node-for-node (identity).
        design_nodes: Which shared-mesh nodes the design vector addresses (the
            silicon nodes on a real mesh). Forwarded to ``pipeline``, which
            scatters the filtered field back to full node order. ``None`` means
            one variable per mesh node.
        mesh_ref: Shared-mesh reference forwarded to the ChargeTransport
            solves so they run on the real 2D grid. ``None`` leaves CT on its
            1D fallback device.
        components: Live pipeline components to compose. Defaults to the
            in-process components (see ``pipeline``).
        loss_weight: Weight of the modal free-carrier loss in the objective
            ``delta_neff - loss_weight * loss_db_cm``. ``0`` keeps
            the pure Δneff objective; the loss is still recorded when
            ``mode_overlap`` is given.
        mode_overlap: ``(n_design_cells,)`` mode-overlap weights from
            ``pipeline.read_mode_overlap``; required for a positive
            ``loss_weight``.

    Returns:
        ``(rho_opt, history)`` where ``rho_opt`` is the best design whose
        physics solved and ``history`` is a list of per-evaluation records:
        ``objective`` (what MMA maximized), ``delta_n_eff``,
        ``modal_loss_db_cm`` (``None`` when not evaluated), step and timing
        fields.

    Raises:
        OptimizationCancelled: If the user interrupts with Ctrl-C.
        ValueError: On a non-positive move limit or missing sizing inputs.
    """
    if move_limit <= 0.0:
        raise ValueError("move_limit must be positive")
    if initial_rho is None:
        if n_nodes is None:
            raise ValueError("initial_rho or n_nodes must be provided")
        initial_rho = np.full(n_nodes, 0.25, dtype=float)
    else:
        initial_rho = np.asarray(initial_rho, dtype=float)
        n_nodes = len(initial_rho)

    if H is None and (mesh_coords is not None or mesh_path is not None):
        if mesh_path is not None:
            from prismo.waveguide_mesh import read_mesh_node_coordinates

            mesh_coords = read_mesh_node_coordinates(mesh_path)
            if mesh_coords.shape[0] == 0:
                raise ValueError(f"Could not read node coordinates from {mesh_path}")
        if mesh_coords is None:
            raise ValueError("mesh_coords or mesh_path required to build filter matrix")
        H_sparse = assemble_filter_matrix(mesh_coords, r_min=r_min)
        H = jnp.asarray(H_sparse.toarray())
        H_sum = jnp.sum(H, axis=1)

    if components is None:
        components = default_components()

    def _pipe(rho: jax.Array) -> tuple[jax.Array, Any]:
        return pipeline_with_terms(
            rho,
            H=H,
            H_sum=H_sum,
            mesh_ref=mesh_ref,
            design_transfer=design_transfer,
            design_nodes=design_nodes,
            components=components,
            loss_weight=loss_weight,
            mode_overlap=mode_overlap,
        )

    value_and_grad_fn = jax.value_and_grad(_pipe, has_aux=True)
    if use_jit:
        value_and_grad_fn = jax.jit(value_and_grad_fn)

    history: list[_HistoryEntry] = []
    cancelled: list[bool] = [False]
    t_start = time.perf_counter()
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None

    def _sigint_handler(signum: int, frame: Any) -> None:
        cancelled[0] = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    # The current iterate and its physics; ``x`` is always the best design
    # whose physics solved, since a step is only accepted when it improves.
    x = initial_rho.copy()
    f_x = 0.0
    g_x = np.zeros(n_nodes)
    delta = float(move_limit)
    halvings = 0
    stop_reason = "max_iter reached"

    def _evaluate(
        rho_np: np.ndarray, step_from: np.ndarray | None
    ) -> tuple[float, np.ndarray]:
        """One physics evaluation: history, printing, checkpoint. Raises on failure."""
        callback_started_at = time.perf_counter()
        iter_count = len(history) + 1
        if on_iteration is not None:
            on_iteration(iter_count, rho_np.copy())
        (value, terms), grad = value_and_grad_fn(jnp.asarray(rho_np))
        f_val = float(value)
        delta_neff = float(terms.delta_neff)
        loss_db_cm = float(terms.modal_loss_db_cm)
        grad_phys = np.asarray(grad, dtype=float)
        callback_time = time.perf_counter() - callback_started_at

        delta_rho = 0.0
        max_step = 0.0
        if step_from is not None:
            delta_rho = float(np.linalg.norm(rho_np - step_from))
            max_step = float(np.max(np.abs(rho_np - step_from)))
        g_norm = float(np.linalg.norm(grad_phys))
        wall = time.perf_counter() - t_start

        # ``objective`` is what MMA maximizes (``delta_n_eff`` when the loss
        # weight is zero); the two physical terms ride alongside so the
        # trade-off is visible per iteration. A loss that was not evaluated
        # (no mode-overlap weights) is ``None``, which JSON can carry.
        # ``design`` is the evaluated design vector itself, so the checkpoint
        # can replay the run (``prismo animate``) without re-solving anything;
        # a failed solve writes no record, so every record has its design.
        history.append(
            {
                "iteration": iter_count,
                "objective": f_val,
                "delta_n_eff": delta_neff,
                "modal_loss_db_cm": loss_db_cm if np.isfinite(loss_db_cm) else None,
                "design": np.asarray(rho_np, dtype=float).tolist(),
                "delta_rho": delta_rho,
                "max_step": max_step,
                "move_limit": delta,
                "grad_norm": g_norm,
                "wall_time": wall,
                "callback_time": callback_time,
            }
        )
        loss_text = f"alpha={loss_db_cm:.4g}dB/cm  " if np.isfinite(loss_db_cm) else ""
        objective_text = f"f={f_val:+.6e}  " if loss_weight > 0.0 else ""
        print(
            f"iter {iter_count:4d}  "
            f"{objective_text}"
            f"Δneff={delta_neff:+.6e}  "
            f"{loss_text}"
            f"‖Δρ‖={delta_rho:.4e}  "
            f"max|Δθ|={max_step:.3e}  "
            f"Δ={delta:.3e}  "
            f"‖∇f‖={g_norm:.4e}  "
            f"callback={callback_time:.1f}s  "
            f"wall={wall:.1f}s",
            # Each iteration is minutes of solver time; without an explicit
            # flush the block-buffered stream hides all progress until exit.
            flush=True,
        )
        return f_val, grad_phys

    def _checkpoint() -> None:
        if checkpoint is not None:
            _save_checkpoint(x, history, checkpoint, move_limit=delta)

    def _step_objective(
        x_k: np.ndarray,
        f_k: float,
        g_k: np.ndarray,
        scale: float,
        trial: list[tuple[np.ndarray, float, np.ndarray]],
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        """NLopt objective for one outer step from the iterate ``x_k``.

        Serves the iterate itself from cache, evaluates the physics at the one
        trial point MMA proposes (recording it in ``trial``), and converts a
        failed solve into ``_TrialFailed`` so the step can be shrunk.
        """

        def _obj(rho_np: np.ndarray, grad_out: np.ndarray) -> float:
            if cancelled[0]:
                grad_out[:] = 0.0
                return 0.0
            if np.array_equal(rho_np, x_k):
                grad_out[:] = g_k * scale
                return f_k * scale
            try:
                f_val, grad_phys = _evaluate(rho_np, x_k)
            except Exception as exc:
                raise _TrialFailed(
                    f"physics solve failed at evaluation {len(history) + 1} "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
            finally:
                _checkpoint()
            trial.append((rho_np.copy(), f_val, grad_phys))
            grad_out[:] = grad_phys * scale
            return f_val * scale

        return _obj

    try:
        # The seed must solve: with nothing feasible to fall back to, surface
        # the failure rather than pretending the run can continue.
        f_x, g_x = _evaluate(x, None)
        _checkpoint()

        while len(history) < max_iter:
            if cancelled[0]:
                stop_reason = "cancelled"
                break

            # Fresh MMA subproblem inside the move-limit box. The
            # objective is scaled so the steepest variable sees an O(10)
            # gradient over the box; see ``_MMA_GRADIENT_SCALE_TARGET``.
            g_max = float(np.max(np.abs(g_x))) if g_x.size else 0.0
            scale = (
                _MMA_GRADIENT_SCALE_TARGET / (g_max * delta)
                if g_max * delta > _OBJECTIVE_SCALE_FLOOR
                else 1.0
            )
            lower = np.maximum(x - delta, -1.0)
            upper = np.minimum(x + delta, 1.0)
            trial: list[tuple[np.ndarray, float, np.ndarray]] = []

            opt = nlopt.opt(nlopt.LD_MMA, n_nodes)
            opt.set_lower_bounds(lower)
            opt.set_upper_bounds(upper)
            opt.set_max_objective(_step_objective(x, f_x, g_x, scale, trial))
            opt.set_maxeval(1 + _TRIALS_PER_STEP)
            try:
                opt.optimize(x.copy())
            except _TrialFailed as exc:
                print(
                    f"      {exc}; halving the move limit to {delta / 2:.3e} and "
                    "re-proposing from the current design.",
                    flush=True,
                )
            except nlopt.RoundoffLimited:
                # The subproblem could not resolve a step at this scale; shrink
                # like a non-improving step.
                print(
                    f"      MMA subproblem roundoff-limited at move limit {delta:.3e}; "
                    "halving.",
                    flush=True,
                )

            if cancelled[0]:
                stop_reason = "cancelled"
                break

            improved = [t for t in trial if t[1] > f_x]
            if improved:
                x_new, f_new, g_new = max(improved, key=lambda t: t[1])
                rel_gain = (f_new - f_x) / max(abs(f_x), _OBJECTIVE_SCALE_FLOOR)
                x, f_x, g_x = x_new, f_new, g_new
                halvings = 0
                full_step = delta >= float(move_limit)
                delta = min(delta * 2.0, float(move_limit))
                _checkpoint()
                # ftol only judges a step taken at the full move limit: a tiny
                # gain from a freshly-halved box says the box is small, not that
                # the objective has converged.
                if full_step and rel_gain < ftol_rel:
                    stop_reason = f"converged: relative gain {rel_gain:.2e} < ftol_rel"
                    break
            else:
                halvings += 1
                if halvings > max_move_halvings:
                    stop_reason = (
                        f"stalled: no improving step after {max_move_halvings} "
                        f"move-limit halvings (move limit {delta:.3e})"
                    )
                    break
                delta /= 2.0
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if cancelled[0]:
        _checkpoint()
        where = f" Progress saved to {checkpoint}." if checkpoint is not None else ""
        raise OptimizationCancelled(
            f"Optimization interrupted by user at iteration {len(history)}.{where}"
        )

    print(f"      Optimization ended: {stop_reason}.", flush=True)
    return x, history


def _save_checkpoint(
    rho: np.ndarray,
    history: list[_HistoryEntry],
    path: str | Path = Path("outputs") / "checkpoint.json",
    *,
    move_limit: float | None = None,
) -> None:
    """Save optimization progress (best design + history) to ``path``.

    Written after every evaluation, so a killed run still yields a
    convergence figure and a design to plot; each history record carries its
    evaluated ``design``, so ``prismo animate`` can replay the run from the
    checkpoint alone.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "rho_opt": np.asarray(rho, dtype=float).tolist(),
        "history": [{k: v for k, v in entry.items()} for entry in history],
    }
    if move_limit is not None:
        checkpoint["move_limit"] = float(move_limit)
    path.write_text(json.dumps(checkpoint, indent=2))
