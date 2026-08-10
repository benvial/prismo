"""NLopt MMA optimization loop for the PRISMO pipeline.

Wraps the JAX-differentiable pipeline in an NLopt MMA optimizer.
Design variables: per-node rho in [0, 1].
Objective: maximize delta_n_eff (or minimize -delta_n_eff).
Ref: ticket 15.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import nlopt
import numpy as np

from prismo.density_filter import assemble_filter_matrix
from prismo.pipeline import pipeline

_HistoryEntry = dict[str, Any]


class OptimizationCancelled(Exception):
    """Raised when the user interrupts the optimization loop."""


def optimize_doping(
    initial_rho: np.ndarray | None = None,
    n_nodes: int | None = None,
    H: jax.Array | None = None,
    H_sum: jax.Array | None = None,
    mesh_coords: np.ndarray | None = None,
    mesh_path: str | Path | None = None,
    r_min: float = 50e-9,
    *,
    max_iter: int = 200,
    ftol_rel: float = 1e-5,
    use_jit: bool = True,
) -> tuple[np.ndarray, list[_HistoryEntry]]:
    """Run the NLopt MMA optimization loop.

    Args:
        initial_rho: Starting design vector ``(n_nodes,)`` in ``[0, 1]``.
            Defaults to uniform 0.25.
        n_nodes: Number of mesh nodes (derived from ``initial_rho`` if given,
            else required when ``H`` / ``mesh_coords`` / ``mesh_path`` is
            provided).
        H: Dense filter matrix ``(n_nodes, n_nodes)``. Built from
            ``mesh_coords`` or ``mesh_path`` if omitted.
        H_sum: Pre-computed row sums of ``H``.
        mesh_coords: ``(n_nodes, 2)`` node coordinates for building the
            filter matrix.
        mesh_path: Path to a ``.msh`` file for building the filter matrix
            (requires ``gmsh``).
        r_min: Filter radius in meters (default 50 nm).
        max_iter: Maximum MMA iterations.
        ftol_rel: Relative tolerance on the objective for early stopping.
        use_jit: JIT-compile the gradient computation.

    Returns:
        ``(rho_opt, history)`` where ``rho_opt`` is the optimized design
        vector and ``history`` is a list of per-iteration records.

    Raises:
        OptimizationCancelled: If the user interrupts with Ctrl-C.
    """
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
                raise ValueError(
                    f"Could not read node coordinates from {mesh_path}"
                )
        if mesh_coords is None:
            raise ValueError(
                "mesh_coords or mesh_path required to build filter matrix"
            )
        H_sparse = assemble_filter_matrix(mesh_coords, r_min=r_min)
        H = jnp.asarray(H_sparse.toarray())
        H_sum = jnp.sum(H, axis=1)

    def _pipe(rho: jax.Array) -> jax.Array:
        return pipeline(rho, H=H, H_sum=H_sum)

    _pipe_jit = jax.jit(_pipe) if use_jit else _pipe
    grad_fn = jax.grad(_pipe_jit)
    if use_jit:
        grad_fn = jax.jit(grad_fn)

    history: list[_HistoryEntry] = []
    prev_rho: np.ndarray | None = None
    cancelled: list[bool] = [False]
    t_start = time.perf_counter()

    def _sigint_handler(signum: int, frame: Any) -> None:
        cancelled[0] = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    try:
        def _obj(rho_np: np.ndarray, grad_out: np.ndarray) -> float:
            nonlocal prev_rho

            if cancelled[0]:
                grad_out[:] = 0.0
                return 0.0

            rho = jnp.asarray(rho_np)
            value = _pipe_jit(rho)
            grad = grad_fn(rho)

            f_val = float(value)
            grad_out[:] = np.asarray(grad)

            iter_count = len(history) + 1
            delta = 0.0
            if prev_rho is not None:
                delta = float(np.linalg.norm(rho_np - prev_rho))
            g_norm = float(np.linalg.norm(grad_out))
            wall = time.perf_counter() - t_start

            history.append({
                "iteration": iter_count,
                "delta_n_eff": float(value),
                "delta_rho": delta,
                "grad_norm": g_norm,
                "wall_time": wall,
            })
            print(
                f"iter {iter_count:4d}  "
                f"Δneff={value:+.6e}  "
                f"‖Δρ‖={delta:.4e}  "
                f"‖∇f‖={g_norm:.4e}  "
                f"wall={wall:.1f}s"
            )

            prev_rho = rho_np.copy()
            return f_val

        for algorithm in (nlopt.LD_MMA, nlopt.LD_CCSAQ):
            opt = nlopt.opt(algorithm, n_nodes)
            opt.set_lower_bounds(0.0)
            opt.set_upper_bounds(1.0)
            opt.set_max_objective(_obj)
            opt.set_maxeval(max_iter)
            opt.set_ftol_rel(ftol_rel)
            try:
                rho_opt = opt.optimize(initial_rho.copy())
                break
            except nlopt.RoundoffLimited:
                if algorithm == nlopt.LD_CCSAQ:
                    rho_opt = initial_rho
                    break
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if cancelled[0]:
        saved = prev_rho if prev_rho is not None else initial_rho
        _save_checkpoint(saved, history)
        raise OptimizationCancelled(
            f"Optimization interrupted by user at iteration {len(history)}."
            f" Progress saved to outputs/checkpoint.json."
        )

    return rho_opt, history


def _save_checkpoint(
    rho: np.ndarray, history: list[_HistoryEntry],
) -> None:
    """Save optimization progress to ``outputs/checkpoint.json``."""
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "rho_opt": rho.tolist(),
        "history": [
            {k: v for k, v in entry.items()}
            for entry in history
        ],
    }
    (out_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2))
