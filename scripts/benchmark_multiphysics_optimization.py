"""Record cold and warm PRISMO value-and-gradient callback timings.

Run with ``make benchmark``. Pass command options through
``BENCHMARK_ARGS``, for example ``make benchmark BENCHMARK_ARGS='--iterations 8'``.
The default container mode measures the deployed Tesseract boundary. Use
``--mode in-process`` only in an environment with both solver dependencies.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--n-nodes", type=int, default=62)
    parser.add_argument(
        "--mode",
        choices=("containers", "in-process"),
        default="containers",
    )
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/multiphysics-benchmark.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Run one cold callback and requested warm callbacks, then save JSON."""
    args = _parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least one")
    if args.n_nodes < 1:
        raise ValueError("--n-nodes must be at least one")

    import jax
    import jax.numpy as jnp
    from prismo.pipeline import (
        begin_pipeline_callback_timing,
        clear_pipeline_runtime_state,
        finish_pipeline_callback_timing,
        init_tesseract_containers,
        pipeline,
        teardown_containers,
    )

    clear_pipeline_runtime_state()
    if args.mode == "containers":
        init_tesseract_containers()

    try:
        callback = jax.value_and_grad(pipeline)
        if not args.no_jit:
            callback = jax.jit(callback)
        initial_rho = jnp.full((args.n_nodes,), 0.25, dtype=jnp.float64)
        direction = jnp.linspace(-1.0, 1.0, args.n_nodes, dtype=jnp.float64)
        direction = direction / jnp.linalg.norm(direction)
        measurements: list[dict[str, object]] = []
        for iteration in range(args.iterations):
            rho = initial_rho + iteration * 1e-4 * direction
            started_at = time.perf_counter()
            begin_pipeline_callback_timing()
            try:
                value, gradient = callback(rho)
                value_float = float(value)
                gradient_norm = float(jnp.linalg.norm(gradient))
            finally:
                phase_timing = finish_pipeline_callback_timing()
            measurements.append(
                {
                    "kind": "cold" if iteration == 0 else "warm",
                    "callback_seconds": time.perf_counter() - started_at,
                    "delta_n_eff": value_float,
                    "gradient_norm": gradient_norm,
                    "rho_change_norm": float(jnp.linalg.norm(rho - initial_rho)),
                    "phase_timing": phase_timing,
                }
            )
    finally:
        if args.mode == "containers":
            teardown_containers()

    result = {
        "metadata": {
            "mode": args.mode,
            "n_nodes": args.n_nodes,
            "jit": not args.no_jit,
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax": jax.__version__,
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
