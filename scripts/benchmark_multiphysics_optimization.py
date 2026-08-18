"""Record cold and warm PRISMO value-and-gradient callback timings.

Run with ``make benchmark``. Pass command options through
``BENCHMARK_ARGS``, for example ``make benchmark BENCHMARK_ARGS='--iterations 8'``.
The default container mode measures the deployed Tesseract boundary. Use
``--mode in-process`` only in an environment with both solver dependencies.

``--component chargetransport`` or ``--component gyptis`` benchmark one
Tesseract component in isolation (its own public ``apply``/
``vector_jacobian_product``, imported directly, independent of JAX
composition or the optimizer callback) rather than the full composed
pipeline. Both require an environment with that component's solver
dependencies importable in-process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--n-nodes", type=int, default=62)
    parser.add_argument(
        "--mode",
        choices=("containers", "in-process"),
        default="containers",
    )
    parser.add_argument(
        "--component",
        choices=("full", "chargetransport", "gyptis"),
        default="full",
        help="Benchmark the full composed pipeline, or one Tesseract "
        "component in isolation.",
    )
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/multiphysics-benchmark.json"),
    )
    return parser.parse_args()


def _load_component_api(name: str) -> Any:
    """Import one Tesseract component's tesseract_api.py directly."""
    api_path = (
        Path(__file__).resolve().parents[1]
        / "components"
        / "tesseracts"
        / name
        / "tesseract_api.py"
    )
    spec = importlib.util.spec_from_file_location(f"_{name}_benchmark_api", api_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Tesseract component API at {api_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_chargetransport_benchmark(
    n_nodes: int, iterations: int
) -> list[dict[str, object]]:
    """Cold/warm apply()/vector_jacobian_product() at both fixed bias points."""
    import numpy as np

    api = _load_component_api("chargetransport")
    magnitude = np.geomspace(1e14, 1e20, n_nodes)
    doping = np.where(np.arange(n_nodes) < n_nodes // 2, -magnitude[::-1], magnitude)
    cotangent = {
        "electrons": np.ones(n_nodes),
        "holes": np.ones(n_nodes),
    }

    measurements: list[dict[str, object]] = []
    for iteration in range(iterations):
        for bias_voltage in (0.0, -5.0):
            inputs = api.InputSchema(doping=doping, bias_voltage=bias_voltage)

            started_at = time.perf_counter()
            api.apply(inputs)
            forward_seconds = time.perf_counter() - started_at

            started_at = time.perf_counter()
            api.vector_jacobian_product(
                inputs, {"doping"}, {"electrons", "holes"}, cotangent
            )
            vjp_seconds = time.perf_counter() - started_at

            measurements.append(
                {
                    "kind": "cold" if iteration == 0 else "warm",
                    "bias_voltage": bias_voltage,
                    "forward_seconds": forward_seconds,
                    "vjp_seconds": vjp_seconds,
                }
            )
    return measurements


def _run_gyptis_benchmark(iterations: int) -> list[dict[str, object]]:
    """Cold/warm apply()/vector_jacobian_product() for the layered case."""
    import numpy as np

    api = _load_component_api("gyptis")
    n_domains = 4
    epsilon = np.full(n_domains, 12.0)
    cotangent = 1.0

    measurements: list[dict[str, object]] = []
    for iteration in range(iterations):
        inputs = api.InputSchema(epsilon=epsilon)

        started_at = time.perf_counter()
        api.apply(inputs)
        forward_seconds = time.perf_counter() - started_at

        started_at = time.perf_counter()
        api.vector_jacobian_product(
            inputs, {"epsilon"}, {"neff_sq"}, {"neff_sq": cotangent}
        )
        vjp_seconds = time.perf_counter() - started_at

        measurements.append(
            {
                "kind": "cold" if iteration == 0 else "warm",
                "forward_seconds": forward_seconds,
                "vjp_seconds": vjp_seconds,
            }
        )
    return measurements


def _run_component_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Benchmark one Tesseract component in isolation and save JSON."""
    if args.component == "chargetransport":
        measurements = _run_chargetransport_benchmark(args.n_nodes, args.iterations)
    else:
        measurements = _run_gyptis_benchmark(args.iterations)

    result = {
        "metadata": {
            "component": args.component,
            "n_nodes": args.n_nodes,
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)
    return result


def main() -> None:
    """Run one cold callback and requested warm callbacks, then save JSON."""
    args = _parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least one")
    if args.n_nodes < 1:
        raise ValueError("--n-nodes must be at least one")

    if args.component != "full":
        _run_component_benchmark(args)
        return

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
