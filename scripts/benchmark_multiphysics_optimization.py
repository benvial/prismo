"""Record cold and warm PRISMO value-and-gradient callback timings.

Run with ``make benchmark``. Pass command options through
``BENCHMARK_ARGS``, for example ``make benchmark BENCHMARK_ARGS='--iterations 8'``.
The default container mode measures the deployed Tesseract boundary and uses
the same mesh, fixed P/N polarity, and mesh-transfer setup as ``prismo run
--use-containers``. Use ``--mode in-process`` only in an environment with
both solver dependencies.

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

# Node count for the standalone ChargeTransport benchmark's synthetic 1D
# doping profile, used when ``--n-nodes`` is not given.
_DEFAULT_CT_BENCHMARK_NODES = 62


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=None,
        help="Node count. For --component chargetransport this sizes the "
        "synthetic doping profile (default 62); for the full pipeline it is "
        "an optional assertion on the shared mesh's node count, which the "
        "mesh author decides.",
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=Path("outputs/waveguide.msh"),
        help="Shared waveguide mesh; generated when absent.",
    )
    # Micrometres, like the mesh coordinates it is compared against (ticket 15).
    parser.add_argument("--r-min", type=float, default=0.05)
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
    """Cold/warm apply()/vector_jacobian_product() on the design-epsilon field.

    The design field is sized from the component's own design region: the
    unified mesh (ticket 05) fixed how many DG0 design cells the rib interior
    has, and ``apply`` rejects any other length. A hardcoded four-domain
    ``epsilon`` vector -- the input the layered model took before the unified
    mesh -- is what made this benchmark dead code (ticket 15).
    """
    import numpy as np

    api = _load_component_api("gyptis")
    n_design = int(np.asarray(api.design_cell_centroids()).shape[0])
    design_epsilon = np.full(n_design, api.DEFAULT_CORE_EPSILON)
    cotangent = 1.0

    measurements: list[dict[str, object]] = []
    for iteration in range(iterations):
        inputs = api.InputSchema(design_epsilon=design_epsilon)

        started_at = time.perf_counter()
        api.apply(inputs)
        forward_seconds = time.perf_counter() - started_at

        started_at = time.perf_counter()
        api.vector_jacobian_product(
            inputs, {"design_epsilon"}, {"neff_sq"}, {"neff_sq": cotangent}
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
        # This benchmark sizes its own synthetic 1D doping profile, so it needs
        # a node count even when no shared mesh has been authored.
        n_nodes = _DEFAULT_CT_BENCHMARK_NODES if args.n_nodes is None else args.n_nodes
        measurements = _run_chargetransport_benchmark(n_nodes, args.iterations)
    else:
        measurements = _run_gyptis_benchmark(args.iterations)

    result = {
        "metadata": {
            "component": args.component,
            "n_nodes": n_nodes if args.component == "chargetransport" else None,
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


def _prepare_full_pipeline(args: argparse.Namespace, components: Any) -> Any:
    """Build the full-callback inputs exactly as the container CLI does.

    Delegates to ``prismo.main.build_pipeline_inputs`` -- the same mesh author,
    density filter, junction seed, ``mesh_ref`` and mesh-transfer operator that
    ``prismo run --use-containers`` drives. Assembling a second copy here is
    what let this script drift out of contract with the code it benchmarks: it
    authored the *local* rib mesh in container mode and then passed that mesh's
    path where ``build_design_transfer`` expects design-cell vertices, so the
    container benchmark raised before timing anything (ticket 15).
    """
    from prismo.main import build_pipeline_inputs

    inputs = build_pipeline_inputs(
        r_min=args.r_min,
        mesh_path=str(args.mesh_path),
        use_containers=args.mode == "containers",
        components=components,
    )
    if inputs.n_nodes == 0:
        raise RuntimeError("waveguide mesh has no nodes; install gmsh and retry")
    if args.n_nodes is not None and args.n_nodes != inputs.n_nodes:
        raise ValueError(
            f"--n-nodes={args.n_nodes} does not match the shared mesh "
            f"({inputs.n_nodes} nodes)"
        )
    return inputs


def main() -> None:
    """Run one cold callback and requested warm callbacks, then save JSON."""
    args = _parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least one")
    if args.n_nodes is not None and args.n_nodes < 1:
        raise ValueError("--n-nodes must be at least one")

    if args.component != "full":
        _run_component_benchmark(args)
        return

    import jax
    import jax.numpy as jnp
    from prismo.pipeline import default_components, init_tesseract_containers, pipeline

    components = (
        # The shared mesh's directory is bind-mounted into the ChargeTransport
        # container, which is how the ``mesh_ref`` below resolves to a file it
        # can actually read; without the mount it silently solves on its 1D
        # fallback instead.
        init_tesseract_containers(mesh_dir=args.mesh_path.parent)
        if args.mode == "containers"
        else default_components()
    )

    try:
        inputs = _prepare_full_pipeline(args, components)
        n_nodes = inputs.n_nodes
        # The callback's argument is the design vector, which spans the silicon
        # nodes rather than every mesh node (``pipeline.DesignNodes``).
        n_design = len(inputs.design_nodes)

        def objective(theta: Any) -> Any:
            # ``mesh_ref`` included: without it ChargeTransport falls back to
            # its 1D device, and the timings would describe a solve the real
            # callback never runs (ticket 15).
            return pipeline(
                theta,
                H=inputs.H_dense,
                H_sum=inputs.H_sum,
                mesh_ref=inputs.mesh_ref,
                design_transfer=inputs.design_transfer,
                design_nodes=inputs.design_nodes,
                components=components,
            )

        callback = jax.value_and_grad(objective)
        if not args.no_jit:
            callback = jax.jit(callback)
        initial_rho = jnp.asarray(inputs.theta_init, dtype=jnp.float64)
        direction = jnp.linspace(-1.0, 1.0, n_design, dtype=jnp.float64)
        direction = direction / jnp.linalg.norm(direction)
        measurements: list[dict[str, object]] = []
        for iteration in range(args.iterations):
            rho = initial_rho + iteration * 1e-4 * direction
            started_at = time.perf_counter()
            value, gradient = callback(rho)
            value_float = float(value)
            gradient_norm = float(jnp.linalg.norm(gradient))
            measurements.append(
                {
                    "kind": "cold" if iteration == 0 else "warm",
                    "callback_seconds": time.perf_counter() - started_at,
                    "delta_n_eff": value_float,
                    "gradient_norm": gradient_norm,
                    "rho_change_norm": float(jnp.linalg.norm(rho - initial_rho)),
                }
            )
    finally:
        if args.mode == "containers":
            components.close()

    result = {
        "metadata": {
            "mode": args.mode,
            "n_nodes": n_nodes,
            "n_design": n_design,
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
