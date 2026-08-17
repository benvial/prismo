"""Main entrypoint for the PRISMO pipeline.

Invoke via ``make run`` or ``prismo run``.
Generates waveguide mesh, runs NLopt MMA optimization, and produces
paper-ready plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

app = typer.Typer(name="prismo")

_DEFAULT_OUTPUT_DIR = Path("outputs")
_DEFAULT_MESH = _DEFAULT_OUTPUT_DIR / "waveguide.msh"


@app.command()
def run(
    r_min: float = typer.Option(50e-9, help="Density filter radius [m]"),
    max_iter: int = typer.Option(200, help="Max MMA iterations"),
    ftol_rel: float = typer.Option(1e-5, help="Relative tolerance on objective"),
    mesh_path: str = typer.Option(
        str(_DEFAULT_MESH),
        help="Path to waveguide .msh file",
    ),
    output_dir: str = typer.Option(
        str(_DEFAULT_OUTPUT_DIR),
        help="Directory for output plots",
    ),
    no_jit: bool = typer.Option(False, help="Disable JIT compilation"),
    use_containers: bool = typer.Option(
        False,
        "--use-containers",
        help="Run tesseract components via Docker containers",
    ),
) -> None:
    """Run the PRISMO differentiable pipeline end-to-end.

    Steps:
    1. Generate waveguide mesh (gmsh required, falls back to empty mesh)
    2. Build density filter matrix from mesh node coordinates
    3. Run NLopt MMA optimization (maximize delta_n_eff)
    4. Generate paper-ready plots
    """
    from prismo.pipeline import (
        init_tesseract_containers,
        teardown_containers,
    )

    if use_containers:
        typer.echo("Starting tesseract Docker containers...")
        init_tesseract_containers()

    try:
        _run_pipeline(
            r_min=r_min,
            max_iter=max_iter,
            ftol_rel=ftol_rel,
            mesh_path=mesh_path,
            output_dir=output_dir,
            no_jit=no_jit,
            use_containers=use_containers,
        )
    finally:
        if use_containers:
            typer.echo("Stopping tesseract containers...")
            teardown_containers()


def _run_pipeline(
    r_min: float,
    max_iter: int,
    ftol_rel: float,
    mesh_path: str,
    output_dir: str,
    no_jit: bool,
    use_containers: bool,
) -> None:
    from prismo.density_filter import assemble_filter_matrix
    from prismo.optimizer import OptimizationCancelled, optimize_doping
    from prismo.outputs import generate_outputs
    from prismo.pipeline import pipeline as pipeline_fn
    from prismo.waveguide_mesh import (
        RibWaveguideGeometry,
        build_rib_waveguide_mesh,
        read_mesh_node_coordinates,
    )

    mesh_path_obj = Path(mesh_path)

    typer.echo("=== PRISMO Pipeline ===")
    typer.echo()

    typer.echo("[1/4] Generating waveguide mesh...")
    geometry = RibWaveguideGeometry()
    actual_mesh = build_rib_waveguide_mesh(mesh_path=mesh_path_obj, geometry=geometry)
    typer.echo(f"      Mesh written to {actual_mesh}")

    typer.echo("[2/4] Building density filter matrix...")
    coords = read_mesh_node_coordinates(actual_mesh)
    if coords.shape[0] == 0:
        typer.echo(
            "      WARNING: empty mesh (gmsh not available?). Using synthetic 8x8 grid."
        )
        n_side = 8
        xs, ys = np.meshgrid(
            np.arange(n_side) * 20e-9,
            np.arange(n_side) * 20e-9,
            indexing="xy",
        )
        coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
    n_nodes = coords.shape[0]
    typer.echo(f"      {n_nodes} nodes")

    import jax.numpy as jnp

    H_sparse = assemble_filter_matrix(coords, r_min=r_min)
    H_dense = jnp.asarray(H_sparse.toarray())
    H_sum = jnp.sum(H_dense, axis=1)
    typer.echo(f"      Filter radius: {r_min * 1e9:.0f} nm")

    typer.echo("[3/4] Running NLopt MMA optimization...")
    try:
        rho_opt, history = optimize_doping(
            n_nodes=n_nodes,
            H=H_dense,
            H_sum=H_sum,
            max_iter=max_iter,
            ftol_rel=ftol_rel,
            use_jit=not no_jit,
        )
        typer.echo(f"      Optimization complete: {len(history)} iterations")
        if history:
            typer.echo(f"      Final Delta_n_eff = {history[-1]['delta_n_eff']:+.6e}")
    except OptimizationCancelled:
        typer.echo("      Optimization cancelled by user.")
        return

    if use_containers:
        if len(history) < 5:
            raise RuntimeError(
                f"Container pipeline completed only {len(history)} iterations; "
                "expected at least 5"
            )
        invalid = [
            entry
            for entry in history
            if entry["delta_n_eff"] <= 0.0 or entry["grad_norm"] <= 0.0
        ]
        if invalid:
            raise RuntimeError(
                "Container pipeline produced invalid optimization signal at "
                f"{len(invalid)} iteration(s)"
            )

    typer.echo("[4/4] Generating outputs...")
    rho_initial = np.full(n_nodes, 0.25, dtype=float)
    plot_paths = generate_outputs(
        rho_initial=rho_initial,
        rho_opt=rho_opt,
        history=history,
        mesh_coords=coords,
        geometry=geometry,
        pipeline_fn=pipeline_fn,
        ftol_rel=ftol_rel,
        output_dir=output_dir,
    )
    for p in plot_paths:
        typer.echo(f"      {p}")

    typer.echo()
    typer.echo("=== Done ===")


def entrypoint() -> None:
    """CLI entrypoint for the application."""
    app()


if __name__ == "__main__":
    entrypoint()
