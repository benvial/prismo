"""Main entrypoint for the PRISMO pipeline.

Invoke via ``make run`` or ``prismo run``.
Generates waveguide mesh, runs NLopt MMA optimization, and produces
paper-ready plots.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import typer

app = typer.Typer(name="prismo")

_DEFAULT_OUTPUT_DIR = Path("outputs")
_DEFAULT_MESH = _DEFAULT_OUTPUT_DIR / "waveguide.msh"
_MIN_CONTAINER_OBJECTIVE = 1e-12


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
        PipelineComponents,
        init_tesseract_containers,
        teardown_containers,
    )

    components: PipelineComponents | None = None
    if use_containers:
        typer.echo("Starting tesseract Docker containers...")
        components = init_tesseract_containers(mesh_dir=Path(mesh_path).parent)

    try:
        _run_pipeline(
            r_min=r_min,
            max_iter=max_iter,
            ftol_rel=ftol_rel,
            mesh_path=mesh_path,
            output_dir=output_dir,
            no_jit=no_jit,
            use_containers=use_containers,
            components=components,
        )
    finally:
        if use_containers and components is not None:
            typer.echo("Stopping tesseract containers...")
            teardown_containers(components)


def _run_pipeline(
    r_min: float,
    max_iter: int,
    ftol_rel: float,
    mesh_path: str,
    output_dir: str,
    no_jit: bool,
    use_containers: bool,
    components: Any | None = None,
) -> None:
    from prismo_shared.schemas import MeshRef

    from prismo.density_filter import assemble_filter_matrix
    from prismo.optimizer import OptimizationCancelled, optimize_doping
    from prismo.outputs import generate_outputs, plot_live_doping_field
    from prismo.pipeline import (
        build_design_transfer,
        doping_from_theta,
        seed_signed_junction,
        vpi_lpi_v_cm,
    )
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
        real_mesh = False
    else:
        real_mesh = True
    n_nodes = coords.shape[0]
    typer.echo(f"      {n_nodes} nodes")

    # Reference the real 2D grid so ChargeTransport solves on it instead of its
    # 1D fallback device, where the gmsh-order lateral junction collapses into a
    # many-junction line the reverse-bias solve cannot converge on.
    mesh_ref = (
        MeshRef(path=str(actual_mesh), n_nodes=n_nodes, node_ordering="gmsh")
        if real_mesh
        else None
    )

    import jax.numpy as jnp

    H_sparse = assemble_filter_matrix(coords, r_min=r_min)
    H_dense = jnp.asarray(H_sparse.toarray())
    H_sum = jnp.sum(H_dense, axis=1)
    # Seed a signed lateral P/N junction in every run path (sign(theta) is a free
    # design variable, so the optimizer can move or dissolve it).
    theta_init = seed_signed_junction(coords)

    design_transfer = None
    if use_containers:
        # Carry the full nodal permittivity field onto the gyptis design cells
        # instead of the identity fallback, so a fixed-mean topology change moves
        # neff through the real eigenmode solve (ticket 08). The transfer is
        # assembled from live sources keyed to the shared mesh's nodes.
        if components is None:
            raise RuntimeError("Container pipeline requires live components")
        design_transfer = build_design_transfer(components, coords)
        typer.echo(
            f"      Mesh-transfer operator: {design_transfer.shape[0]} design cells "
            f"<- {n_nodes} nodes"
        )
    typer.echo(f"      Filter radius: {r_min * 1e9:.0f} nm")

    typer.echo("[3/4] Running NLopt MMA optimization...")
    try:
        optimization_max_iter = max(max_iter, 5) if use_containers else max_iter
        optimization_ftol_rel = ftol_rel
        on_iteration = None
        if use_containers:

            def on_iteration(iteration: int, theta: np.ndarray) -> None:

                name = f"doping_field_{iteration}"
                plot_live_doping_field(
                    np.asarray(doping_from_theta(theta)),
                    coords,
                    iteration,
                    geometry=geometry,
                    output_dir=output_dir,
                    name=name,
                )

        rho_opt, history = optimize_doping(
            initial_rho=np.asarray(theta_init, dtype=float),
            n_nodes=n_nodes,
            H=H_dense,
            H_sum=H_sum,
            max_iter=optimization_max_iter,
            ftol_rel=optimization_ftol_rel,
            min_mma_evaluations=5 if use_containers else 0,
            use_jit=not no_jit,
            on_iteration=on_iteration,
            design_transfer=design_transfer,
            mesh_ref=mesh_ref,
            components=components,
        )
        typer.echo(f"      Optimization complete: {len(history)} iterations")
        if history:
            final_delta_neff = history[-1]["delta_n_eff"]
            typer.echo(f"      Final Delta_n_eff = {final_delta_neff:+.6e}")
            # VπLπ headline (V·cm): the field-standard modulation efficiency,
            # reported from Δneff at the fixed -5 V bias (smaller |VπLπ| better).
            typer.echo(
                f"      VpiLpi = {vpi_lpi_v_cm(final_delta_neff):+.4e} V·cm"
            )
    except OptimizationCancelled:
        typer.echo("      Optimization cancelled by user.")
        return

    if use_containers:
        invalid = [
            entry
            for entry in history
            if (
                entry["delta_n_eff"] <= _MIN_CONTAINER_OBJECTIVE
                or entry["grad_norm"] <= 0.0
            )
        ]
        if invalid:
            raise RuntimeError(
                "Container pipeline produced invalid optimization signal at "
                f"{len(invalid)} iteration(s)"
            )

    typer.echo("[4/4] Generating outputs...")
    rho_initial = np.asarray(theta_init, dtype=float)
    plot_paths = generate_outputs(
        rho_initial=rho_initial,
        rho_opt=rho_opt,
        history=history,
        mesh_coords=coords,
        geometry=geometry,
        pipeline_fn=partial(
            pipeline_fn,
            design_transfer=design_transfer,
            components=components,
        ),
        ftol_rel=optimization_ftol_rel,
        gradient_validation_directions=1 if use_containers else 3,
        gradient_validation_steps=(np.logspace(-4, -2, 3) if use_containers else None),
        gradient_validation_rho=rho_initial if use_containers else None,
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
