"""Main entrypoint for the PRISMO pipeline.

Invoke via ``make run`` or ``prismo run``.
Generates waveguide mesh, runs NLopt MMA optimization, and produces
paper-ready plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import typer

if TYPE_CHECKING:
    from prismo.outputs import GradientValidationResult, ModeField

app = typer.Typer(name="prismo")

_DEFAULT_OUTPUT_DIR = Path("outputs")
_DEFAULT_MESH = _DEFAULT_OUTPUT_DIR / "waveguide.msh"
_MIN_CONTAINER_OBJECTIVE = 1e-12


@app.command()
def run(
    r_min: float = typer.Option(0.05, help="Density filter radius [µm]"),
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


# Acceptance bar mirrors ``outputs.GRADIENT_VALIDATION_TOLERANCE`` (kept as a
# literal here so the CLI's ``--help`` never eagerly imports jax/matplotlib).
_DEFAULT_GRADIENT_TOLERANCE = 1e-2


@app.command(name="validate-gradient")
def validate_gradient(
    r_min: float = typer.Option(0.05, help="Density filter radius [µm]"),
    tolerance: float = typer.Option(
        _DEFAULT_GRADIENT_TOLERANCE,
        help="Acceptance bar on the worst per-direction relative error",
    ),
    n_directions: int = typer.Option(3, help="Number of sampled θ directions"),
    n_steps: int = typer.Option(
        12,
        help="Central-difference steps, log-spaced over 1e-4..1e-1 "
        "(below 1e-4 the CT readout's own 1e-8 FD floor dominates)",
    ),
    mesh_path: str = typer.Option(
        str(_DEFAULT_MESH), help="Path to waveguide .msh file"
    ),
    output_dir: str = typer.Option(
        str(_DEFAULT_OUTPUT_DIR), help="Directory for the validation figure"
    ),
    use_containers: bool = typer.Option(
        False,
        "--use-containers",
        help="Run tesseract components via Docker containers",
    ),
) -> None:
    """Validate the composed ∂(Δneff)/∂θ gradient against central FD (ticket 06).

    The hackathon's "gradients do real work" proof: checks the adjoint against
    central finite differences on sampled θ directions at the seeded junction and
    writes ``gradient_validation.pdf``. Run with ``--use-containers`` to exercise
    the real ChargeTransport + gyptis boundary (CT included, audit #7); exits
    non-zero if the worst relative error exceeds the tolerance.
    """
    from prismo.pipeline import (
        PipelineComponents,
        init_tesseract_containers,
        teardown_containers,
    )

    # Central FD across the real CT+gyptis boundary is expensive (two CT Newton
    # solves per pipeline evaluation), so restrict the sweep to the band where
    # central differences actually resolve the gradient before the CT state
    # readout's internal 1e-8 finite difference sets the floor.
    step_sizes = np.logspace(-4, -1, n_steps)

    components: PipelineComponents | None = None
    if use_containers:
        typer.echo("Starting tesseract Docker containers...")
        components = init_tesseract_containers(mesh_dir=Path(mesh_path).parent)

    try:
        result = _run_gradient_validation(
            r_min=r_min,
            mesh_path=mesh_path,
            output_dir=output_dir,
            tolerance=tolerance,
            n_directions=n_directions,
            step_sizes=step_sizes,
            use_containers=use_containers,
            components=components,
        )
    finally:
        if use_containers and components is not None:
            typer.echo("Stopping tesseract containers...")
            teardown_containers(components)

    if not result.passed:
        raise typer.Exit(code=1)


@dataclass
class PipelineInputs:
    """Everything ``pipeline()`` needs at a fixed design, shared by run/validate.

    Assembled once from the mesh: the node coordinates, the ``design_nodes``
    that carry a design variable (the silicon nodes), the signed-junction seed
    ``theta_init`` over those nodes, the density-filter matrix ``(H_dense,
    H_sum)`` over those nodes, the ``mesh_ref`` that routes ChargeTransport onto
    the real 2D grid, the ``design_transfer`` that carries the nodal field onto
    the gyptis design cells, and the ``design_vertices`` those cells occupy
    (container path only), whose bounding box outlines the silicon rib on the
    mode figure.

    ``coords`` and ``n_nodes`` stay full-mesh -- they key the solvers and the
    figures -- while ``theta_init``, ``H_dense`` and ``H_sum`` are sized by
    ``len(design_nodes)``.
    """

    geometry: Any
    coords: np.ndarray
    n_nodes: int
    real_mesh: bool
    actual_mesh: Path
    mesh_ref: Any | None
    H_dense: Any
    H_sum: Any
    theta_init: Any
    design_transfer: Any | None
    design_nodes: Any | None = None
    design_vertices: np.ndarray | None = None


def _silicon_design_nodes(actual_mesh: Path, n_nodes: int, real_mesh: bool) -> Any:
    """The design set: the shared mesh's silicon nodes, else every node.

    Reads the ``slab`` + ``rib_silicon`` triangles -- the same pair
    ``ct_common.jl`` collects -- and keeps the nodes they touch. The synthetic
    fallback grid has no physical groups at all, and a mesh whose silicon
    groups cannot be read would silently shrink the design to nothing, so both
    fall back to one variable per node rather than guessing.
    """
    from prismo.pipeline import DesignNodes
    from prismo.waveguide_mesh import read_mesh_silicon_triangulation

    if not real_mesh:
        return DesignNodes.all_nodes(n_nodes)

    triangles = read_mesh_silicon_triangulation(actual_mesh)
    if triangles.size == 0:
        typer.echo(
            "      WARNING: no silicon physical groups in the mesh; "
            "keeping one design variable per node."
        )
        return DesignNodes.all_nodes(n_nodes)

    indices = np.unique(triangles.ravel()).astype(np.intp)
    return DesignNodes(indices=indices, n_mesh_nodes=n_nodes)


def build_pipeline_inputs(
    r_min: float,
    mesh_path: str,
    use_containers: bool,
    components: Any | None,
) -> PipelineInputs:
    """Author the shared mesh and assemble the fixed pipeline inputs.

    ``prismo run`` (optimization), ``prismo validate-gradient`` (the ticket 06
    deliverable) and ``scripts/benchmark_multiphysics_optimization.py`` all
    start here so they solve on exactly the same mesh, filter, transfer, and
    seed -- the benchmark used to assemble its own and drifted out of contract
    with this one (ticket 15). Container runs author the mesh via gyptis
    ``write_mesh`` and require live components; the in-process path builds the
    rib mesh locally. Lengths are micrometres throughout, ``r_min`` included.
    """
    import jax.numpy as jnp
    from prismo_shared.schemas import MeshRef

    from prismo.density_filter import assemble_filter_matrix
    from prismo.pipeline import build_design_transfer, seed_signed_junction
    from prismo.waveguide_mesh import (
        RibWaveguideGeometry,
        build_rib_waveguide_mesh,
        read_mesh_node_coordinates,
    )

    mesh_path_obj = Path(mesh_path)

    typer.echo("Generating waveguide mesh...")
    geometry = RibWaveguideGeometry()
    if use_containers:
        if components is None or components.write_mesh is None:
            raise RuntimeError("Container pipeline requires gyptis mesh authoring")
        design_vertices = components.write_mesh(mesh_path_obj)
        actual_mesh = mesh_path_obj
    else:
        actual_mesh = build_rib_waveguide_mesh(
            mesh_path=mesh_path_obj, geometry=geometry
        )
        design_vertices = None
    typer.echo(f"      Mesh written to {actual_mesh}")

    coords = read_mesh_node_coordinates(actual_mesh)
    if coords.shape[0] == 0:
        typer.echo(
            "      WARNING: empty mesh (gmsh not available?). Using synthetic 8x8 grid."
        )
        n_side = 8
        # Micrometres, like every real mesh this app reads (ticket 15): a 20 nm
        # node pitch is 0.02 here, so the µm-scaled ``--r-min`` still means the
        # same physical radius on the fallback grid.
        xs, ys = np.meshgrid(
            np.arange(n_side) * 0.02,
            np.arange(n_side) * 0.02,
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

    # Only silicon nodes carry physics: ChargeTransport gathers doping on the
    # silicon subgrid and every gyptis design cell is a rib triangle with
    # silicon vertices. A variable anywhere else has an exactly-zero gradient,
    # or -- within r_min of silicon -- dopes the device from outside it. Both
    # are parameterizations with no physical referent, so the design set is the
    # silicon nodes and the dense filter shrinks with it.
    design_nodes = _silicon_design_nodes(actual_mesh, n_nodes, real_mesh)
    design_coords = coords[design_nodes.indices]
    typer.echo(
        f"      {len(design_nodes)} design variables on silicon nodes "
        f"({n_nodes - len(design_nodes)} non-silicon nodes carry no variable)"
    )

    typer.echo("Building density filter matrix...")
    H_sparse = assemble_filter_matrix(design_coords, r_min=r_min)
    H_dense = jnp.asarray(H_sparse.toarray())
    H_sum = jnp.sum(H_dense, axis=1)
    # Seed a signed lateral P/N junction in every run path (sign(theta) is a free
    # design variable, so the optimizer can move or dissolve it). Seeded on the
    # design nodes, so the junction splits the silicon at its own median x
    # rather than the whole domain's.
    theta_init = seed_signed_junction(design_coords)

    design_transfer = None
    if use_containers:
        # Carry the full nodal permittivity field onto the gyptis design cells
        # instead of the identity fallback, so a fixed-mean topology change moves
        # neff through the real eigenmode solve (ticket 08). The transfer is
        # assembled from live sources keyed to the shared mesh's nodes.
        if components is None:
            raise RuntimeError("Container pipeline requires live components")
        if design_vertices is None:
            raise RuntimeError(
                "Container pipeline requires gyptis design-cell vertices"
            )
        design_transfer = build_design_transfer(
            components, coords, design_cell_vertices=design_vertices
        )
        typer.echo(
            f"      Mesh-transfer operator: {design_transfer.shape[0]} design cells "
            f"<- {n_nodes} nodes"
        )
    typer.echo(f"      Filter radius: {r_min:.3g} µm")

    return PipelineInputs(
        geometry=geometry,
        coords=coords,
        n_nodes=n_nodes,
        real_mesh=real_mesh,
        actual_mesh=actual_mesh,
        mesh_ref=mesh_ref,
        H_dense=H_dense,
        H_sum=H_sum,
        theta_init=theta_init,
        design_transfer=design_transfer,
        design_nodes=design_nodes,
        design_vertices=design_vertices,
    )


def _optimized_mode_field(
    inputs: PipelineInputs,
    rho_opt: np.ndarray,
    components: Any | None,
) -> ModeField | None:
    """The tracked optical mode ``|E|`` at the optimized design (ticket 07).

    Re-runs the pipeline's own pre-eigensolve stages at ``rho_opt`` -- filter,
    doping, both ChargeTransport solves, Soref-Bennett, mesh transfer -- so the
    figure shows the mode of the *reverse-biased optimized device*, the same
    permittivity the final objective was evaluated on, rather than a
    separately-reconstructed one. Returns ``None`` when no gyptis backend is
    bound to answer the query.
    """
    import jax.numpy as jnp

    from prismo.outputs import ModeField
    from prismo.pipeline import (
        DEFAULT_BACKGROUND_EPSILON,
        default_components,
        design_epsilon_from_theta,
    )

    bundle = components if components is not None else default_components()
    if bundle.mode_field is None:
        return None

    _epsilon_bg, epsilon_pert = design_epsilon_from_theta(
        jnp.asarray(rho_opt, dtype=jnp.float64),
        H=inputs.H_dense,
        H_sum=inputs.H_sum,
        mesh_ref=inputs.mesh_ref,
        background_epsilon=DEFAULT_BACKGROUND_EPSILON,
        design_transfer=inputs.design_transfer,
        design_nodes=inputs.design_nodes,
        components=bundle,
    )
    abs_e, coords_um = bundle.mode_field(
        np.asarray(epsilon_pert), DEFAULT_BACKGROUND_EPSILON
    )

    rib_bounds = None
    if inputs.design_vertices is not None and inputs.design_vertices.size:
        verts = inputs.design_vertices.reshape(-1, 2)
        rib_bounds = (
            float(verts[:, 0].min()),
            float(verts[:, 0].max()),
            float(verts[:, 1].min()),
            float(verts[:, 1].max()),
        )
    return ModeField(abs_e=abs_e, coords_um=coords_um, rib_bounds=rib_bounds)


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
    from prismo.optimizer import OptimizationCancelled, optimize_doping
    from prismo.outputs import generate_outputs, plot_live_doping_field
    from prismo.pipeline import doping_from_theta, vpi_lpi_v_cm
    from prismo.pipeline import pipeline as pipeline_fn

    typer.echo("=== PRISMO Pipeline ===")
    typer.echo()

    typer.echo("[1/3] Preparing pipeline inputs...")
    inputs = build_pipeline_inputs(r_min, mesh_path, use_containers, components)
    geometry = inputs.geometry
    coords = inputs.coords
    mesh_ref = inputs.mesh_ref
    H_dense = inputs.H_dense
    H_sum = inputs.H_sum
    theta_init = inputs.theta_init
    design_transfer = inputs.design_transfer
    design_nodes = inputs.design_nodes

    typer.echo("[2/3] Running NLopt MMA optimization...")
    try:
        optimization_max_iter = max(max_iter, 5) if use_containers else max_iter
        optimization_ftol_rel = ftol_rel
        on_iteration = None
        if use_containers:

            def on_iteration(iteration: int, theta: np.ndarray) -> None:

                name = f"doping_field_{iteration}"
                plot_live_doping_field(
                    np.asarray(doping_from_theta(design_nodes.scatter_numpy(theta))),
                    coords,
                    iteration,
                    geometry=geometry,
                    output_dir=output_dir,
                    name=name,
                )

        rho_opt, history = optimize_doping(
            initial_rho=np.asarray(theta_init, dtype=float),
            n_nodes=len(design_nodes),
            H=H_dense,
            H_sum=H_sum,
            max_iter=optimization_max_iter,
            ftol_rel=optimization_ftol_rel,
            min_mma_evaluations=5 if use_containers else 0,
            use_jit=not no_jit,
            on_iteration=on_iteration,
            design_transfer=design_transfer,
            design_nodes=design_nodes,
            mesh_ref=mesh_ref,
            components=components,
        )
        typer.echo(f"      Optimization complete: {len(history)} iterations")
        if history:
            final_delta_neff = history[-1]["delta_n_eff"]
            typer.echo(f"      Final Delta_n_eff = {final_delta_neff:+.6e}")
            # VπLπ headline (V·cm): the field-standard modulation efficiency,
            # reported from Δneff at the fixed -5 V bias (smaller |VπLπ| better).
            typer.echo(f"      VpiLpi = {vpi_lpi_v_cm(final_delta_neff):+.4e} V·cm")
    except OptimizationCancelled:
        typer.echo("      Optimization cancelled by user.")
        return

    if use_containers:
        # This audits that the containers produced a *live* signal, not that the
        # optimizer liked it. The magnitude is what carries that: a dead or
        # stubbed pipeline reads |Delta_neff| ~ 0 with no gradient. The sign is
        # not a validity condition -- with a signed design field the optimizer
        # legitimately crosses the junction-polarity boundary while searching,
        # and a wrong-polarity *result* is surfaced by the reported VpiLpi.
        invalid = [
            entry
            for entry in history
            if (
                abs(entry["delta_n_eff"]) <= _MIN_CONTAINER_OBJECTIVE
                or entry["grad_norm"] <= 0.0
            )
        ]
        if invalid:
            raise RuntimeError(
                "Container pipeline produced invalid optimization signal at "
                f"{len(invalid)} iteration(s)"
            )

    typer.echo("[3/3] Generating outputs...")
    rho_initial = np.asarray(theta_init, dtype=float)
    # The figures are drawn on the full mesh, so the design field goes back into
    # full node order first; non-design nodes read theta = 0 (net-intrinsic),
    # which is what oxide means anyway.
    plot_initial = design_nodes.scatter_numpy(rho_initial)
    plot_opt = design_nodes.scatter_numpy(rho_opt)
    # The mode figure is a post-hoc query, so a backend that cannot answer it --
    # an image predating the ``mode_field`` operation, say -- must cost one
    # figure, not every figure of a finished multi-minute optimization.
    try:
        mode_field = _optimized_mode_field(inputs, rho_opt, components)
    except Exception as exc:
        typer.echo(f"      WARNING: mode figure skipped ({exc})")
        mode_field = None
    plot_paths = generate_outputs(
        rho_initial=plot_initial,
        rho_opt=plot_opt,
        history=history,
        # Node coordinates are micrometres on both paths (ticket 15), which is
        # what the figures plot in -- no conversion at this seam.
        mesh_coords=coords,
        geometry=geometry,
        # Bind the *same* pipeline the optimizer drove -- filter, mesh_ref and
        # transfer included. Omitting them made the figure's finite differences
        # probe an unfiltered pipeline with ChargeTransport on its 1D fallback
        # device, i.e. a different function than the one that was optimized
        # (ticket 15).
        pipeline_fn=partial(
            pipeline_fn,
            H=H_dense,
            H_sum=H_sum,
            mesh_ref=mesh_ref,
            design_transfer=design_transfer,
            design_nodes=design_nodes,
            components=components,
        ),
        ftol_rel=optimization_ftol_rel,
        gradient_validation_directions=1 if use_containers else 3,
        gradient_validation_steps=(np.logspace(-4, -2, 3) if use_containers else None),
        # Explicit on both paths: the figure's finite differences probe the
        # bound pipeline, which takes a design-node vector, while ``rho_opt``
        # above has already been scattered to full node order for plotting.
        gradient_validation_rho=rho_initial if use_containers else rho_opt,
        mode_field=mode_field,
        output_dir=output_dir,
    )
    for p in plot_paths:
        typer.echo(f"      {p}")

    typer.echo()
    typer.echo("=== Done ===")


def _run_gradient_validation(
    r_min: float,
    mesh_path: str,
    output_dir: str,
    tolerance: float,
    n_directions: int,
    use_containers: bool,
    components: Any | None = None,
    step_sizes: np.ndarray | None = None,
) -> GradientValidationResult:
    """Check the composed gradient at the seeded design and write the figure.

    Unlike the optimization run, this binds the *full* pipeline -- filter,
    ``mesh_ref`` (so ChargeTransport solves on the shared 2D grid, not its 1D
    fallback), and design transfer -- into the function the finite-difference
    sweep probes, so the CT adjoint is the thing being proven (audit #7).
    """
    from prismo.outputs import validate_gradient as validate_gradient_fn
    from prismo.pipeline import pipeline as pipeline_fn

    typer.echo("=== PRISMO Gradient Validation ===")
    typer.echo()

    typer.echo("[1/2] Preparing pipeline inputs...")
    inputs = build_pipeline_inputs(r_min, mesh_path, use_containers, components)

    typer.echo("[2/2] Checking adjoint against central finite differences...")
    bound_pipeline = partial(
        pipeline_fn,
        H=inputs.H_dense,
        H_sum=inputs.H_sum,
        mesh_ref=inputs.mesh_ref,
        design_transfer=inputs.design_transfer,
        design_nodes=inputs.design_nodes,
        components=components,
    )
    result = validate_gradient_fn(
        bound_pipeline,
        inputs.theta_init,
        n_directions=n_directions,
        step_sizes=step_sizes,
        tolerance=tolerance,
        output_dir=output_dir,
    )

    verdict = "PASS" if result.passed else "FAIL"
    typer.echo(
        f"      {verdict}: worst relative error {result.worst_rel_error:.3e} "
        f"(tolerance {result.tolerance:.3e})"
    )
    for i, err in enumerate(result.best_rel_errors):
        typer.echo(f"        direction {i + 1}: best relative error {err:.3e}")
    typer.echo(f"      Figure: {result.figure_path}")

    typer.echo()
    typer.echo("=== Done ===")
    return result


def entrypoint() -> None:
    """CLI entrypoint for the application."""
    app()


if __name__ == "__main__":
    entrypoint()
