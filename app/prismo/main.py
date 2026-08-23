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
    from prismo.outputs import (
        ColdReevaluation,
        GradientValidationResult,
        ModeField,
        ObjectiveLineScan,
    )

app = typer.Typer(name="prismo")

_DEFAULT_OUTPUT_DIR = Path("outputs")
_DEFAULT_MESH = _DEFAULT_OUTPUT_DIR / "waveguide.msh"
_MIN_CONTAINER_OBJECTIVE = 1e-12
# Per-iteration move limit on theta (mirrors ``optimizer.DEFAULT_MOVE_LIMIT``;
# kept as a literal so ``--help`` never eagerly imports jax/nlopt).
_DEFAULT_MOVE_LIMIT = 0.05
# Density-filter radius [µm]: 3-4 elements of the 0.04 µm container mesh, so
# the result is smooth enough to be an implant mask. The previous
# 0.05 µm reached one neighbour and left checkerboard-like slab values.
_DEFAULT_R_MIN = 0.10
# Junction seeds (``--seed``); mirrors ``pipeline.SEED_KINDS`` as literals so
# ``--help`` never eagerly imports jax.
_SEED_CHOICES = ("lateral", "vertical", "u")
_CHECKPOINT_NAME = "checkpoint.json"
# Live per-iteration doping frames: ``<prefix><iteration>.png`` in the output dir.
_LIVE_FRAME_PREFIX = "doping_field_"


def _seed_option() -> Any:
    return typer.Option(
        "lateral",
        "--seed",
        help="Initial junction: lateral (n left / p right), vertical (p over n "
        "in the rib), or u (n wrapped under and beside a p core)",
    )


def _contact_offset_option() -> Any:
    return typer.Option(
        None,
        "--contact-offset",
        help="Gap from the rib edge to the near contact edge [µm] "
        "(default: the mesh author's own, 0.2 µm; foundries use 0.5-1 µm)",
    )


def _domain_width_option() -> Any:
    return typer.Option(
        None,
        "--domain-width",
        help="Physical box width [µm] the slab spans, PML excluded "
        "(default: the mesh author's own, 2.0 µm in containers, 3.0 µm local)",
    )


def _check_geometry_knobs(
    seed: str, contact_offset: float | None, domain_width: float | None
) -> None:
    """Reject bad ``--seed`` / geometry values before any container is started.

    The same values reach the gyptis mesh author at container start, where a
    bad one surfaces as an import failure inside the container rather than as
    a CLI error; validate here first.
    """
    if seed not in _SEED_CHOICES:
        raise typer.BadParameter(f"--seed must be one of {', '.join(_SEED_CHOICES)}")
    if contact_offset is not None and contact_offset <= 0.0:
        raise typer.BadParameter("--contact-offset must be positive [µm]")
    if domain_width is not None and domain_width <= 0.0:
        raise typer.BadParameter("--domain-width must be positive [µm]")


@app.command()
def run(
    r_min: float = typer.Option(_DEFAULT_R_MIN, help="Density filter radius [µm]"),
    max_iter: int = typer.Option(200, help="Max MMA iterations"),
    ftol_rel: float = typer.Option(1e-5, help="Relative tolerance on objective"),
    move_limit: float = typer.Option(
        _DEFAULT_MOVE_LIMIT,
        help="Per-iteration move limit on theta: no design variable moves more "
        "than this per iteration; halved and retried after a failed solve",
    ),
    mesh_path: str = typer.Option(
        str(_DEFAULT_MESH),
        help="Path to waveguide .msh file",
    ),
    mesh_size: float = typer.Option(
        None,
        help="Silicon element size [µm]; smaller refines the shared mesh "
        "(default: the mesh author's own, 0.04 µm in containers)",
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
    mode_index: int = typer.Option(
        0,
        "--mode-index",
        min=0,
        help="Guided mode whose Δneff is optimized: 0 = fundamental (largest "
        "neff), k = k-th guided mode in descending neff; ranked on the first "
        "eigensolve and tracked by nearest eigenvalue afterwards",
    ),
    loss_weight: float = typer.Option(
        0.0,
        "--loss-weight",
        min=0.0,
        help="Weight w of the modal free-carrier loss in the objective "
        "Δneff - w·alpha [neff per dB/cm]; 0 optimizes Δneff alone and only "
        "reports the loss. ~4e-6 trades 1e-4 of Δneff against "
        "25 dB/cm",
    ),
    seed: str = _seed_option(),
    contact_offset: float = _contact_offset_option(),
    domain_width: float = _domain_width_option(),
) -> None:
    """Run the PRISMO differentiable pipeline end-to-end.

    Steps:
    1. Generate waveguide mesh (gmsh required, falls back to empty mesh)
    2. Build density filter matrix from mesh node coordinates
    3. Run NLopt MMA optimization (maximize delta_n_eff - loss_weight·alpha)
    4. Generate paper-ready plots
    """
    _check_geometry_knobs(seed, contact_offset, domain_width)
    from prismo.pipeline import (
        PipelineComponents,
        build_default_components,
        init_tesseract_containers,
        teardown_containers,
    )

    components: PipelineComponents | None = None
    if use_containers:
        typer.echo("Starting tesseract Docker containers...")
        components = init_tesseract_containers(
            mesh_dir=Path(mesh_path).parent,
            mesh_size=mesh_size,
            mode_index=mode_index,
            contact_offset=contact_offset,
            domain_width=domain_width,
        )
    elif mode_index:
        # The shared in-process bundle targets the fundamental; a higher-order
        # target needs its own bundle (the mode is bound at build time).
        components = build_default_components(mode_index=mode_index)

    try:
        _run_pipeline(
            r_min=r_min,
            max_iter=max_iter,
            ftol_rel=ftol_rel,
            mesh_path=mesh_path,
            mesh_size=mesh_size,
            output_dir=output_dir,
            no_jit=no_jit,
            use_containers=use_containers,
            components=components,
            move_limit=move_limit,
            mode_index=mode_index,
            loss_weight=loss_weight,
            seed=seed,
            contact_offset=contact_offset,
            domain_width=domain_width,
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
    r_min: float = typer.Option(_DEFAULT_R_MIN, help="Density filter radius [µm]"),
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
    cold: bool = typer.Option(
        False,
        "--cold",
        help="Reset the ChargeTransport worker before every finite-difference "
        "evaluation so the FD reference carries no warm-start path dependence "
        "(default: warm)",
    ),
    loss_weight: float = typer.Option(
        0.0,
        "--loss-weight",
        min=0.0,
        help="Validate the loss-penalized objective Δneff - w·alpha instead of "
        "Δneff alone",
    ),
    seed: str = _seed_option(),
    contact_offset: float = _contact_offset_option(),
    domain_width: float = _domain_width_option(),
) -> None:
    """Validate the composed ∂(Δneff)/∂θ gradient against central FD.

    The hackathon's "gradients do real work" proof: checks the adjoint against
    central finite differences on sampled θ directions at the seeded junction and
    writes ``gradient_validation.pdf``. Run with ``--use-containers`` to exercise
    the real ChargeTransport + gyptis boundary (CT included); exits
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
    _check_geometry_knobs(seed, contact_offset, domain_width)

    components: PipelineComponents | None = None
    if use_containers:
        typer.echo("Starting tesseract Docker containers...")
        components = init_tesseract_containers(
            mesh_dir=Path(mesh_path).parent,
            contact_offset=contact_offset,
            domain_width=domain_width,
        )

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
            cold=cold,
            loss_weight=loss_weight,
            seed=seed,
            contact_offset=contact_offset,
            domain_width=domain_width,
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
    silicon_triangles: np.ndarray | None = None


def _silicon_design_nodes(
    actual_mesh: Path, n_nodes: int, real_mesh: bool
) -> tuple[Any, np.ndarray | None]:
    """The design set: the shared mesh's silicon nodes, else every node.

    Reads the ``slab`` + ``rib_silicon`` triangles -- the same pair
    ``ct_common.jl`` collects -- and keeps the nodes they touch. The synthetic
    fallback grid has no physical groups at all, and a mesh whose silicon
    groups cannot be read would silently shrink the design to nothing, so both
    fall back to one variable per node rather than guessing.

    Returns ``(design_nodes, silicon_triangles)``; the triangulation (``None``
    when unavailable) is what the doping figures draw the field on, so they
    paint the device's own elements rather than a Delaunay fill of the domain.
    """
    from prismo.pipeline import DesignNodes
    from prismo.waveguide_mesh import read_mesh_silicon_triangulation

    if not real_mesh:
        return DesignNodes.all_nodes(n_nodes), None

    triangles = read_mesh_silicon_triangulation(actual_mesh)
    if triangles.size == 0:
        typer.echo(
            "      WARNING: no silicon physical groups in the mesh; "
            "keeping one design variable per node."
        )
        return DesignNodes.all_nodes(n_nodes), None

    indices = np.unique(triangles.ravel()).astype(np.intp)
    return DesignNodes(indices=indices, n_mesh_nodes=n_nodes), triangles


def _local_geometry(
    geometry_cls: Any,
    mesh_size: float | None,
    contact_offset: float | None = None,
    domain_width: float | None = None,
) -> Any:
    """Rib geometry at the requested size knobs (local mesh path).

    ``mesh_size`` is the core resolution, matching what the gyptis container
    reads from ``PRISMO_GYPTIS_MESH_SIZE``; the junction and bulk sizes follow
    it at the class defaults' own ratios (half the core at the junction, 2.5x it
    in the bulk), so one knob refines the whole local mesh proportionally.
    ``contact_offset`` and ``domain_width`` mirror the gyptis
    ``PRISMO_GYPTIS_CONTACT_OFFSET`` / ``PRISMO_GYPTIS_WIDTH`` knobs; ``None``
    keeps the class defaults.
    """
    kwargs: dict[str, float] = {}
    if mesh_size is not None:
        if mesh_size <= 0.0:
            raise typer.BadParameter("--mesh-size must be positive [µm]")
        kwargs.update(
            mesh_res_junction=mesh_size / 2.0,
            mesh_res_core=mesh_size,
            mesh_res_bulk=mesh_size * 2.5,
        )
    if contact_offset is not None:
        if contact_offset <= 0.0:
            raise typer.BadParameter("--contact-offset must be positive [µm]")
        kwargs["contact_offset"] = contact_offset
    if domain_width is not None:
        if domain_width <= 0.0:
            raise typer.BadParameter("--domain-width must be positive [µm]")
        kwargs["box_width"] = domain_width
    geometry = geometry_cls(**kwargs)
    # Same rule as the gyptis author: both contact footprints inside the box.
    contact_outer = (
        geometry.rib_right + geometry.contact_offset + geometry.contact_width
    )
    if contact_outer >= geometry.box_width / 2.0:
        raise typer.BadParameter(
            f"--contact-offset {geometry.contact_offset} µm puts the contact edge at "
            f"{contact_outer} µm, outside the {geometry.box_width} µm wide domain"
        )
    return geometry


def build_pipeline_inputs(
    r_min: float,
    mesh_path: str,
    use_containers: bool,
    components: Any | None,
    mesh_size: float | None = None,
    seed: str = "lateral",
    contact_offset: float | None = None,
    domain_width: float | None = None,
) -> PipelineInputs:
    """Author the shared mesh and assemble the fixed pipeline inputs.

    ``prismo run`` (optimization), ``prismo validate-gradient`` (the gradient
    check) and ``scripts/benchmark_multiphysics_optimization.py`` all
    start here so they solve on exactly the same mesh, filter, transfer, and
    seed -- the benchmark used to assemble its own and drifted out of contract
    with this one. Container runs author the mesh via gyptis
    ``write_mesh`` and require live components; the in-process path builds the
    rib mesh locally. Lengths are micrometres throughout, ``r_min`` included.

    ``mesh_size`` refines the silicon: on the container path the caller has
    already passed it to :func:`init_tesseract_containers`, which is where the
    mesh author lives, so here it only sizes the local rib mesh. ``None`` keeps
    each author's default; likewise ``contact_offset`` / ``domain_width``, which
    on the container path were already passed to the gyptis author and here
    size the local mesh and the figure overlay. ``seed`` picks
    the initial junction topology (``pipeline.SEED_KINDS``).
    """
    import jax.numpy as jnp
    from prismo_shared.schemas import MeshRef

    from prismo.density_filter import assemble_filter_matrix
    from prismo.pipeline import build_design_transfer, seed_design_field
    from prismo.waveguide_mesh import (
        RibWaveguideGeometry,
        build_rib_waveguide_mesh,
        read_mesh_node_coordinates,
    )

    mesh_path_obj = Path(mesh_path)

    typer.echo("Generating waveguide mesh...")
    geometry = _local_geometry(
        RibWaveguideGeometry,
        mesh_size,
        contact_offset=contact_offset,
        domain_width=domain_width,
    )
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
        # Micrometres, like every real mesh this app reads: a 20 nm
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
    design_nodes, silicon_triangles = _silicon_design_nodes(
        actual_mesh, n_nodes, real_mesh
    )
    design_coords = coords[design_nodes.indices]
    typer.echo(
        f"      {len(design_nodes)} design variables on silicon nodes "
        f"({n_nodes - len(design_nodes)} non-silicon nodes carry no variable)"
    )

    typer.echo("Building density filter matrix...")
    H_sparse = assemble_filter_matrix(design_coords, r_min=r_min)
    H_dense = jnp.asarray(H_sparse.toarray())
    H_sum = jnp.sum(H_dense, axis=1)
    # Seed a signed P/N junction in every run path (sign(theta) is a free
    # design variable, so the optimizer can move or dissolve it). Seeded on the
    # design nodes, so the junction splits the silicon at its own median x
    # rather than the whole domain's; ``seed`` picks the topology.
    theta_init = seed_design_field(design_coords, seed)
    typer.echo(f"      Seed: {seed} junction")

    design_transfer = None
    if use_containers:
        # Carry the full nodal permittivity field onto the gyptis design cells
        # instead of the identity fallback, so a fixed-mean topology change moves
        # neff through the real eigenmode solve. The transfer is
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
        silicon_triangles=silicon_triangles,
    )


def _container_overlay_geometry(inputs: PipelineInputs) -> Any:
    """The figure overlay frame for a container run, from the gyptis mesh itself.

    ``RibWaveguideGeometry`` describes the local mesh author's frame: y from 0
    at the substrate bottom, 0.5 µm substrate. The gyptis author centres its
    layer stack on y = 0 with a 0.35 µm substrate, so drawing the local frame
    over container figures put the rib outline and shading off the device.
    Derive the rib rectangle from the design-cell vertices (the rib-interior
    triangles of the shared mesh), the domain half-width from the
    node coordinates, and the slab/contact dimensions from the values both mesh
    authors share. Falls back to the local geometry when the vertices are
    unavailable.
    """
    from prismo.outputs import OverlayGeometry

    if inputs.design_vertices is None or inputs.design_vertices.size == 0:
        return inputs.geometry
    verts = inputs.design_vertices.reshape(-1, 2)
    local = inputs.geometry
    slab_top = float(verts[:, 1].min())
    return OverlayGeometry(
        rib_left=float(verts[:, 0].min()),
        rib_right=float(verts[:, 0].max()),
        slab_top=slab_top,
        rib_top=float(verts[:, 1].max()),
        # Slab thickness and contact footprint are the same in both authors;
        # only the vertical origin and substrate thickness differ.
        substrate_top=slab_top - local.slab_thickness,
        half_width=float(np.abs(inputs.coords[:, 0]).max()),
        contact_offset=local.contact_offset,
        contact_width=local.contact_width,
    )


def _optimized_mode_field(
    inputs: PipelineInputs,
    rho_opt: np.ndarray,
    components: Any | None,
    mode_index: int = 0,
) -> ModeField | None:
    """The tracked optical mode ``|E|`` at the optimized design.

    Re-runs the pipeline's own pre-eigensolve stages at ``rho_opt`` -- filter,
    doping, both ChargeTransport solves, Soref-Bennett, mesh transfer -- so the
    figure shows the mode of the *reverse-biased optimized device*, the same
    permittivity the final objective was evaluated on, rather than a
    separately-reconstructed one. Returns ``None`` when no gyptis backend is
    bound to answer the query. ``mode_index`` only labels the figure: the
    bundle's ``mode_field`` query is already bound to the mode it optimized.
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
    return ModeField(
        abs_e=abs_e, coords_um=coords_um, rib_bounds=rib_bounds, mode_index=mode_index
    )


def _optimized_swept_carriers(
    inputs: PipelineInputs, rho_opt: np.ndarray, components: Any | None
) -> np.ndarray:
    """``(n+p)(V_bias) - (n+p)(0 V)`` per node at the optimized design.

    Two warm ChargeTransport solves through the bound pipeline's own filter
    and doping map; feeds the swept-carriers-under-the-mode figure.
    """
    import jax.numpy as jnp

    from prismo.pipeline import carrier_fields, default_components

    bundle = components if components is not None else default_components()
    n0, p0, n1, p1 = carrier_fields(
        jnp.asarray(rho_opt, dtype=jnp.float64),
        H=inputs.H_dense,
        H_sum=inputs.H_sum,
        mesh_ref=inputs.mesh_ref,
        design_nodes=inputs.design_nodes,
        components=bundle,
    )
    return np.asarray(n1 + p1 - n0 - p0, dtype=float)


def _clear_live_frames(output_dir: str | Path) -> int:
    """Delete ``doping_field_<n>.png`` frames a previous run left behind.

    The live frames are written one per optimizer iteration; a shorter run
    after a longer one would otherwise leave stale frames that read as part of
    its own trajectory. Returns the number of frames removed.
    """
    out = Path(output_dir)
    if not out.is_dir():
        return 0
    removed = 0
    for frame in out.glob(f"{_LIVE_FRAME_PREFIX}*.png"):
        suffix = frame.stem[len(_LIVE_FRAME_PREFIX) :]
        if suffix.isdigit():
            frame.unlink()
            removed += 1
    return removed


def _mode_overlap_weights(
    inputs: PipelineInputs, components: Any | None, loss_weight: float
) -> np.ndarray | None:
    """The frozen mode-overlap weights the loss term needs.

    One background eigensolve + adjoint through the bound gyptis component.
    Always attempted so every run reports its modal loss; a backend that cannot
    answer (no gyptis bound, an image predating the field VJP) costs the loss
    *report* when the weight is zero and is a hard error when the loss is part
    of the objective.
    """
    from prismo.pipeline import default_components, read_mode_overlap

    bundle = components if components is not None else default_components()
    n_cells = (
        int(inputs.design_transfer.shape[0])
        if inputs.design_transfer is not None
        else int(inputs.n_nodes)
    )
    try:
        weights = read_mode_overlap(bundle, n_cells)
    except Exception as exc:
        if loss_weight > 0.0:
            raise RuntimeError(
                "--loss-weight needs the mode-overlap weights from a live gyptis "
                f"backend ({type(exc).__name__}: {exc})"
            ) from exc
        typer.echo(
            "      Modal loss not reported: mode-overlap weights unavailable "
            f"({type(exc).__name__})"
        )
        return None
    typer.echo(
        f"      Mode-overlap weights: {n_cells} design cells, "
        f"sum d(neff²)/dε = {float(weights.sum()):.4f}"
    )
    return weights


def _mode_overlap_if_weighted(
    inputs: PipelineInputs, components: Any | None, loss_weight: float
) -> np.ndarray | None:
    """Overlap weights only when the loss is in the objective (diagnostic paths)."""
    if loss_weight > 0.0:
        return _mode_overlap_weights(inputs, components, loss_weight)
    return None


def _doping_frames(
    history: list[dict[str, Any]],
    H_dense: Any,
    H_sum: Any,
    design_nodes: Any,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Full-node net doping per evaluated design, for the doping animation.

    Each history record carries the raw design vector it evaluated; the
    solvers saw the *filtered* field, so filter, scatter to full node order
    and map to doping here -- the same three steps the pipeline takes.
    Returns the frames and the matching records (entries predating the
    ``design`` key are skipped, so an old checkpoint still animates what it
    has).
    """
    from prismo.pipeline import doping_from_theta

    H_np = np.asarray(H_dense)
    H_sum_np = np.asarray(H_sum)
    frames: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []
    for entry in history:
        design = entry.get("design")
        if design is None:
            continue
        theta = np.asarray(design, dtype=float)
        theta_tilde = H_np @ theta / H_sum_np
        frames.append(
            np.asarray(doping_from_theta(design_nodes.scatter_numpy(theta_tilde)))
        )
        kept.append(entry)
    return frames, kept


def _best_history_entry(history: list[dict[str, Any]]) -> dict[str, Any]:
    """The accepted design's record: the best *objective* (Δneff when unweighted).

    The optimizer only accepts improving steps, so the last entry may be a
    rejected trial; older histories carry no ``objective`` key
    and fall back to Δneff.
    """
    return max(history, key=lambda e: e.get("objective", e["delta_n_eff"]))


def _reset_chargetransport(components: Any | None) -> bool:
    """Drop the ChargeTransport worker's warm solutions; ``False`` if no seam."""
    from prismo.pipeline import default_components

    bundle = components if components is not None else default_components()
    reset = getattr(bundle, "reset_chargetransport", None)
    if reset is None:
        return False
    reset()
    return True


def _cold_reevaluation(
    bound_terms: Any,
    rho_opt: np.ndarray,
    warm_delta_neff: float,
    components: Any | None,
) -> ColdReevaluation | None:
    """Re-solve the reported design cold and compare with the optimizer's value.

    The headline Δneff must be a property of the design, not of the solve
    history that produced it: the ChargeTransport worker's warm
    state is dropped, the best design evaluated once more, and both numbers
    reported. Returns ``None`` (with a note) when no reset seam is bound, and
    ``None`` with a warning when the cold solve itself fails: a design that
    only solves from a warm start is a finding worth the headline figures, not
    a crash after a finished optimization. ``bound_terms`` is the
    ``pipeline_with_terms`` partial: the comparison is on Δneff itself, not on
    a loss-penalized objective.
    """
    from prismo.outputs import ColdReevaluation

    if not _reset_chargetransport(components):
        typer.echo(
            "      Cold re-evaluation skipped: no ChargeTransport reset seam bound."
        )
        return None
    try:
        _objective, terms = bound_terms(np.asarray(rho_opt, dtype=float))
        cold_value = float(terms.delta_neff)
        cold_loss = float(terms.modal_loss_db_cm)
    except Exception as exc:
        typer.echo(
            "      WARNING: cold re-solve of the reported design FAILED "
            f"({type(exc).__name__}); the warm value is reported, but this "
            "design solved only from a warm start -- its objective is "
            "path-dependent until the cold solve succeeds."
        )
        return None
    result = ColdReevaluation(
        warm_delta_neff=float(warm_delta_neff), cold_delta_neff=cold_value
    )
    typer.echo(f"      Delta_n_eff (warm, optimizer) = {result.warm_delta_neff:+.6e}")
    typer.echo(f"      Delta_n_eff (cold re-solve)   = {result.cold_delta_neff:+.6e}")
    if np.isfinite(cold_loss):
        typer.echo(f"      Modal loss (0 V, cold re-solve) = {cold_loss:.4g} dB/cm")
    if not result.passed:
        typer.echo(
            "      WARNING: warm/cold Delta_n_eff disagree by "
            f"{result.rel_discrepancy:.2e} relative (tolerance "
            f"{result.tolerance:.0e}); the warm optimum depended on the solve "
            "history, not only on the design."
        )
    return result


def _run_pipeline(
    r_min: float,
    max_iter: int,
    ftol_rel: float,
    mesh_path: str,
    output_dir: str,
    no_jit: bool,
    use_containers: bool,
    mesh_size: float | None = None,
    components: Any | None = None,
    move_limit: float = _DEFAULT_MOVE_LIMIT,
    mode_index: int = 0,
    loss_weight: float = 0.0,
    seed: str = "lateral",
    contact_offset: float | None = None,
    domain_width: float | None = None,
) -> None:
    from prismo.optimizer import OptimizationCancelled, optimize_doping
    from prismo.outputs import generate_outputs, plot_live_doping_field
    from prismo.pipeline import (
        doping_from_theta,
        loss_figure_of_merit_v_db,
        pipeline_with_terms,
        vpi_lpi_v_cm,
    )
    from prismo.pipeline import pipeline as pipeline_fn

    typer.echo("=== PRISMO Pipeline ===")
    typer.echo()

    removed = _clear_live_frames(output_dir)
    if removed:
        typer.echo(f"      Cleared {removed} live doping frame(s) from a previous run")

    typer.echo("[1/3] Preparing pipeline inputs...")
    inputs = build_pipeline_inputs(
        r_min,
        mesh_path,
        use_containers,
        components,
        mesh_size=mesh_size,
        seed=seed,
        contact_offset=contact_offset,
        domain_width=domain_width,
    )
    typer.echo(
        "      Target mode: "
        + ("fundamental (index 0)" if mode_index == 0 else f"guided index {mode_index}")
    )
    # Loss-aware objective: Δneff - w·alpha with alpha the first-order modal
    # free-carrier loss of the unbiased device; w = 0 reports the loss only.
    mode_overlap = _mode_overlap_weights(inputs, components, loss_weight)
    typer.echo(
        f"      Loss weight: {loss_weight:g}"
        + (" (Δneff alone is optimized)" if loss_weight == 0.0 else " [neff per dB/cm]")
    )
    # Container figures draw the gyptis frame, not the local author's
    #: the two meshes differ in vertical origin and substrate
    # thickness, and the overlay must describe the mesh the nodes came from.
    geometry = (
        _container_overlay_geometry(inputs) if use_containers else inputs.geometry
    )
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
            H_np = np.asarray(H_dense)
            H_sum_np = np.asarray(H_sum)

            def on_iteration(iteration: int, theta: np.ndarray) -> None:
                # Snapshot the doping the solvers actually see: the optimizer
                # hands back the raw design vector, but the physics runs on the
                # density-filtered field, so filter before mapping to doping.
                theta_tilde = H_np @ theta / H_sum_np
                name = f"{_LIVE_FRAME_PREFIX}{iteration}"
                plot_live_doping_field(
                    np.asarray(
                        doping_from_theta(design_nodes.scatter_numpy(theta_tilde))
                    ),
                    coords,
                    iteration,
                    geometry=geometry,
                    output_dir=output_dir,
                    name=name,
                    triangles=inputs.silicon_triangles,
                )

        rho_opt, history = optimize_doping(
            initial_rho=np.asarray(theta_init, dtype=float),
            n_nodes=len(design_nodes),
            H=H_dense,
            H_sum=H_sum,
            max_iter=optimization_max_iter,
            ftol_rel=optimization_ftol_rel,
            move_limit=move_limit,
            checkpoint_path=Path(output_dir) / _CHECKPOINT_NAME,
            use_jit=not no_jit,
            on_iteration=on_iteration,
            design_transfer=design_transfer,
            design_nodes=design_nodes,
            mesh_ref=mesh_ref,
            components=components,
            loss_weight=loss_weight,
            mode_overlap=mode_overlap,
        )
        typer.echo(f"      Optimization complete: {len(history)} iterations")
    except OptimizationCancelled:
        typer.echo("      Optimization cancelled by user.")
        return

    # Bind the *same* pipeline the optimizer drove -- filter, mesh_ref and
    # transfer included -- for the cold re-evaluation and the figure's finite
    # differences. Omitting them made the figure probe an unfiltered pipeline
    # with ChargeTransport on its 1D fallback device, i.e. a different function
    # than the one that was optimized.
    bound_kwargs = dict(
        H=H_dense,
        H_sum=H_sum,
        mesh_ref=mesh_ref,
        design_transfer=design_transfer,
        design_nodes=design_nodes,
        components=components,
        loss_weight=loss_weight,
        mode_overlap=mode_overlap,
    )
    bound_pipeline = partial(pipeline_fn, **bound_kwargs)
    bound_terms = partial(pipeline_with_terms, **bound_kwargs)

    # The mode figure is a post-hoc query, so a backend that cannot answer it --
    # an image predating the ``mode_field`` operation, say -- must cost one
    # figure, not every figure of a finished multi-minute optimization. It runs
    # before the cold re-evaluation so it sees the worker still warm at the
    # reported design: a design that solves only warm would otherwise lose its
    # mode figure to the reset.
    try:
        mode_field = _optimized_mode_field(inputs, rho_opt, components, mode_index)
    except Exception as exc:
        typer.echo(f"      WARNING: mode figure skipped ({exc})")
        mode_field = None
    # Same warm-state window and the same soft failure for the swept-carriers
    # figure (two ChargeTransport solves at the reported design).
    try:
        swept_carriers = _optimized_swept_carriers(inputs, rho_opt, components)
    except Exception as exc:
        typer.echo(f"      WARNING: depletion figure skipped ({exc})")
        swept_carriers = None

    cold = None
    if history:
        # The reported design is the best one whose physics solved: the best
        # objective in the history (the optimizer only accepts improving
        # steps, so the last entry may be a rejected trial).
        best = _best_history_entry(history)
        warm_delta_neff = float(best["delta_n_eff"])
        if loss_weight > 0.0:
            typer.echo(f"      Best objective (warm) = {float(best['objective']):+.6e}")
        typer.echo(f"      Best Delta_n_eff (warm) = {warm_delta_neff:+.6e}")
        cold = _cold_reevaluation(bound_terms, rho_opt, warm_delta_neff, components)
        headline = cold.cold_delta_neff if cold is not None else warm_delta_neff
        # VπLπ headline (V·cm): the field-standard modulation efficiency,
        # reported from Δneff at the fixed -5 V bias (smaller |VπLπ| better),
        # computed from the cold value when available.
        typer.echo(f"      VpiLpi = {vpi_lpi_v_cm(headline):+.4e} V·cm")
        # Modal free-carrier loss of the reported design and the VπLπ·alpha figure
        # of merit the literature compares on.
        warm_loss = best.get("modal_loss_db_cm")
        if warm_loss is not None:
            typer.echo(f"      Modal loss (0 V, warm) = {float(warm_loss):.4g} dB/cm")
            typer.echo(
                "      VpiLpi x loss = "
                f"{loss_figure_of_merit_v_db(headline, float(warm_loss)):+.4g} V·dB"
            )

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
    doping_frames, animated_history = _doping_frames(
        history, H_dense, H_sum, design_nodes
    )
    plot_paths = generate_outputs(
        rho_initial=plot_initial,
        rho_opt=plot_opt,
        history=history,
        # Node coordinates are micrometres on both paths, which is
        # what the figures plot in -- no conversion at this seam.
        mesh_coords=coords,
        geometry=geometry,
        pipeline_fn=bound_pipeline,
        ftol_rel=optimization_ftol_rel,
        gradient_validation_directions=1 if use_containers else 3,
        gradient_validation_steps=(np.logspace(-4, -2, 3) if use_containers else None),
        # Explicit on both paths: the figure's finite differences probe the
        # bound pipeline, which takes a design-node vector, while ``rho_opt``
        # above has already been scattered to full node order for plotting.
        gradient_validation_rho=rho_initial if use_containers else rho_opt,
        mode_field=mode_field,
        output_dir=output_dir,
        # The silicon triangulation: the doping figure paints the device's own
        # mesh elements, leaving the oxide blank instead of Delaunay-smearing
        # the design field across it.
        mesh_triangles=inputs.silicon_triangles,
        cold_reevaluation=cold,
        design_history=doping_frames,
        animation_history=animated_history,
        swept_carriers=swept_carriers,
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
    cold: bool = False,
    loss_weight: float = 0.0,
    seed: str = "lateral",
    contact_offset: float | None = None,
    domain_width: float | None = None,
) -> GradientValidationResult:
    """Check the composed gradient at the seeded design and write the figure.

    Unlike the optimization run, this binds the *full* pipeline -- filter,
    ``mesh_ref`` (so ChargeTransport solves on the shared 2D grid, not its 1D
    fallback), and design transfer -- into the function the finite-difference
    sweep probes, so the CT adjoint is the thing being proven.
    ``cold`` resets the ChargeTransport worker before every evaluation.
    ``loss_weight > 0`` validates the loss-penalized objective, i.e. the loss
    term's adjoint through the 0 V carriers too.
    """
    from prismo.outputs import validate_gradient as validate_gradient_fn
    from prismo.pipeline import pipeline as pipeline_fn

    typer.echo("=== PRISMO Gradient Validation ===")
    typer.echo()

    typer.echo("[1/2] Preparing pipeline inputs...")
    inputs = build_pipeline_inputs(
        r_min,
        mesh_path,
        use_containers,
        components,
        seed=seed,
        contact_offset=contact_offset,
        domain_width=domain_width,
    )
    mode_overlap = _mode_overlap_if_weighted(inputs, components, loss_weight)

    typer.echo("[2/2] Checking adjoint against central finite differences...")
    before_evaluation = None
    if cold:
        if not _reset_chargetransport(components):
            raise RuntimeError(
                "--cold requires a ChargeTransport backend with a reset seam"
            )
        before_evaluation = partial(_reset_chargetransport, components)
        typer.echo(
            "      Cold start: resetting the ChargeTransport worker before "
            "every evaluation"
        )
    bound_pipeline = partial(
        pipeline_fn,
        H=inputs.H_dense,
        H_sum=inputs.H_sum,
        mesh_ref=inputs.mesh_ref,
        design_transfer=inputs.design_transfer,
        design_nodes=inputs.design_nodes,
        components=components,
        loss_weight=loss_weight,
        mode_overlap=mode_overlap,
    )
    result = validate_gradient_fn(
        bound_pipeline,
        inputs.theta_init,
        n_directions=n_directions,
        step_sizes=step_sizes,
        tolerance=tolerance,
        output_dir=output_dir,
        before_evaluation=before_evaluation,
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


_DEFAULT_PROBE_SPACING = 1e-5
_DEFAULT_PROBE_POINTS = 21


@app.command(name="probe-objective")
def probe_objective(
    r_min: float = typer.Option(_DEFAULT_R_MIN, help="Density filter radius [µm]"),
    design: str = typer.Option(
        None,
        "--design",
        help="checkpoint.json whose rho_opt is the centre of the scan "
        "(default: the seeded junction)",
    ),
    direction: str = typer.Option(
        "gradient",
        help="'gradient' (normalized adjoint gradient at the centre, projected "
        "off rail-pinned variables) or 'random' (seeded unit vector)",
    ),
    spacing: float = typer.Option(
        _DEFAULT_PROBE_SPACING, help="Uniform step t between samples along d"
    ),
    n_points: int = typer.Option(
        _DEFAULT_PROBE_POINTS, help="Number of samples (odd keeps t = 0 centred)"
    ),
    mesh_path: str = typer.Option(
        str(_DEFAULT_MESH), help="Path to waveguide .msh file"
    ),
    output_dir: str = typer.Option(
        str(_DEFAULT_OUTPUT_DIR), help="Directory for objective_line_scan.{pdf,json}"
    ),
    use_containers: bool = typer.Option(
        False,
        "--use-containers",
        help="Run tesseract components via Docker containers",
    ),
    cold: bool = typer.Option(
        False,
        "--cold",
        help="Reset the ChargeTransport worker before every evaluation",
    ),
    loss_weight: float = typer.Option(
        0.0,
        "--loss-weight",
        min=0.0,
        help="Scan the loss-penalized objective Δneff - w·alpha",
    ),
    seed: str = _seed_option(),
    contact_offset: float = _contact_offset_option(),
    domain_width: float = _domain_width_option(),
) -> None:
    """Scan Δneff along one direction at fine spacing.

    Separates a kink in θ → Δneff from an evaluation noise floor: samples
    ``f(θ₀ + t·d)`` on a uniform grid, fits a quadratic, and reports the fit
    residual and the white-noise amplitude implied by second differences,
    alongside the adjoint's directional derivative against the fitted slope.
    """
    from prismo.pipeline import (
        PipelineComponents,
        init_tesseract_containers,
        teardown_containers,
    )

    _check_geometry_knobs(seed, contact_offset, domain_width)
    components: PipelineComponents | None = None
    if use_containers:
        typer.echo("Starting tesseract Docker containers...")
        components = init_tesseract_containers(
            mesh_dir=Path(mesh_path).parent,
            contact_offset=contact_offset,
            domain_width=domain_width,
        )
    try:
        _run_objective_probe(
            r_min=r_min,
            mesh_path=mesh_path,
            output_dir=output_dir,
            design_path=design,
            direction=direction,
            spacing=spacing,
            n_points=n_points,
            use_containers=use_containers,
            components=components,
            cold=cold,
            loss_weight=loss_weight,
            seed=seed,
            contact_offset=contact_offset,
            domain_width=domain_width,
        )
    finally:
        if use_containers and components is not None:
            typer.echo("Stopping tesseract containers...")
            teardown_containers(components)


def _load_checkpoint_design(path: str | Path, n_design: int) -> np.ndarray:
    """``rho_opt`` from a ``checkpoint.json`` written by the optimizer."""
    import json

    payload = json.loads(Path(path).read_text())
    rho = np.asarray(payload["rho_opt"], dtype=float)
    if rho.shape != (n_design,):
        raise ValueError(
            f"{path} holds {rho.size} design variables but this mesh has "
            f"{n_design}; probe the checkpoint on the mesh that produced it"
        )
    return rho


def _run_objective_probe(
    r_min: float,
    mesh_path: str,
    output_dir: str,
    design_path: str | None,
    direction: str,
    spacing: float,
    n_points: int,
    use_containers: bool,
    components: Any | None = None,
    cold: bool = False,
    loss_weight: float = 0.0,
    seed: str = "lateral",
    contact_offset: float | None = None,
    domain_width: float | None = None,
) -> ObjectiveLineScan:
    """Line-scan the bound pipeline around a design."""
    import jax.numpy as jnp

    from prismo.outputs import feasible_offsets, probe_direction, scan_objective_line
    from prismo.pipeline import pipeline as pipeline_fn

    if spacing <= 0.0:
        raise typer.BadParameter("--spacing must be positive")
    if n_points < 3:
        raise typer.BadParameter("--n-points must be at least 3")
    if direction not in ("gradient", "random"):
        raise typer.BadParameter("--direction must be 'gradient' or 'random'")

    typer.echo("=== PRISMO Objective Line Scan ===")
    typer.echo()
    typer.echo("[1/2] Preparing pipeline inputs...")
    inputs = build_pipeline_inputs(
        r_min,
        mesh_path,
        use_containers,
        components,
        seed=seed,
        contact_offset=contact_offset,
        domain_width=domain_width,
    )
    mode_overlap = _mode_overlap_if_weighted(inputs, components, loss_weight)

    rho = jnp.asarray(inputs.theta_init, dtype=jnp.float64)
    if design_path is not None:
        rho = jnp.asarray(_load_checkpoint_design(design_path, rho.shape[0]))
        typer.echo(f"      Centre: rho_opt from {design_path}")
    else:
        typer.echo("      Centre: the seeded junction")

    before_evaluation = None
    if cold:
        if not _reset_chargetransport(components):
            raise RuntimeError(
                "--cold requires a ChargeTransport backend with a reset seam"
            )
        before_evaluation = partial(_reset_chargetransport, components)
        typer.echo(
            "      Cold start: resetting the ChargeTransport worker before "
            "every evaluation"
        )

    bound_pipeline = partial(
        pipeline_fn,
        H=inputs.H_dense,
        H_sum=inputs.H_sum,
        mesh_ref=inputs.mesh_ref,
        design_transfer=inputs.design_transfer,
        design_nodes=inputs.design_nodes,
        components=components,
        loss_weight=loss_weight,
        mode_overlap=mode_overlap,
    )

    typer.echo(f"[2/2] Scanning along the {direction} direction...")
    d, gradient = probe_direction(bound_pipeline, rho, direction, before_evaluation)

    half = n_points // 2
    offsets = spacing * np.arange(-half, n_points - half, dtype=float)
    kept = feasible_offsets(rho, d, offsets)
    if kept.size < offsets.size:
        typer.echo(
            f"      {offsets.size - kept.size} offset(s) dropped to stay inside "
            "the [-1, 1] box"
        )
    scan = scan_objective_line(
        bound_pipeline,
        rho,
        d,
        kept,
        output_dir=output_dir,
        before_evaluation=before_evaluation,
        gradient=gradient,
    )

    for offset, value, residual in zip(
        scan.offsets, scan.values, scan.residuals, strict=True
    ):
        typer.echo(f"      t={offset:+.3e}  f={value:+.9e}  residual={residual:+.3e}")
    typer.echo(
        f"      Quadratic-fit residual: rms {scan.rms_rel_residual:.2e}, "
        f"max {scan.max_rel_residual:.2e} (relative to |f(θ₀)|)"
    )
    typer.echo(
        f"      Noise floor from second differences: {scan.noise_estimate:.2e} relative"
    )
    typer.echo(
        f"      Slope along d: adjoint {scan.adjoint_slope:+.6e}, "
        f"fitted {scan.fitted_slope:+.6e}"
    )
    typer.echo(f"      Figure: {scan.figure_path}")
    typer.echo(f"      Data:   {scan.json_path}")
    typer.echo()
    typer.echo("=== Done ===")
    return scan


_DEFAULT_ANIMATION_FPS = 6


@app.command()
def animate(
    checkpoint: str = typer.Option(
        str(_DEFAULT_OUTPUT_DIR / _CHECKPOINT_NAME),
        "--checkpoint",
        help="checkpoint.json of the run to replay (every record carries its design)",
    ),
    mesh_path: str = typer.Option(
        str(_DEFAULT_MESH), help="The run's shared .msh (read, not re-authored)"
    ),
    r_min: float = typer.Option(
        _DEFAULT_R_MIN, help="Density filter radius the run used [µm]"
    ),
    output_dir: str = typer.Option(
        str(_DEFAULT_OUTPUT_DIR), help="Directory for doping_evolution.{gif,mp4}"
    ),
    fps: int = typer.Option(_DEFAULT_ANIMATION_FPS, min=1, help="Frames per second"),
    fmt: str = typer.Option(
        "gif,mp4",
        "--format",
        help="Comma-separated encoders: gif (Pillow), mp4 (ffmpeg)",
    ),
    contact_offset: float = typer.Option(
        0.2, "--contact-offset", help="Contact gap the run used, for the overlay [µm]"
    ),
) -> None:
    """Replay a run's doping field as an animation, from its checkpoint alone.

    No solver is touched: each history record carries the design it
    evaluated, which is filtered, scattered and mapped to net doping exactly
    as the run did, and drawn on the run's own mesh with a colour scale fixed
    across the whole run. Rejected trials are labelled.
    """
    formats = tuple(part.strip() for part in fmt.split(",") if part.strip())
    paths = _run_animate(
        checkpoint_path=checkpoint,
        mesh_path=mesh_path,
        r_min=r_min,
        output_dir=output_dir,
        fps=fps,
        formats=formats,
        contact_offset=contact_offset,
    )
    if not paths:
        raise typer.Exit(code=1)


def _overlay_from_mesh(
    coords: np.ndarray, design_nodes: Any, contact_offset: float
) -> Any:
    """An overlay frame read off the mesh itself (no geometry object at hand).

    The rib box comes from the design nodes the same way the seeds find it
    (``pipeline._seed_rib_box``); the contact footprint is the one knob the
    mesh does not reveal, so the caller passes it.
    """
    from prismo.outputs import OverlayGeometry
    from prismo.pipeline import _seed_rib_box
    from prismo.waveguide_mesh import RibWaveguideGeometry

    design_coords = coords[design_nodes.indices]
    x_left, x_right, slab_top, rib_top, _x_centre = _seed_rib_box(design_coords)
    return OverlayGeometry(
        rib_left=x_left,
        rib_right=x_right,
        slab_top=slab_top,
        rib_top=rib_top,
        substrate_top=float(design_coords[:, 1].min()),
        half_width=float(np.abs(coords[:, 0]).max()),
        contact_offset=contact_offset,
        contact_width=RibWaveguideGeometry().contact_width,
    )


def _run_animate(
    checkpoint_path: str,
    mesh_path: str,
    r_min: float,
    output_dir: str,
    fps: int = _DEFAULT_ANIMATION_FPS,
    formats: tuple[str, ...] = ("gif", "mp4"),
    contact_offset: float = 0.2,
) -> list[Path]:
    """Build the doping animation for a finished (or killed) run."""
    import json

    import jax.numpy as jnp

    from prismo.density_filter import assemble_filter_matrix
    from prismo.outputs import animate_doping_evolution
    from prismo.waveguide_mesh import read_mesh_node_coordinates

    typer.echo("=== PRISMO Doping Animation ===")
    payload = json.loads(Path(checkpoint_path).read_text())
    history = list(payload.get("history", []))
    with_design = [entry for entry in history if entry.get("design") is not None]
    if not with_design:
        typer.echo(
            f"      {checkpoint_path} carries no per-iteration designs "
            "(written by runs predating the animation); nothing to replay."
        )
        return []
    typer.echo(f"      {len(with_design)} evaluated designs in {checkpoint_path}")

    coords = read_mesh_node_coordinates(mesh_path)
    if coords.shape[0] == 0:
        raise RuntimeError(f"could not read node coordinates from {mesh_path}")
    n_nodes = coords.shape[0]
    design_nodes, silicon_triangles = _silicon_design_nodes(
        Path(mesh_path), n_nodes, real_mesh=True
    )
    n_design = len(design_nodes)
    if len(with_design[0]["design"]) != n_design:
        raise ValueError(
            f"{checkpoint_path} holds {len(with_design[0]['design'])} design "
            f"variables but {mesh_path} has {n_design} silicon nodes; replay the "
            "checkpoint on the mesh that produced it"
        )
    H_sparse = assemble_filter_matrix(coords[design_nodes.indices], r_min=r_min)
    H_dense = jnp.asarray(H_sparse.toarray())
    H_sum = jnp.sum(H_dense, axis=1)
    frames, kept = _doping_frames(history, H_dense, H_sum, design_nodes)
    geometry = _overlay_from_mesh(coords, design_nodes, contact_offset)

    paths = animate_doping_evolution(
        frames,
        kept,
        coords,
        geometry=geometry,
        output_dir=output_dir,
        triangles=silicon_triangles,
        fps=fps,
        formats=formats,
    )
    for path in paths:
        typer.echo(f"      {path}")
    typer.echo("=== Done ===")
    return paths


def entrypoint() -> None:
    """CLI entrypoint for the application."""
    app()


if __name__ == "__main__":
    entrypoint()
