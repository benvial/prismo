# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for tesseract_photonic_waveguide_devsim
# Semiconductor drift-diffusion component (DEVSIM).
#
# Real implementation: 1D PN junction drift-diffusion solve + implicit-diff VJP
# via Newton Jacobian extraction + adjoint solve, per tickets 02 and 09.
# Falls back to identity stub when DEVSIM is not installed.

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_photonic_waveguide_shared.schemas import MeshRef

#
# Schemas
#


class InputSchema(BaseModel):
    """Inputs to the DEVSIM drift-diffusion solve.

    Attributes:
        doping: Net doping concentration at every mesh node [m⁻³].
            Positive = n-type, negative = p-type.
        mesh_ref: Optional reference to a shared Gmsh 2D mesh file.
            When present, the solve runs in 2D on the imported mesh.
            When absent, falls back to a 1D PN junction (test path).
    """

    doping: Differentiable[Array[(None,), Float64]]
    mesh_ref: MeshRef | None = None


class OutputSchema(BaseModel):
    """Outputs of the DEVSIM drift-diffusion solve.

    Attributes:
        charge: Total carrier concentration (electrons + holes) at every
            mesh node [m⁻³]. The Soref-Bennett coupling layer maps this to
            a refractive-index perturbation.
    """

    charge: Differentiable[Array[(None,), Float64]]


#
# Module-level state
#

# Cached after apply() so vector_jacobian_product() can re-extract the Jacobian.
# Keys: "device", "region", "n_nodes", "dim" (1 or 2)
_solve_state: dict[str, Any] | None = None


#
# Internal helpers
#


def _ensure_devsim() -> Any:
    """Import DEVSIM lazily; raises ImportError with a helpful message if absent."""
    import devsim  # type: ignore[import-untyped]

    return devsim


def _cleanup_device(devsim: Any, device: str) -> None:
    """Remove existing DEVSIM device if present (suppresses error on missing)."""
    devices = devsim.get_device_list()
    if device in devices:
        devsim.delete_device(device=device)


def _build_1d_pn_junction(
    devsim: Any, device: str, region: str, doping: np.ndarray, mesh_name: str = "mesh"
) -> None:
    """Create a 1D PN junction device with the given doping profile.

    Mesh: 1 μm silicon bar with node count matching the doping array.
    Physics: simple_physics drift-diffusion (Silicon).
    Contacts: ohmic contacts at both ends, grounded (0 V bias).
    """
    n = len(doping)
    length = 1e-6  # 1 μm silicon bar

    _cleanup_device(devsim, device)

    devsim.create_device(device=device)
    devsim.create_region(device=device, region=region, material="Silicon")

    devsim.create_1d_mesh(device=device, region=region, mesh=mesh_name)
    devsim.add_1d_mesh_line(
        device=device,
        region=region,
        mesh=mesh_name,
        tag="line",
        pos=0.0,
        ps=length,
        ns=n - 1,
    )
    devsim.add_1d_contact(
        device=device,
        region=region,
        mesh=mesh_name,
        name="anode",
        tag="line",
        pos=0.0,
        material="metal",
    )
    devsim.add_1d_contact(
        device=device,
        region=region,
        mesh=mesh_name,
        name="cathode",
        tag="line",
        pos=length,
        material="metal",
    )
    devsim.add_1d_region(
        device=device,
        region=region,
        mesh=mesh_name,
        tag="line",
        material="Silicon",
    )
    devsim.finalize_mesh(device=device, region=region)

    devsim.set_node_values(
        device=device,
        region=region,
        name="NetDoping",
        init_from="list",
        values=doping.tolist(),
    )

    from devsim.python_packages.simple_physics import (  # type: ignore[import-untyped]
        CreateSiliconDriftDiffusion,
    )

    CreateSiliconDriftDiffusion(device, region)

    for contact in devsim.get_contact_list(device=device):
        devsim.set_parameter(
            device=device,
            name=devsim.get_contact_parameter_name(
                device=device, contact=contact, parameter="bias"
            ),
            value=0.0,
        )


def _build_2d_pn_junction(
    devsim: Any,
    device: str,
    region: str,
    doping: np.ndarray,
    mesh_path: str,
    mesh_name: str = "mesh",
) -> None:
    """Create a 2D PN junction device on an imported Gmsh mesh.

    Mesh: imported from a ``.msh`` file via ``devsim.create_gmsh_mesh``.
    Physics: simple_physics drift-diffusion (Silicon).
    Doping: per-node NetDoping assigned on all mesh nodes.
    Contacts: identified by boundary-edge detection on the mesh (edges
        belonging to exactly one element are boundary edges; nodes on
        those edges are candidate contact nodes). They are partitioned
        into anode (x < median) and cathode (x ≥ median) — no hardcoded
        geometry.
    """
    n = len(doping)

    _cleanup_device(devsim, device)

    devsim.create_device(device=device)
    devsim.create_region(device=device, region=region, material="Silicon")

    devsim.create_gmsh_mesh(
        device=device, region=region, mesh=mesh_name, file=mesh_path
    )

    devsim.set_node_values(
        device=device,
        region=region,
        name="NetDoping",
        init_from="list",
        values=doping.tolist(),
    )

    from devsim.python_packages.simple_physics import (  # type: ignore[import-untyped]
        CreateSiliconDriftDiffusion,
    )

    CreateSiliconDriftDiffusion(device, region)

    _setup_2d_contacts_from_mesh(devsim, device, region, mesh_name)

    for contact in devsim.get_contact_list(device=device):
        devsim.set_parameter(
            device=device,
            name=devsim.get_contact_parameter_name(
                device=device, contact=contact, parameter="bias"
            ),
            value=0.0,
        )


def _setup_2d_contacts_from_mesh(
    devsim: Any, device: str, region: str, mesh_name: str
) -> None:
    """Identify and create ohmic contacts via boundary-edge detection.

    Detects boundary nodes by counting edge occurrences across elements:
    edges appearing exactly once are boundary edges; their incident
    nodes are candidate contact nodes. Boundary nodes are partitioned
    into anode (x-coordinate below median) and cathode (above median).
    """
    x_coords = np.array(
        devsim.get_node_model_values(device=device, region=region, name="x"),
        dtype=float,
    )

    elem_nodes_list = devsim.get_element_node_numbers(device=device, region=region)

    boundary_node_set: set[int] = set()
    edge: tuple[int, int]
    node_to_edges: dict[int, list[tuple[int, int]]] = {}

    for elem_nodes in elem_nodes_list:
        if len(elem_nodes) < 2:
            continue
        for i in range(len(elem_nodes)):
            a_i = int(elem_nodes[i])
            b_i = int(elem_nodes[(i + 1) % len(elem_nodes)])
            if a_i < b_i:
                edge = (a_i, b_i)
            else:
                edge = (b_i, a_i)
            node_to_edges.setdefault(a_i, []).append(edge)
            node_to_edges.setdefault(b_i, []).append(edge)

    edge_count: dict[tuple[int, int], int] = {}
    for edges in node_to_edges.values():
        for edge in edges:
            edge_count[edge] = edge_count.get(edge, 0) + 1

    for node, edges in node_to_edges.items():
        for edge in edges:
            if edge_count[edge] == 1:
                boundary_node_set.add(node)
                break

    x_center = float(np.median(x_coords))

    anode_nodes = sorted(
        [n for n in boundary_node_set if x_coords[n] < x_center]
    )
    cathode_nodes = sorted(
        [n for n in boundary_node_set if x_coords[n] >= x_center]
    )

    if anode_nodes:
        devsim.add_2d_mesh_line(
            device=device,
            region=region,
            mesh=mesh_name,
            tag="anode",
            ns=len(anode_nodes) - 1,
            ps=0,
        )
        devsim.add_2d_contact(
            device=device,
            region=region,
            mesh=mesh_name,
            name="anode",
            material="metal",
        )

    if cathode_nodes:
        devsim.add_2d_mesh_line(
            device=device,
            region=region,
            mesh=mesh_name,
            tag="cathode",
            ns=len(cathode_nodes) - 1,
            ps=0,
        )
        devsim.add_2d_contact(
            device=device,
            region=region,
            mesh=mesh_name,
            name="cathode",
            material="metal",
        )


#
# Required endpoint
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward drift-diffusion solve.

    Dispatch:
        - ``mesh_ref`` absent → 1D PN junction (``_build_1d_pn_junction``).
        - ``mesh_ref`` present → 2D drift-diffusion on imported Gmsh mesh
          (``_build_2d_pn_junction``).

    Args:
        inputs: Net doping at every mesh node [m⁻³] plus optional mesh reference.

    Returns:
        Total carrier concentration (electrons + holes) at every node [m⁻³].
    """
    doping = np.asarray(inputs.doping, dtype=float)

    try:
        devsim = _ensure_devsim()
    except ImportError:
        return OutputSchema(charge=doping.copy())

    device = "pn_junction"
    region = "silicon"
    mesh_name = "mesh"

    if inputs.mesh_ref is not None:
        _build_2d_pn_junction(
            devsim, device, region, doping, str(inputs.mesh_ref.path)
        )
        dim = 2
    else:
        _build_1d_pn_junction(devsim, device, region, doping, mesh_name)
        dim = 1

    devsim.solve(
        type="dc",
        absolute_error=1e-10,
        relative_error=1e-10,
        maximum_iterations=30,
    )

    electrons = np.array(
        devsim.get_node_model_values(
            device=device, region=region, name="Electrons"
        ),
        dtype=float,
    )
    holes = np.array(
        devsim.get_node_model_values(device=device, region=region, name="Holes"),
        dtype=float,
    )

    charge = electrons + holes

    global _solve_state
    _solve_state = {
        "device": device,
        "region": region,
        "n_nodes": len(doping),
        "dim": dim,
    }

    return OutputSchema(charge=charge)


#
# Optional endpoint
#


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, npt.ArrayLike],
) -> dict[str, npt.ArrayLike]:
    """Adjoint gradient pass via implicit differentiation.

    Works for both 1D and 2D solves (dim tracked in ``_solve_state["dim"]``).

    Implements the implicit-function-theorem VJP:

        dJ/d(doping) = λᵀ · ∂F/∂(doping)

    where A^T lambda = -u, A = dF/dx (Newton Jacobian), u = dJ/dx (objective
    sensitivity w.r.t. solution variables).

    For the ``charge = electrons + holes`` output and direct doping-input
    interface, ∂F/∂(doping) reduces to the NetDoping sensitivity of the
    Poisson equation rows (-q per node).

    In 2D the Jacobian is 3N×3N (N = mesh nodes) with the same per-node
    equation ordering as 1D: Potential, ElectronContinuity, HoleContinuity.

    Args:
        inputs: Same InputSchema as the preceding apply() call.
        vjp_inputs: Input fields to compute cotangents for ({"doping"}).
        vjp_outputs: Output fields the cotangent_vector was taken w.r.t.
            ({"charge"}).
        cotangent_vector: Cotangent on output fields, e.g.
            ``{"charge": v}`` where v is the vector ∂L/∂(charge).

    Returns:
        Dict mapping requested input fields to their cotangents, e.g.
        ``{"doping": ∂L/∂(doping)}``.
    """
    vjp: dict[str, npt.ArrayLike] = {}

    if "doping" not in vjp_inputs or "charge" not in vjp_outputs:
        return vjp

    cotangent = np.asarray(cotangent_vector["charge"], dtype=float)
    n = len(cotangent)

    try:
        devsim = _ensure_devsim()
        import scipy.sparse
        import scipy.sparse.linalg
    except ImportError:
        vjp["doping"] = cotangent.copy()
        return vjp

    global _solve_state
    if _solve_state is None:
        raise RuntimeError(
            "vector_jacobian_product called before apply(). "
            "Run apply() first to populate the solve state."
        )
    if _solve_state["n_nodes"] != n:
        raise RuntimeError(
            f"Input dimension mismatch: VJP expects n_nodes={_solve_state['n_nodes']}, "
            f"got {n}. Re-run apply() with matching doping array."
        )

    device = _solve_state["device"]
    region = _solve_state["region"]

    # --- 1. Extract Newton Jacobian A = ∂F/∂x ---
    r = devsim.get_matrix_and_rhs(device=device, region=region, format="csc")
    static = r["static"]
    n_eqs = len(static["rhs"])
    a_mat = scipy.sparse.csc_matrix(
        (static["av"], static["ai"], static["ap"]), shape=(n_eqs, n_eqs)
    )

    if n_eqs != 3 * n:
        raise RuntimeError(
            f"Jacobian size mismatch: expected 3*{n}={3*n} equations, "
            f"got {n_eqs}. The solve may have additional degrees of freedom."
        )

    # --- 2. Build objective RHS u = ∂J/∂x ---
    # charge_i = Electrons_i + Holes_i, so ∂(charge_i)/∂x has ones at the
    # ElectronContinuityEquation and HoleContinuityEquation rows for node i.
    # Equation order: [PotentialEquation, ElectronContinuityEquation,
    #                  HoleContinuityEquation] (simple_physics convention).
    u = np.zeros(3 * n)
    u[1 * n : 2 * n] = cotangent  # ElectronContinuityEquation rows
    u[2 * n : 3 * n] = cotangent  # HoleContinuityEquation rows

    # --- 3. Adjoint solve: A^T lambda = -u ---
    lam = scipy.sparse.linalg.spsolve(a_mat.T.tocsc(), -u)

    # --- 4. ∂F/∂(NetDoping): NetDoping enters Poisson via -q ---
    q = 1.602176634e-19  # electron charge (C)
    dF_ddoping = -q  # scalar per-node contribution to Poisson rows

    # VJP_i = λ[PotentialRow_i] * dF/d(doping_i)
    # Potential row for node i: 0*n + i
    result = lam[:n] * dF_ddoping

    vjp["doping"] = result
    return vjp
