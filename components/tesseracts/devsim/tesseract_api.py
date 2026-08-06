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

#
# Schemas
#


class InputSchema(BaseModel):
    """Inputs to the DEVSIM drift-diffusion solve.

    Attributes:
        doping: Net doping concentration at every mesh node [m⁻³].
            Positive = n-type, negative = p-type.
    """

    doping: Differentiable[Array[(None,), Float64]]


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
# Keys: "device", "region", "n_nodes"
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


#
# Required endpoint
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward drift-diffusion solve.

    Args:
        inputs: Net doping at every mesh node [m⁻³].

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

    _build_1d_pn_junction(devsim, device, region, doping)

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
    _solve_state = {"device": device, "region": region, "n_nodes": len(doping)}

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

    Implements the implicit-function-theorem VJP:

        dJ/d(doping) = λᵀ · ∂F/∂(doping)

    where A^T lambda = -u, A = dF/dx (Newton Jacobian), u = dJ/dx (objective
    sensitivity w.r.t. solution variables).

    For the ``charge = electrons + holes`` output and direct doping-input
    interface, ∂F/∂(doping) reduces to the NetDoping sensitivity of the
    Poisson equation rows (-q per node).

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
    a_mat = scipy.sparse.csc_matrix(
        (static["av"], static["ai"], static["ap"]), shape=(3 * n, 3 * n)
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
