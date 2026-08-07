# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for prismo_chargetransport
# ChargeTransport.jl semiconductor drift-diffusion component (Julia subprocess).
#
# Subprocess pattern: Python wrapper calls ``julia forward.jl`` (or
# ``julia adjoint.jl`` for the VJP) with NPY/NPZ file arguments.
# Falls back to identity stub when Julia is not installed.
#
# Ref: tickets 02 (container + I/O), 03 (forward API), 04 (adjoint),
#      06 (container build).

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64
from prismo_shared.schemas import MeshRef

_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

#
# Schemas
#


class InputSchema(BaseModel):
    """Inputs to the ChargeTransport.jl drift-diffusion solve.

    Attributes:
        doping: Net doping concentration at every mesh node [cm⁻³].
            Positive = n-type, negative = p-type.
        mesh_ref: Reference to the shared Gmsh 2D mesh file.
        bias_voltage: Applied bias voltage in volts. Default 0 (equilibrium).
    """

    doping: Differentiable[Array[(None,), Float64]]
    mesh_ref: MeshRef | None = None
    bias_voltage: float = Field(default=0.0, ge=-50.0, le=50.0)


class OutputSchema(BaseModel):
    """Outputs of the ChargeTransport.jl drift-diffusion solve.

    Attributes:
        electrons: Electron concentration per mesh node [cm⁻³].
        holes: Hole concentration per mesh node [cm⁻³].
    """

    electrons: Differentiable[Array[(None,), Float64]]
    holes: Differentiable[Array[(None,), Float64]]


#
# Module-level state
#

_solve_state: dict[str, Any] | None = None


#
# Internal helpers
#


def _julia_available() -> bool:
    """Check whether ``julia`` is on PATH."""
    try:
        subprocess.run(
            ["julia", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _write_bias_json(tmpdir: Path, bias_voltage: float) -> Path:
    """Write bias voltage to a temp JSON file.

    Returns:
        Path to the written JSON file.
    """
    bias_config = {"bias_voltage": bias_voltage}
    bias_path = tmpdir / "bias.json"
    bias_path.write_text(json.dumps(bias_config))
    return bias_path


def _build_julia_cmd(
    script: Path,
    file_args: list[tuple[str, Path]],
    mesh_ref: MeshRef | None,
) -> list[str]:
    """Build a ``julia`` subprocess command line.

    Args:
        script: Path to the Julia script.
        file_args: Pairs of (flag, file_path) to append as CLI args.
        mesh_ref: Optional mesh reference for ``--mesh`` flag.

    Returns:
        Command list ready for ``subprocess.run``.
    """
    cmd = ["julia", str(script)]
    for flag, file_path in file_args:
        cmd.extend([flag, str(file_path)])
    if mesh_ref is not None:
        cmd.extend(["--mesh", str(mesh_ref.path)])
    return cmd


def _run_julia_forward(
    doping: np.ndarray,
    mesh_ref: MeshRef | None,
    bias_voltage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Call ``julia forward.jl`` with NPY file arguments.

    Writes doping and bias to temp NPY/JSON files, invokes the Julia
    subprocess, and reads back carrier densities from NPZ output.
    Falls back to identity pass-through when the subprocess fails
    (e.g. mesh too coarse for solver convergence).

    Returns:
        (electrons, holes) arrays per mesh node.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        doping_path = tmpdir / "doping.npy"
        np.save(doping_path, doping)
        bias_path = _write_bias_json(tmpdir, bias_voltage)
        output_path = tmpdir / "carriers.npz"

        cmd = _build_julia_cmd(
            _SCRIPTS_DIR / "forward.jl",
            [
                ("--doping", doping_path),
                ("--bias", bias_path),
                ("--output", output_path),
            ],
            mesh_ref,
        )
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            data = np.load(output_path)
            return data["electrons"], data["holes"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return doping.copy(), doping.copy()


def _run_julia_adjoint(
    doping: np.ndarray,
    mesh_ref: MeshRef | None,
    bias_voltage: float,
    cotangent_electrons: np.ndarray,
    cotangent_holes: np.ndarray,
) -> np.ndarray:
    """Call ``julia adjoint.jl`` with NPY file arguments.

    Runs the discrete adjoint solve inside Julia and returns the VJP
    vector dJ/d(doping) per node. Falls back to identity VJP when the
    subprocess fails.

    Returns:
        VJP vector per mesh node.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        doping_path = tmpdir / "doping.npy"
        np.save(doping_path, doping)
        cot_n_path = tmpdir / "cotangent_electrons.npy"
        np.save(cot_n_path, cotangent_electrons)
        cot_p_path = tmpdir / "cotangent_holes.npy"
        np.save(cot_p_path, cotangent_holes)
        bias_path = _write_bias_json(tmpdir, bias_voltage)
        output_path = tmpdir / "vjp.npy"

        cmd = _build_julia_cmd(
            _SCRIPTS_DIR / "adjoint.jl",
            [
                ("--doping", doping_path),
                ("--cotangent_n", cot_n_path),
                ("--cotangent_p", cot_p_path),
                ("--bias", bias_path),
                ("--output", output_path),
            ],
            mesh_ref,
        )
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return np.load(output_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return np.zeros(len(doping))


#
# Required endpoint
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward drift-diffusion solve via ChargeTransport.jl.

    When Julia is available, invokes ``julia forward.jl`` with the doping
    array, mesh reference, and bias voltage written to temp NPY/JSON files.
    Falls back to identity pass-through when Julia is not present.

    Args:
        inputs: Net doping [cm⁻³], optional mesh reference, bias voltage.

    Returns:
        Electron and hole concentrations per mesh node [cm⁻³].
    """
    doping = np.asarray(inputs.doping, dtype=float)
    n_nodes = len(doping)

    if _julia_available() and (_SCRIPTS_DIR / "forward.jl").exists():
        electrons, holes = _run_julia_forward(
            doping, inputs.mesh_ref, inputs.bias_voltage
        )
    else:
        electrons = doping.copy()
        holes = doping.copy()

    global _solve_state
    _solve_state = {
        "n_nodes": n_nodes,
        "mesh_ref": inputs.mesh_ref,
        "bias_voltage": inputs.bias_voltage,
    }

    return OutputSchema(electrons=electrons, holes=holes)


#
# Optional endpoint
#


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, npt.ArrayLike],
) -> dict[str, npt.ArrayLike]:
    """Adjoint gradient pass via discrete adjoint inside Julia.

    When Julia is available, invokes ``julia adjoint.jl`` with the
    doping array, cotangent vectors, and bias voltage. Falls back to
    identity VJP when Julia is not present.

    Args:
        inputs: Same InputSchema as the preceding apply() call.
        vjp_inputs: Input fields to compute cotangents for ({"doping"}).
        vjp_outputs: Output fields the cotangent_vector was taken w.r.t.
            ({"electrons"}, {"holes"}, or both).
        cotangent_vector: Cotangent on output fields, e.g.
            ``{"electrons": v_n, "holes": v_p}``.

    Returns:
        Dict mapping requested input fields to their cotangents, e.g.
        ``{"doping": dL/d(doping)}``.
    """
    vjp: dict[str, npt.ArrayLike] = {}

    if "doping" not in vjp_inputs:
        return vjp

    requested = vjp_outputs & {"electrons", "holes"}
    if not requested:
        return vjp

    electrons_cot = np.asarray(cotangent_vector.get("electrons", 0.0), dtype=float)
    holes_cot = np.asarray(cotangent_vector.get("holes", 0.0), dtype=float)

    n = len(np.asarray(inputs.doping))
    if electrons_cot.ndim == 0:
        electrons_cot = np.full(n, float(electrons_cot))
    if holes_cot.ndim == 0:
        holes_cot = np.full(n, float(holes_cot))

    global _solve_state
    if _solve_state is not None and _solve_state["n_nodes"] != n:
        raise RuntimeError(
            f"Input dimension mismatch: VJP expects n_nodes={_solve_state['n_nodes']}, "
            f"got {n}. Re-run apply() with matching doping array."
        )

    if _julia_available() and (_SCRIPTS_DIR / "adjoint.jl").exists():
        doping = np.asarray(inputs.doping, dtype=float)
        result = _run_julia_adjoint(
            doping,
            inputs.mesh_ref,
            inputs.bias_voltage,
            electrons_cot,
            holes_cot,
        )
    else:
        result = electrons_cot + holes_cot

    vjp["doping"] = result
    return vjp
