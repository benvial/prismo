# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for prismo_chargetransport
# ChargeTransport.jl semiconductor drift-diffusion component (Julia worker).
#
# Persistent worker pattern: Python owns one Julia process for the Tesseract
# lifecycle and exchanges NPY/NPZ file references over JSON lines.
# A missing Julia backend is a hard error -- there is no physics-free identity
# fallback that would fabricate carrier densities (rethink ticket 04).
#
# Ref: tickets 02 (container + I/O), 03 (forward API), 04 (adjoint),
#      06 (container build).

import atexit
import hashlib
import json
import os
import select
import subprocess
import tempfile
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Literal

import numpy as np
import numpy.typing as npt
from prismo_shared.schemas import MeshRef
from prismo_shared.session import SolveSessionRegistry, array_identity
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64

_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

# Backstop wall-clock limit for one Julia worker request (ticket 18). The solve
# itself is bounded *inside* Julia: ``ct_common.jl`` checks a wall-clock budget
# (``PRISMO_CT_SOLVE_BUDGET_S``, default 120 s) in every continuation loop and
# returns ``ok: false`` with a descriptive error when it runs out, so a hard
# design costs the optimizer one evaluation and the worker -- with every warm
# solution it holds -- survives. This Python timeout therefore only fires when
# the process is truly hung (no response at all), and when it fires the worker
# is killed and restarted cold. Keep it well above the Julia budget; override
# via env for slower hosts. (Used to be 25 s, which killed the worker on every
# hard-but-solvable design, 12-70 s on the shared mesh.)
_JULIA_TIMEOUT_S = float(os.environ.get("PRISMO_CT_JULIA_TIMEOUT_S", "600"))

# Precompile-warmed PackageCompiler sysimage baked into the Julia base image
# (ticket 17). Without it the first solve pays ~22s of JIT compilation.
# Absent locally (plain ``julia script.jl`` still works, just slower).
_JULIA_SYSIMAGE = Path(
    os.environ.get("PRISMO_CT_JULIA_SYSIMAGE", "/opt/julia_sysimage/prismo_ct.so")
)

#
# Schemas
#


class InputSchema(BaseModel):
    """Inputs to the ChargeTransport.jl drift-diffusion solve.

    Attributes:
        operation: ``"solve"`` runs the drift-diffusion solve. ``"reset"`` drops
            the persistent worker's warm solutions (equilibrium and biased
            Newton starting points) so the next solve is cold, i.e. a function
            of the doping alone rather than of the solve history -- the cold
            re-evaluation of ticket 20. ``reset`` needs no ``doping``.
        doping: Net doping concentration at every mesh node [cm⁻³].
            Positive = n-type, negative = p-type. Required for ``"solve"``.
        mesh_ref: Reference to the shared Gmsh 2D mesh file.
        bias_voltage: Applied bias voltage in volts. Default 0 (equilibrium).
    """

    operation: Literal["solve", "reset"] = "solve"
    doping: Differentiable[Array[(None,), Float64]] | None = None
    mesh_ref: MeshRef | None = None
    bias_voltage: float = Field(default=0.0, ge=-50.0, le=50.0)


class OutputSchema(BaseModel):
    """Outputs of the ChargeTransport.jl drift-diffusion solve.

    Attributes:
        electrons: Electron concentration per mesh node [cm⁻³]. Empty for the
            ``reset`` operation.
        holes: Hole concentration per mesh node [cm⁻³]. Empty for ``reset``.
    """

    electrons: Differentiable[Array[(None,), Float64]]
    holes: Differentiable[Array[(None,), Float64]]


#
# Module-level state
#

# Forward solves live here behind the fixed apply/vjp endpoints; the adjoint
# retrieves the session whose inputs match. Scoped by doping profile + worker
# generation so the two bias solves of one autodiff pass coexist while solves
# from earlier passes (or a restarted worker) are evicted.
_session_registry = SolveSessionRegistry()
_julia_worker: "_JuliaWorker | None" = None
_worker_lock = RLock()
_worker_generation = 0


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


def _forward_state_key(
    doping: np.ndarray,
    mesh_ref: MeshRef | None,
    bias_voltage: float,
    worker_generation: int,
) -> tuple[tuple[object, ...], float, int]:
    """Return an exact identity for a forward solve available to its VJP."""
    mesh_identity = None if mesh_ref is None else mesh_ref.model_dump_json()
    profile = (*array_identity(doping), mesh_identity)
    return profile, float(bias_voltage), worker_generation


def _worker_profile_key(doping: np.ndarray, mesh_ref: MeshRef | None) -> str:
    """Return a compact exact profile identity shared with the Julia worker."""
    mesh_identity = _worker_mesh_identity(mesh_ref)
    digest = hashlib.sha256()
    digest.update(repr((doping.shape, doping.dtype.str)).encode())
    digest.update(np.ascontiguousarray(doping).tobytes())
    digest.update(mesh_identity)
    return digest.hexdigest()


def _worker_mesh_identity(mesh_ref: MeshRef | None) -> bytes:
    """Fingerprint mesh metadata and file content independently of doping."""
    if mesh_ref is None:
        return b"fallback-1d-mesh"

    digest = hashlib.sha256()
    digest.update(mesh_ref.model_dump_json().encode())
    mesh_path = Path(mesh_ref.path)
    if mesh_path.is_file():
        digest.update(mesh_path.read_bytes())
    else:
        digest.update(b"missing-mesh-file")
    return digest.digest()


def _worker_mesh_key(mesh_ref: MeshRef | None) -> str:
    """Return the worker configuration key for a mesh reference."""
    return hashlib.sha256(_worker_mesh_identity(mesh_ref)).hexdigest()


class _JuliaWorker:
    """One Julia process that owns reusable ChargeTransport solver state."""

    def __init__(self, process: subprocess.Popen[str], generation: int) -> None:
        self._process = process
        self.generation = generation
        self._lock = RLock()

    def is_alive(self) -> bool:
        """Whether the Julia process is still running."""
        return self._process.poll() is None

    def request(self, request: dict[str, object]) -> dict[str, object]:
        """Send one request and wait for its JSON response within the budget."""
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("Julia worker exited before handling the request")

            stdin = self._process.stdin
            stdout = self._process.stdout
            if stdin is None or stdout is None:
                raise RuntimeError("Julia worker does not expose a usable pipe")

            try:
                stdin.write(json.dumps(request) + "\n")
                stdin.flush()
            except OSError as exc:
                self._terminate_locked()
                raise RuntimeError("Could not send request to Julia worker") from exc

            deadline = monotonic() + _JULIA_TIMEOUT_S
            diagnostics: list[str] = []
            while True:
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    self._terminate_locked()
                    raise TimeoutError(
                        f"Julia worker exceeded {_JULIA_TIMEOUT_S:g}s request timeout"
                    )
                ready, _, _ = select.select([stdout], [], [], remaining)
                if not ready:
                    continue
                response_line = stdout.readline()
                if not response_line:
                    self._terminate_locked()
                    detail = "; ".join(diagnostics)
                    raise RuntimeError(
                        "Julia worker closed its response pipe"
                        + (f": {detail}" if detail else "")
                    )
                try:
                    response = json.loads(response_line)
                except json.JSONDecodeError:
                    diagnostics.append(response_line.strip())
                    continue
                if isinstance(response, dict) and "ok" in response:
                    return response
                diagnostics.append(response_line.strip())

    def close(self) -> None:
        """Stop the process, falling back to termination if it is unresponsive."""
        with self._lock:
            self._terminate_locked(graceful=True)

    def _terminate_locked(self, *, graceful: bool = False) -> None:
        process = self._process
        if process.poll() is not None:
            return
        if graceful and process.stdin is not None:
            try:
                process.stdin.write('{"operation":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=2)
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _start_julia_worker(generation: int) -> _JuliaWorker:
    """Launch the lifecycle-owned Julia worker with the configured sysimage."""
    cmd = ["julia"]
    if _JULIA_SYSIMAGE.is_file():
        cmd.extend(["--sysimage", str(_JULIA_SYSIMAGE)])
    cmd.append(str(_SCRIPTS_DIR / "worker.jl"))
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise RuntimeError(f"Julia worker failed to start: {exc}") from exc
    return _JuliaWorker(process, generation)


def _get_julia_worker() -> _JuliaWorker:
    """Return a live worker, invalidating state when a process is replaced."""
    global _julia_worker, _worker_generation
    with _worker_lock:
        if _julia_worker is not None and _julia_worker.is_alive():
            return _julia_worker
        if _julia_worker is not None:
            _julia_worker.close()
        _worker_generation += 1
        _julia_worker = _start_julia_worker(_worker_generation)
        return _julia_worker


def shutdown() -> None:
    """End the Julia-worker lifecycle and invalidate the retained forward session."""
    global _julia_worker, _worker_generation
    with _worker_lock:
        worker = _julia_worker
        _julia_worker = None
        _worker_generation += 1
        _session_registry.clear()
    if worker is not None:
        worker.close()


atexit.register(shutdown)


def reset_worker() -> None:
    """Drop the Julia worker's warm solutions without restarting the process.

    The mesh, system and material data stay resident (they are a function of
    the request); only the Newton starting points that carry solve history go.
    The retained forward sessions go with them: a VJP must follow a forward on
    the reset worker. A worker that is not running has nothing warm to drop.
    """
    with _worker_lock:
        worker = _julia_worker
        _session_registry.clear()
        if worker is None or not worker.is_alive():
            return
        response = worker.request({"operation": "reset"})
    if response.get("ok") is not True:
        raise RuntimeError(
            "Julia worker reset failed: "
            f"{response.get('error', 'unknown worker error')}"
        )


def _run_julia_forward(
    doping: np.ndarray,
    mesh_ref: MeshRef | None,
    bias_voltage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a forward solve in the lifecycle-owned Julia worker."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        doping_path = tmpdir / "doping.npy"
        np.save(doping_path, doping)
        bias_path = _write_bias_json(tmpdir, bias_voltage)
        output_path = tmpdir / "carriers.npz"
        try:
            response = _get_julia_worker().request(
                {
                    "operation": "forward",
                    "doping_path": str(doping_path),
                    "bias_path": str(bias_path),
                    "output_path": str(output_path),
                    "mesh_path": "" if mesh_ref is None else str(mesh_ref.path),
                    "mesh_key": _worker_mesh_key(mesh_ref),
                    "profile_key": _worker_profile_key(doping, mesh_ref),
                }
            )
            if response.get("ok") is not True:
                raise RuntimeError(str(response.get("error", "unknown worker error")))
            with np.load(output_path) as data:
                electrons = np.asarray(data["electrons"], dtype=float)
                holes = np.asarray(data["holes"], dtype=float)
            if electrons.shape != doping.shape or holes.shape != doping.shape:
                raise RuntimeError(
                    "Julia forward solve returned carrier fields with shapes "
                    f"{electrons.shape} and {holes.shape}; expected {doping.shape}."
                )
            if not np.all(np.isfinite(electrons)) or not np.all(np.isfinite(holes)):
                raise RuntimeError(
                    "Julia forward solve returned non-finite carrier fields."
                )
            return electrons, holes
        except (KeyError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Julia forward solve failed: {exc}") from exc


def _run_julia_adjoint(
    doping: np.ndarray,
    mesh_ref: MeshRef | None,
    bias_voltage: float,
    cotangent_electrons: np.ndarray,
    cotangent_holes: np.ndarray,
) -> np.ndarray:
    """Run a matched-state discrete adjoint in the Julia worker."""
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
        try:
            response = _get_julia_worker().request(
                {
                    "operation": "vjp",
                    "doping_path": str(doping_path),
                    "cotangent_electrons_path": str(cot_n_path),
                    "cotangent_holes_path": str(cot_p_path),
                    "bias_path": str(bias_path),
                    "output_path": str(output_path),
                    "mesh_path": "" if mesh_ref is None else str(mesh_ref.path),
                    "mesh_key": _worker_mesh_key(mesh_ref),
                    "profile_key": _worker_profile_key(doping, mesh_ref),
                }
            )
            if response.get("ok") is not True:
                raise RuntimeError(str(response.get("error", "unknown worker error")))
            vjp = np.asarray(np.load(output_path), dtype=float)
            if vjp.shape != doping.shape:
                raise RuntimeError(
                    "Julia adjoint solve returned VJP shape "
                    f"{vjp.shape}; expected {doping.shape}."
                )
            if not np.all(np.isfinite(vjp)):
                raise RuntimeError(
                    "Julia adjoint solve returned non-finite VJP values."
                )
            return vjp
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Julia adjoint solve failed: {exc}") from exc


#
# Required endpoint
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward drift-diffusion solve via ChargeTransport.jl.

    Sends the doping array, mesh reference, and bias voltage to a persistent
    Julia worker that retains compatible solver state. A missing Julia backend
    is a hard error -- there is no physics-free identity fallback that would
    fabricate carrier densities -- and failures from an available Julia solver
    propagate to the caller.

    Args:
        inputs: Net doping [cm⁻³], optional mesh reference, bias voltage.

    Returns:
        Electron and hole concentrations per mesh node [cm⁻³].
    """
    if inputs.operation == "reset":
        reset_worker()
        empty = np.zeros(0, dtype=float)
        return OutputSchema(electrons=empty, holes=empty)

    if inputs.doping is None:
        raise ValueError("ChargeTransport solve requires a doping array")
    doping = np.asarray(inputs.doping, dtype=float)

    if not (_julia_available() and (_SCRIPTS_DIR / "worker.jl").exists()):
        raise RuntimeError(
            "ChargeTransport requires a Julia drift-diffusion backend "
            "(ChargeTransport.jl worker); none is available. There is no "
            "physics-free fallback -- run the component in its container."
        )
    electrons, holes = _run_julia_forward(doping, inputs.mesh_ref, inputs.bias_voltage)
    worker_generation = _get_julia_worker().generation

    identity = _forward_state_key(
        doping,
        inputs.mesh_ref,
        inputs.bias_voltage,
        worker_generation,
    )
    profile, _bias, generation = identity
    _session_registry.open(identity, scope=(profile, generation))

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

    Sends cotangents to the Julia worker that owns the matching forward state.
    A missing Julia backend is a hard error -- there is no physics-free identity
    VJP that would fabricate a gradient -- and failures from an available Julia
    solver propagate to the caller.

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

    if inputs.doping is None:
        raise ValueError("ChargeTransport VJP requires a doping array")
    n = len(np.asarray(inputs.doping))
    if electrons_cot.ndim == 0:
        electrons_cot = np.full(n, float(electrons_cot))
    if holes_cot.ndim == 0:
        holes_cot = np.full(n, float(holes_cot))

    doping = np.asarray(inputs.doping, dtype=float)
    if not (_julia_available() and (_SCRIPTS_DIR / "worker.jl").exists()):
        raise RuntimeError(
            "ChargeTransport VJP requires a Julia adjoint backend "
            "(ChargeTransport.jl worker); none is available. There is no "
            "physics-free fallback -- run the component in its container."
        )
    try:
        worker_generation = _get_julia_worker().generation
    except RuntimeError as exc:
        raise RuntimeError(f"Julia adjoint solve failed: {exc}") from exc
    identity = _forward_state_key(
        doping,
        inputs.mesh_ref,
        inputs.bias_voltage,
        worker_generation,
    )
    if _session_registry.match(identity) is None:
        raise RuntimeError(
            "ChargeTransport VJP requires a preceding apply() with identical "
            "doping, mesh reference, and bias voltage."
        )

    vjp["doping"] = _run_julia_adjoint(
        doping,
        inputs.mesh_ref,
        inputs.bias_voltage,
        electrons_cot,
        holes_cot,
    )
    return vjp
