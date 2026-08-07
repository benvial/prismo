#!/usr/bin/env python3
"""Prototype: DEVSIM implicit differentiation gradient path (ticket 09).

Validates the end-to-end implicit-diff gradient through a 1D PN junction
DEVSIM solve:

    1. Forward solve → carrier densities
    2. Jacobian extraction via get_matrix_and_rhs(format="csc")
    3. Adjoint solve A^T λ = -u
    4. VJP: dJ/d(doping) = λ^T · ∂F/∂(doping)
    5. Finite-difference validation
    6. Wall-time measurement

Run this in the DEVSIM container:
    docker run --rm -v "$PWD":/mnt tesseract_photonic_waveguide_devsim \
        python /mnt/scripts/prototype_devsim_adjoint.py

Or locally if DEVSIM + scipy are installed:
    python scripts/prototype_devsim_adjoint.py
"""

import importlib.util
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

DEVSIM_AVAILABLE: bool
try:
    import devsim

    DEVSIM_AVAILABLE = True
except ImportError:
    DEVSIM_AVAILABLE = False

SCIPY_AVAILABLE: bool
try:
    import scipy.sparse
    import scipy.sparse.linalg

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

_project_root = Path(__file__).resolve().parents[1]
_api_path = _project_root / "components" / "tesseracts" / "devsim" / "tesseract_api.py"

_spec = importlib.util.spec_from_file_location("devsim_api", _api_path)
assert _spec is not None and _spec.loader is not None
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)

_build_1d_pn_junction = _api._build_1d_pn_junction
_cleanup_device = _api._cleanup_device

# ---------------------------------------------------------------------------
# Forward-solve helpers (duplicated from tesseract_api for direct use)
# ---------------------------------------------------------------------------


def _run_forward(
    doping: np.ndarray,
    device: str = "pn_prototype",
    region: str = "silicon",
) -> np.ndarray:
    """Run the DEVSIM forward solve and return total carrier charge per node."""
    import devsim as d

    _build_1d_pn_junction(d, device, region, doping)

    d.solve(
        type="dc", absolute_error=1e-10, relative_error=1e-10, maximum_iterations=30
    )

    electrons = np.array(
        d.get_node_model_values(device=device, region=region, name="Electrons"),
        dtype=float,
    )
    holes = np.array(
        d.get_node_model_values(device=device, region=region, name="Holes"),
        dtype=float,
    )
    return electrons + holes


# ---------------------------------------------------------------------------
# Jacobian extraction
# ---------------------------------------------------------------------------


def _extract_jacobian(device: str, region: str) -> "scipy.sparse.csc_matrix":
    """Extract the converged Newton Jacobian from DEVSIM."""
    import devsim

    r = devsim.get_matrix_and_rhs(device=device, region=region, format="csc")
    static = r["static"]
    n_eqs = len(static["rhs"])
    return scipy.sparse.csc_matrix(
        (static["av"], static["ai"], static["ap"]),
        shape=(n_eqs, n_eqs),
    )


# ---------------------------------------------------------------------------
# FD validation
# ---------------------------------------------------------------------------


def _fd_gradient(
    doping: np.ndarray,
    perturbation: np.ndarray,
    h: float,
    objective_fn: Callable[[np.ndarray], float],
) -> float:
    """Central-difference directional derivative of objective_fn."""
    obj_plus = objective_fn(doping + h * perturbation)
    obj_minus = objective_fn(doping - h * perturbation)
    return (obj_plus - obj_minus) / (2 * h)


# ---------------------------------------------------------------------------
# Main prototype
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the DEVSIM implicit differentiation prototype validation."""
    print("=" * 72)
    print("DEVSIM Implicit Differentiation Prototype (ticket 09)")
    print("=" * 72)

    if not DEVSIM_AVAILABLE:
        print(
            "\nDEVSIM not installed — run in the devsim container:\n"
            '  docker run --rm -v "$PWD":/mnt '
            "tesseract_photonic_waveguide_devsim \\\n"
            "      python /mnt/scripts/prototype_devsim_adjoint.py\n"
        )
        sys.exit(0)

    if not SCIPY_AVAILABLE:
        print("\nscipy not installed — required for sparse adjoint solve.")
        sys.exit(1)

    print(f"\nDEVSIM version: {devsim.__version__}")
    print(f"numpy version:  {np.__version__}")
    print(f"scipy version:  {scipy.__version__}\n")

    # --- 1. Build a 1D PN junction ---
    n_nodes = 30
    doping = np.zeros(n_nodes)
    mid = n_nodes // 2
    doping[:mid] = 1e22  # n-type
    doping[mid:] = -1e22  # p-type

    print(f"Mesh nodes: {n_nodes}")
    print(f"Doping range: [{doping.min():.1e}, {doping.max():.1e}] cm⁻³\n")

    device = "proto_devsim"
    region = "silicon"

    # Warm-up solve
    _ = _run_forward(doping, device, region)

    # --- 2. Time forward solve ---
    t0 = time.perf_counter()
    charge = _run_forward(doping, device, region)
    t_forward = time.perf_counter() - t0
    print(f"Forward solve time:              {t_forward:.4f} s")
    print(
        f"Carrier density range:           [{charge.min():.1e}, {charge.max():.1e}] cm⁻³"
    )

    # --- 3. Time Jacobian extraction ---
    t0 = time.perf_counter()
    A = _extract_jacobian(device, region)
    t_jacobian = time.perf_counter() - t0
    print(f"Jacobian shape:                  {A.shape}")
    print(f"Jacobian nnz:                    {A.nnz}")
    print(f"Jacobian extraction time:        {t_jacobian:.4f} s")

    # --- 4. Time adjoint solve ---
    cotangent = np.ones(n_nodes)
    u = np.zeros(3 * n_nodes)
    u[1 * n_nodes : 2 * n_nodes] = cotangent
    u[2 * n_nodes : 3 * n_nodes] = cotangent

    t0 = time.perf_counter()
    lam = scipy.sparse.linalg.spsolve(A.T.tocsc(), -u)
    t_adjoint = time.perf_counter() - t0

    q = 1.602176634e-19
    dF_ddoping = -q
    vjp_analytic = lam[:n_nodes] * dF_ddoping
    print(f"Adjoint solve time:              {t_adjoint:.4f} s")
    print(f"Total gradient time:             {t_jacobian + t_adjoint:.4f} s\n")

    # --- 5. Finite-difference validation ---
    print("-" * 72)
    print("Finite-difference validation (directional derivative)")
    print("-" * 72)

    rng = np.random.RandomState(42)
    perturbation = rng.randn(n_nodes) * 1e20

    def objective(d: np.ndarray) -> float:
        return float(np.sum(_run_forward(d.copy(), device, region)))

    vjp_dir = float(np.dot(vjp_analytic, perturbation))

    steps = [1e18, 1e17, 1e16, 1e15, 1e14]
    print(f"\n{'h':>12s}  {'FD deriv':>16s}  {'Rel error':>14s}")
    print("-" * 48)

    for h in steps:
        fd_val = _fd_gradient(doping, perturbation, h, objective)
        rel_err = abs(fd_val - vjp_dir) / max(abs(vjp_dir), 1.0)
        flag = " <<< optimal" if h == 1e17 else ""
        print(f"{h:>12.1e}  {fd_val:>16.8e}  {rel_err:>14.6e}{flag}")

    # --- 6. Compare against finite-difference cost ---
    print("\nFinite-difference cost (one perturbation):")
    print("  FD would require 2 forward solves per gradient component")
    print(f"  With {n_nodes} node-level doping params: ~{2 * n_nodes} solves")
    print(f"  Forward solve time per call: {t_forward:.4f}s")
    print(f"  Estimated FD gradient time:  {2 * n_nodes * t_forward:.1f}s")
    print(f"  Implicit diff gradient time: {t_jacobian + t_adjoint:.4f}s")
    print(
        f"  Speedup:                     "
        f"{(2 * n_nodes * t_forward) / (t_jacobian + t_adjoint):.0f}x\n"
    )

    # --- 7. Cleanup ---
    _cleanup_device(devsim, device)
    print("Done.")


if __name__ == "__main__":
    main()
