#!/usr/bin/env python3
"""Prototype: gyptis field-epsilon eigen-adjoint gradient path.

Validates the end-to-end field-valued Hellmann-Feynman gradient through a 2D
waveguide eigenmode solve in gyptis, where the silicon core carries a
spatially-varying permittivity on an embedded, PML-inset design region:

    1. Forward eigenmode solve on the design field -> neff (guided mode)
    2. Structured vs uniform-same-mean field -> neff differs (no mean-collapse)
    3. Single-pass field-valued adjoint -> per-design-cell dneff_sq/deps
    4. Central finite-difference validation on sampled design cells
    5. Wall-time: one shared solve + single-pass assembly vs FD's 2N solves

Run this in the gyptis container:
    docker run --rm --entrypoint python -v "$PWD":/mnt prismo_gyptis:latest \
        /mnt/scripts/prototype_gyptis_eigen_adjoint.py

Or locally if gyptis/FEniCS are installed:
    python scripts/prototype_gyptis_eigen_adjoint.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

GYPTIS_AVAILABLE: bool
try:
    import dolfin
    import gyptis

    GYPTIS_AVAILABLE = True
except ImportError:
    GYPTIS_AVAILABLE = False

_project_root = Path(__file__).resolve().parents[1]
_api_path = _project_root / "components" / "tesseracts" / "gyptis" / "tesseract_api.py"

_spec = importlib.util.spec_from_file_location("gyptis_api", _api_path)
assert _spec is not None and _spec.loader is not None
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)


def _neff_sq(design: np.ndarray) -> float:
    """Forward solve returning neff_sq for a design-region field."""
    return float(_api.apply(_api.InputSchema(design_epsilon=design)).neff_sq)


def _field_gradient(design: np.ndarray) -> np.ndarray:
    """Field-valued adjoint: per-design-cell dneff_sq/deps (one shared solve)."""
    _api.apply(_api.InputSchema(design_epsilon=design))
    return np.asarray(
        _api.vector_jacobian_product(
            _api.InputSchema(design_epsilon=design),
            {"design_epsilon"},
            {"neff_sq"},
            {"neff_sq": 1.0},
        )["design_epsilon"]
    )


def main() -> None:
    """Run the gyptis field-epsilon eigen-adjoint prototype validation."""
    print("=" * 72)
    print("gyptis field-epsilon eigen-adjoint prototype")
    print("=" * 72)

    if not GYPTIS_AVAILABLE:
        print(
            "\ngyptis/FEniCS not installed — run in the gyptis container:\n"
            '  docker run --rm --entrypoint python -v "$PWD":/mnt '
            "prismo_gyptis:latest \\\n"
            "      /mnt/scripts/prototype_gyptis_eigen_adjoint.py\n"
        )
        sys.exit(0)

    print(
        f"\ngyptis {gyptis.__version__}  dolfin {dolfin.__version__}  "
        f"numpy {np.__version__}\n"
    )

    # --- 1. Embedded design region on the silicon core ---
    centroids = _api.design_cell_centroids()
    n_design = centroids.shape[0]
    core_bg = _api.DEFAULT_CORE_EPSILON
    print(f"design cells:            {n_design}")
    print(
        f"design x-range:          "
        f"[{centroids[:, 0].min():.3f}, {centroids[:, 0].max():.3f}]"
    )
    print(
        f"design y-range:          "
        f"[{centroids[:, 1].min():.3f}, {centroids[:, 1].max():.3f}]"
    )

    rng = np.random.default_rng(0)
    pattern = rng.uniform(-1.0, 1.0, n_design)
    pattern -= pattern.mean()  # zero-mean spatial structure
    structured = core_bg + 0.5 * pattern
    uniform = np.full(n_design, core_bg)  # same mean, no structure

    # --- 2. Forward: structured vs uniform-same-mean ---
    t0 = time.perf_counter()
    neff_sq_struct = _neff_sq(structured)
    t_forward = time.perf_counter() - t0
    neff_sq_unif = _neff_sq(uniform)
    n_core = _api.DEFAULT_CORE_EPSILON**0.5
    n_clad = _api.DEFAULT_CLAD_EPSILON**0.5
    print(f"\nneff (structured):       {neff_sq_struct**0.5:.6f}")
    print(f"neff (uniform, same mean): {neff_sq_unif**0.5:.6f}")
    print(f"|dneff_sq|:              {abs(neff_sq_struct - neff_sq_unif):.3e}")
    print(
        f"guided window:           ({n_clad:.3f}, {n_core:.3f})  "
        f"structured guided? {n_clad < neff_sq_struct**0.5 < n_core}"
    )
    print(f"forward solve time:      {t_forward:.2f}s")

    # --- 3. Single-pass field adjoint ---
    t0 = time.perf_counter()
    grad = _field_gradient(structured)
    t_adjoint = time.perf_counter() - t0
    print(
        f"\nfield gradient: shape={grad.shape} "
        f"std={grad.std():.3e} mean={grad.mean():.3e}"
    )
    print(f"adjoint time (shared solve + single pass): {t_adjoint:.2f}s")

    # --- 4. Central finite-difference validation on sampled cells ---
    print("\n" + "-" * 72)
    print("Finite-difference validation (sampled design cells)")
    print("-" * 72)
    print(f"{'cell':>6} {'analytic':>15} {'central-FD':>15} {'rel-err':>10}")
    h = 1e-4
    worst = 0.0
    for local in np.linspace(0, n_design - 1, 3, dtype=int):
        vp = structured.copy()
        vp[local] += h
        vm = structured.copy()
        vm[local] -= h
        fd = (_neff_sq(vp) - _neff_sq(vm)) / (2 * h)
        an = grad[local]
        rel = abs(fd - an) / max(abs(an), 1e-12)
        worst = max(worst, rel)
        print(f"{local:>6} {an:>15.6e} {fd:>15.6e} {rel:>10.2e}")
    print(f"\nworst sampled rel-err:   {worst:.2e}  (target < 1e-3)")

    # --- 5. Cost: one solve + single pass vs FD's 2N solves ---
    print(f"\nCost comparison (n_design = {n_design}):")
    print("  FD would require 2 * n_design eigen solves per gradient")
    print(f"  = {2 * n_design} solves  (~{2 * n_design * t_forward:.0f}s)")
    print(f"  field adjoint:           {t_adjoint:.2f}s")
    print(
        f"  speedup:                 "
        f"{(2 * n_design * t_forward) / max(t_adjoint, 1e-6):.0f}x"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
