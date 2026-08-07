#!/usr/bin/env python3
"""Prototype: gyptis eigen-adjoint gradient path (ticket 10).

Validates the end-to-end Hellmann-Feynman gradient through a 2D waveguide
eigenmode solve in gyptis:

    1. Forward eigenmode solve → neff (effective index)
    2. Re-assemble A and B matrices from formulation.weak
    3. Compute dA/dε_d and dB/dε_d per subdomain
    4. Hellmann-Feynman: dλ/dε_d = (x^H (dA/dε_d - λ dB/dε_d) x) / (x^H B x)
    5. Chain: dneff_sq/dε_d = 2·neff·dneff/dε_d
    6. Finite-difference validation per subdomain
    7. Wall-time measurement vs FD baseline

Run this in the gyptis container:
    docker run --rm -v "$PWD":/mnt tesseract_photonic_waveguide_gyptis \
        python /mnt/scripts/prototype_gyptis_eigen_adjoint.py

Or locally if gyptis/FEniCS are installed:
    python scripts/prototype_gyptis_eigen_adjoint.py
"""

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

GYPTIS_AVAILABLE: bool
try:
    import dolfin  # noqa: F401
    import gyptis  # noqa: F401

    GYPTIS_AVAILABLE = True
except ImportError:
    GYPTIS_AVAILABLE = False

_project_root = Path(__file__).resolve().parents[1]
_api_path = _project_root / "components" / "tesseracts" / "gyptis" / "tesseract_api.py"

_spec = importlib.util.spec_from_file_location("gyptis_api", _api_path)
assert _spec is not None and _spec.loader is not None
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)

_build_waveguide = _api._build_waveguide

# ---------------------------------------------------------------------------
# Forward-solve helpers (duplicated from tesseract_api for direct use)
# ---------------------------------------------------------------------------


def _run_eigenmode(epsilon: np.ndarray, wavelength: float = 1.55e-6) -> float:
    """Run gyptis eigenmode solve and return neff_sq."""
    simu, _geom, _n_domains, k0 = _build_waveguide(epsilon.copy(), wavelength)
    _ = simu.eigensolve(n_eig=4, target=k0)

    ev_re, ev_im, _rx, _cx = simu.eigensolver.get_eigenpair(0)
    lam = ev_re + 1j * ev_im
    kz = np.sqrt(lam)
    neff = float(kz / k0)
    return neff * neff


def _solve_and_extract_eigen(
    epsilon: np.ndarray, wavelength: float = 1.55e-6
) -> dict[str, Any]:
    """Solve eigenproblem and return simulation + eigen data for HF gradient.

    Returns a dict usable by _hf_assemble().
    """
    k0 = 2.0 * np.pi / wavelength
    simu, _geom, _n_domains, _k0 = _build_waveguide(epsilon.copy(), wavelength)
    _ = simu.eigensolve(n_eig=4, target=k0)

    ev_re, ev_im, rx, cx = simu.eigensolver.get_eigenpair(0)
    lam = ev_re + 1j * ev_im
    kz = np.sqrt(lam)
    neff = kz / k0

    return {
        "simu": simu,
        "k0": k0,
        "kz": kz,
        "neff": neff,
        "lam": lam,
        "rx": rx,
        "cx": cx,
        "n_domains": _n_domains,
    }


# ---------------------------------------------------------------------------
# Hellmann-Feynman gradient
# ---------------------------------------------------------------------------


def _hf_assemble(eigen_data: dict[str, Any]) -> np.ndarray:
    """Compute dneff_sq/dε per subdomain from a pre-solved eigenstate.

    Args:
        eigen_data: Dict from _solve_and_extract_eigen with simu, k0, kz,
            neff, lam, rx, cx, n_domains.

    Returns:
        Gradient of neff_sq w.r.t. per-subdomain permittivity.
    """
    import dolfin  # type: ignore[import-untyped]

    simu = eigen_data["simu"]
    k0 = eigen_data["k0"]
    kz = eigen_data["kz"]
    neff = eigen_data["neff"]
    lam = eigen_data["lam"]
    rx = eigen_data["rx"]
    cx = eigen_data["cx"]
    n_domains = eigen_data["n_domains"]

    wf = simu.formulation.weak
    dv = dolfin.PETScVector()
    V = simu.formulation.space
    trial = dolfin.TrialFunction(V)
    dv.init(dolfin.as_backend_type(trial.vector()).vec())

    bcs = simu.formulation.build_boundary_conditions()

    A_mat = dolfin.PETScMatrix()
    B_mat = dolfin.PETScMatrix()
    b = dolfin.PETScVector()
    dolfin.assemble_system(wf[0], dv, bcs, A_tensor=A_mat, b_tensor=b)
    dolfin.assemble_system(wf[1], dv, bcs, A_tensor=B_mat, b_tensor=b)
    Bmat_mat = B_mat.mat()

    u = simu.formulation.trial
    v = simu.formulation.test
    et = dolfin.as_vector([u[0], u[1], 0.0])
    ez = u[2]
    vt = dolfin.as_vector([v[0], v[1], 0.0])
    vz = v[2]
    zhat = dolfin.as_vector([0.0, 0.0, 1.0])

    dneff_sq_deps = np.zeros(n_domains)

    for d_idx in range(n_domains):
        domain_name = str(d_idx + 1)
        dx_d = simu.dx(domain_name)

        form_tt = -dolfin.Constant(k0 * k0) * dolfin.inner(et, vt) * dx_d
        F_tt = form_tt.real + form_tt.imag
        Mtt = dolfin.PETScMatrix()
        dolfin.assemble(F_tt, tensor=Mtt)
        for bc in bcs:
            bc.apply(Mtt)
        Mtt_mat = Mtt.mat()

        form_zz = dolfin.Constant(k0 * k0) * dolfin.inner(ez * zhat, vz * zhat) * dx_d
        F_zz = form_zz.real + form_zz.imag
        Mzz = dolfin.PETScMatrix()
        dolfin.assemble(F_zz, tensor=Mzz)
        for bc in bcs:
            bc.apply(Mzz)
        Mzz_mat = Mzz.mat()

        num = rx.dot(Mtt_mat * rx) + cx.dot(Mtt_mat * cx)
        num -= lam * (rx.dot(Mzz_mat * rx) + cx.dot(Mzz_mat * cx))
        den = rx.dot(Bmat_mat * rx) + cx.dot(Bmat_mat * cx)
        dlam_deps = num / den
        dneff_deps = dlam_deps / (2.0 * k0 * kz)
        dneff_sq_deps[d_idx] = 2.0 * neff * dneff_deps

    return dneff_sq_deps


def _hf_gradient(epsilon: np.ndarray, wavelength: float = 1.55e-6) -> np.ndarray:
    """Convenience wrapper: solve + HF gradient in one call."""
    eigen_data = _solve_and_extract_eigen(epsilon, wavelength)
    return _hf_assemble(eigen_data)


# ---------------------------------------------------------------------------
# Main prototype
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the gyptis eigen-adjoint (Hellmann-Feynman) prototype validation."""
    print("=" * 72)
    print("gyptis Eigen-Adjoint Prototype (ticket 10)")
    print("=" * 72)

    if not GYPTIS_AVAILABLE:
        print(
            "\ngyptis/FEniCS not installed — run in the gyptis container:\n"
            '  docker run --rm -v "$PWD":/mnt '
            "tesseract_photonic_waveguide_gyptis \\\n"
            "      python /mnt/scripts/prototype_gyptis_eigen_adjoint.py\n"
        )
        sys.exit(0)

    import dolfin  # type: ignore[import-untyped]
    import gyptis  # type: ignore[import-untyped]

    print(f"\ngyptis version: {gyptis.__version__}")
    print(f"dolfin version: {dolfin.__version__}")
    print(f"numpy version:  {np.__version__}\n")

    # --- 1. Build waveguide with subdomains ---
    n_domains = 3
    epsilon = np.array([12.0, 2.0, 1.0], dtype=float)

    print(f"Subdomains:       {n_domains}")
    print(f"Permittivities:   {epsilon.tolist()}")

    # --- 2. Time forward solve ---
    t0 = time.perf_counter()
    eigen_data = _solve_and_extract_eigen(epsilon)
    t_forward = time.perf_counter() - t0
    neff = eigen_data["neff"]
    print(f"neff (fundamental):              {neff:.6f}")
    print(f"Forward solve time:              {t_forward:.4f} s")

    # --- 3. Time HF assembly only (no re-solve) ---
    t0 = time.perf_counter()
    dneff_sq_hf = _hf_assemble(eigen_data)
    t_hf_assembly = time.perf_counter() - t0
    print(f"HF assembly time (no re-solve):  {t_hf_assembly:.4f} s\n")

    # --- 4. Finite-difference validation per subdomain ---
    print("-" * 72)
    print("Finite-difference validation (per subdomain)")
    print("-" * 72)

    fd_steps = [1e-3, 1e-4, 1e-5]
    print(f"\n{'Domain':>8s}  {'eps':>6s}  ", end="")
    for h in fd_steps:
        print(f"{'FD(h=' + str(h) + ')':>18s}  ", end="")
    print(f"{'HF analytic':>18s}  {'Rel err':>12s}")
    print("-" * (30 + 24 * len(fd_steps) + 30))

    for d_idx in range(n_domains):
        eps_ref = epsilon[d_idx]
        hf_val = dneff_sq_hf[d_idx]

        perturbation = epsilon.copy()
        row_parts = [f"{d_idx + 1:>8d}  {eps_ref:>6.3f}  "]

        for h in fd_steps:
            perturbation[d_idx] = eps_ref + h
            f_plus = _run_eigenmode(perturbation)
            perturbation[d_idx] = eps_ref - h
            f_minus = _run_eigenmode(perturbation)
            perturbation[d_idx] = eps_ref  # restore
            fd_val = (f_plus - f_minus) / (2 * h)
            row_parts.append(f"{fd_val:>18.8f}  ")

        rel_err = abs(fd_val - hf_val) / max(abs(hf_val), 1e-10) if hf_val != 0 else 0
        row_parts.append(f"{hf_val:>18.8f}  {rel_err:>12.2e}")
        print("".join(row_parts))

    # --- 5. Timing comparison ---
    print(f"\nCost comparison (n_domains = {n_domains}):")
    print("  FD would require 2 * n_domains eigen solves per gradient")
    print(f"  = {2 * n_domains} extra eigen solves")
    print(f"  Eigen solve time:     {t_forward:.4f}s")
    print(f"  Estimated FD time:    {2 * n_domains * t_forward:.4f}s")
    print(f"  HF assembly time:     {t_hf_assembly:.4f}s")
    print(
        f"  Speedup:              "
        f"{(2 * n_domains * t_forward) / max(t_hf_assembly, 1e-6):.0f}x"
    )

    # --- 6. Additional check: gradient consistency with Soref-Bennett-like perturbation ---
    print(f"\n{'=' * 72}")
    print("Perturbation test (simulated permittivity change)")
    print("=" * 72)

    pert = np.array([0.1, 0.0, 0.0], dtype=float)

    def neff_sq_fn(eps: np.ndarray) -> float:
        return _run_eigenmode(eps)

    h_test = 1e-3
    neff_sq_plus = neff_sq_fn(epsilon + h_test * pert)
    neff_sq_minus = neff_sq_fn(epsilon - h_test * pert)
    fd_dir = (neff_sq_plus - neff_sq_minus) / (2 * h_test)

    hf_dir = float(np.dot(dneff_sq_hf, pert))

    print(f"\nEpsilon perturbation:             {pert.tolist()}")
    print(f"FD directional derivative:        {fd_dir:.8f}")
    print(f"HF directional derivative:        {hf_dir:.8f}")
    print(
        f"Relative error:                   "
        f"{abs(fd_dir - hf_dir) / max(abs(fd_dir), 1e-10):.2e}\n"
    )

    print("Done.")


if __name__ == "__main__":
    main()
