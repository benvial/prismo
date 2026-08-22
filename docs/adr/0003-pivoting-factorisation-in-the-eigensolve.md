# 0003 — A pivoting direct solver behind the gyptis shift-invert eigensolve

Status: Accepted (2026-08-22)

## Context

The optimizer stalled around the ticket-22 optimum with every trial in a
shrinking move-limit box reading ~2e-3 relative below the iterate, and the gap
did not shrink with the box. A line scan of the objective (`prismo
probe-objective`, ticket 23) showed white, neighbour-uncorrelated scatter of
±1e-6 on Δneff ≈ 3.3e-4 while the adjoint said the true change across the scan
was 3e-9; ChargeTransport's carriers and the design permittivity were exactly
linear along the same line, and the gyptis `neff_sq` alone jumped by ±3e-6 for
permittivity changes of 5e-11 relative.

Inside the gyptis Tesseract the shift-invert transform `(A − σB)⁻¹B` was
factorised by PETSc's native sparse LU. That factorisation has no threshold
pivoting, and the assembled system is an indefinite saddle point (A has zero
diagonals, ‖A‖ ≈ 4.6e6 against ‖B‖ ≈ 9e2). Its solves were accurate to ~1e-5:
Krylov-Schur converged in the transformed operator at any tolerance from 1e-7
to 1e-14 to the *same* eigenvalue, whose true residual was 1e-5 and which sat
~4e-7 relative off the eigenvalue an independent ARPACK solve (SuperLU, tol
1e-13, residual 1e-13) returned — with a roundoff pattern that moved with the
input. The background solve's eigenvalue carried a 6.7e-6 relative bias, so the
headline Δneff = neff(−5 V) − neff(0) was 9 % low on top of being noisy.

## Decision

The shift-invert preconditioner factorises with a pivoting direct solver —
UMFPACK (`_SHIFT_INVERT_FACTOR_SOLVER` in the gyptis `tesseract_api.py`) — and
Krylov-Schur runs at a tolerance of 1e-10 (`_EIGENSOLVER_TOLERANCE`). UMFPACK,
MUMPS and SuperLU all agree with ARPACK to 1e-13 and leave the eigenvalue
smooth along a design line to 1e-13; UMFPACK was the fastest of the three in
this image (8 s for three solves against 15 s for the native LU). With the
transform accurate, the tolerance means what it says: 1e-10 puts the true
residual at 1e-13 for one extra Krylov iteration.

## Consequences

- The objective's noise floor drops from 2e-3 to 2e-11 relative, and a fitted
  slope along the gradient direction matches the adjoint to seven digits; the
  move-limit stall rule now triggers on converged geometry, not on noise.
- Δneff at a given design moves by up to ~1e-5 absolute against values
  computed before this change (the ticket-22 design reads +3.612e-4 instead of
  +3.301e-4). Numbers from earlier runs are solver artefacts at that level and
  are not comparable; the gyptis regression case's expected `neff_sq` was
  regenerated.
- **Trade-off — a dependency on a PETSc external package.** The image must
  ship UMFPACK (the FEniCS base does). A build without it fails the eigensolve
  outright rather than silently falling back to the inaccurate native LU; the
  constant is the one place to point at MUMPS or SuperLU instead.
- `prismo probe-objective` stays as the smoothness gauge for any future solver
  or parameterization change.
