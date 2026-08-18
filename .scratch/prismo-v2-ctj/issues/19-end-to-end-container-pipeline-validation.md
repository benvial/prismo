# 19 — End-to-end container pipeline validation

**What to build:** `make run-containers` runs the full optimization loop with both Docker containers active. Per-iteration logs show Δneff > 0 and ‖∇f‖ > 0. The convergence plot and doping field plot reflect physically meaningful results — a PN junction phase shifter whose effective index shifts measurably under reverse bias.

**Blocked by:** 17, 18

**Status:** superseded by 23

- [ ] `make run-containers` starts both containers, runs at least 5 MMA iterations, and tears down containers
- [ ] Per-iteration log shows `Δneff > 0` (optimizer maximizes Δneff, starts from zero at uniform ρ=0.25, improves over iterations)
- [ ] Per-iteration log shows `‖∇f‖ > 0` (gradient is non-zero — real sensitivity to design variables)
- [ ] `outputs/convergence.pdf` shows increasing Δneff over iterations (monotonic or near-monotonic improvement)
- [ ] `outputs/doping_field.pdf` shows a non-uniform ρ field (optimizer moved away from uniform ρ=0.25)
- [ ] `outputs/gradient_validation.pdf` is generated without errors
- [ ] No container identity-stub fallbacks are triggered during the run (verified via container HTTP logs)

## Comments

Validation attempted 2026-08-17 with built `prismo_chargetransport:latest` and
`prismo_gyptis:latest` images:

- `make run-containers RUN_ARGS='--no-jit --max-iter 5'` started and tore down
  both containers, completed five MMA evaluations, then raised
  `RuntimeError: Container pipeline produced invalid optimization signal at 5
  iteration(s)` because each reported `Δneff=0` and `||∇f||=0`.
- gyptis is responsive: perturbing one domain epsilon from `12.08` to `12.09`
  changes `neff_sq` from `1.0540543328573753` to `1.0544884462283735`.
- ChargeTransport is responsive for a graded `1e14..1e21 cm^-3` doping field,
  producing mean Soref-Bennett `Δε=6.37e-3`. The required uniform initial
  `ρ=0.25` maps to uniform, strictly positive doping and produces zero pipeline
  `Δneff` and gradient in this run.
- The failed run raises before output generation, so `convergence.pdf`,
  `doping_field.pdf`, and `gradient_validation.pdf` remain unverified for
  container mode. Container HTTP logs were not captured, so the no-stub
  fallback acceptance criterion is also unverified.

Although ticket 18 remains claimed, gyptis responds to its multi-domain input.
The immediate blocker is the pipeline's positive-only uniform-doping mapping:
it cannot form the PN-junction baseline required for ticket 19 while retaining
the ticket 15 uniform-`ρ` initial condition.

2026-08-17 follow-up: a fixed p/n polarity split from the mesh x-midline was
tested while preserving uniform `ρ=0.25` as the magnitude field, then reverted.
`make run-containers RUN_ARGS='--no-jit --max-iter 5 --output-dir
/tmp/prismo-ticket-19-outputs'` completed five evaluations and stopped both
containers, but did not meet the signal criterion: iteration 1 returned
`Δneff=-1.192093e-07`, `‖∇f‖=9.5679e+17`; iterations 2–5 returned zero for
both values. This is consistent with ChargeTransport's documented mixed-sign
PN solve fallback after MMA moves the initial profile. Ticket needs a solver
that accepts evolved mixed-sign profiles without identity fallback before it
can generate the required plots.

2026-08-18: tickets 21/22 delivered that mixed-sign PN forward/adjoint
solver, and ticket 23 re-ran this exact scenario successfully — five
container evaluations, Δneff increasing monotonically, nonzero gradients
throughout, all four output plots generated. Superseding this ticket in
favor of 23 rather than duplicating its acceptance criteria here.
