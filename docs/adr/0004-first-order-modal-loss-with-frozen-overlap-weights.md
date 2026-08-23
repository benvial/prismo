# 0004 — Modal free-carrier loss as a first-order perturbation with frozen overlap weights

Status: Accepted (2026-08-23)

## Context

The optimized objective was Δneff alone. For an abrupt junction the swept
charge per unit junction area scales as √N and Δn is linear in the swept
carriers, so with no penalty the optimum is the doping ceiling wherever the
mode is: the 73-iteration run of ticket 25 railed |θ| on both sides of the
junction (|N| ≈ 1e19) and read VπLπ = 0.87 V·cm at a free-carrier loss of
hundreds of dB/cm. Real depletion modulators are bounded by that loss; the
literature compares on VπLπ × α (10–30 V·dB for good devices). A loss term had
been deferred since ticket 03.

Two ways to evaluate the modal loss were open:

1. **Complex permittivity in gyptis.** Make the rib permittivity complex
   (`Im ε` from the Soref–Bennett absorption), solve the complex eigenproblem
   and read `Im neff`. Exact, but it changes the assembled system (the real
   doubling of a complex problem already carries a non-negligible PML `Im λ`,
   ticket 24), the tracking window, the VJP, and the container image.
2. **First-order perturbation with the existing eigen-adjoint.** The
   Hellmann–Feynman adjoint gyptis already computes, `w_cell = ∂(neff²)/∂ε_cell`,
   is exactly the mode-overlap weight a cell's imaginary permittivity enters
   `Im(neff²)` with. The modal loss is then `α_mode = (n_si/neff) Σ w_cell α_cell`
   — the textbook confinement-weighted loss, and nothing new has to be solved.

## Decision

Option 2. `pipeline_with_terms` returns `(objective, PipelineTerms(delta_neff,
modal_loss_db_cm))` with objective `Δneff − w·α_mode`; `--loss-weight w`
defaults to 0, so the default run optimizes exactly what it did and only
*reports* the loss and the V·dB figure of merit.

- `α_cell` is the **absolute** Soref–Bennett absorption `C_e·N_e + C_h·N_h` of
  the **0 V** carriers (the insertion loss of the unbiased phase shifter),
  carried onto the design cells by the same mesh transfer as the permittivity.
- The overlap weights are read **once**, at the uniform background
  (`pipeline.read_mode_overlap`: one eigensolve + one adjoint through the
  bound gyptis component), and passed to the pipeline as a constant. They are
  not re-derived at the perturbed mode: that would need second-order
  eigen-derivatives the adjoint does not provide, and the carrier-induced
  Δε ~ 1e-3 does not reshape the mode.
- The loss gradient flows through the ChargeTransport adjoint at 0 V, already
  in the graph; the loss costs no extra solver call per iteration.
- Loss is counted on the **design cells only** (the rib interior, where the
  permittivity is modulated). Slab doping in the mode tail is not penalized.

## Consequences

- Every run reports modal loss (dB/cm) and VπLπ × α (V·dB) next to VπLπ, in the
  log, the history/checkpoint (`objective`, `delta_n_eff`,
  `modal_loss_db_cm`) and a second panel of the convergence figure; the
  literature comparison is now possible without a second tool.
- With `w > 0` the problem is a real trade-off instead of "push to the rail".
  `w` is in neff per dB/cm: on the committed container mesh the rib's weights
  sum to 0.57, 1e19 n-type across the rib reads ~260 dB/cm, so w ≈ 1e-6 makes
  that cost ~2.6e-4 of Δneff (comparable to the railed optimum's 4.5e-4).
- **Trade-off — first order and frozen.** The loss is accurate to the extent
  the background mode is the mode; a design that moves the mode (a strongly
  asymmetric rib permittivity, a higher-order target) would mis-weight it.
  The exact complex eigenproblem remains the upgrade path and would slot in
  behind the same `PipelineTerms` seam.
- **Trade-off — rib only.** A future extension should carry the overlap
  weights onto the slab cells too (gyptis treats the slab as background
  silicon today), or the optimizer can learn to hide loss in the slab.
- The VπLπ × α figure of merit is reported, not optimized: minimized alone it
  favours ever-lighter doping (α ∝ N, Δneff ∝ √N), so the weighted sum is the
  objective and the FOM is the comparison number.
