# Glossary

Every domain term the code, the CLI and the figures use, defined once. The reasoning behind the bigger choices is in *Implementation choices*.

---

## Signed design field (θ)

**θ ∈ [−1, 1] per design node** — the single design variable of the doping optimization. See **Design nodes** for which mesh nodes those are.

The field is *signed*: `sign(θ)` is the free P/N polarity and `|θ|` is how hard the node is doped. There is one field, not a donor field plus an acceptor field, so counterdoping is not representable and the P/N junction is exactly the θ zero-crossing — the optimizer moves the junction by moving that crossing.

- θ = 0 → net-intrinsic (zero net doping)
- θ > 0 → n-type, θ < 0 → p-type (see **Doping sign convention**)
- |θ| = 1 → the doping ceiling, |N| ≈ 10¹⁹ cm⁻³

Not to be confused with the SIMP material density of classical structural topology optimization (ρ = 0 void, ρ = 1 solid, intermediate values penalized). There is no SIMP penalization here: dopants diffuse, so sharp boundaries are not desirable, and intermediate values are physically realizable doping levels.

## Design nodes

The shared-mesh nodes that carry a design variable: the **silicon** nodes, i.e. the ones touched by the `slab` and `rib_silicon` triangles — the same pair ChargeTransport collects. The density filter and the MMA bound vector are sized by this set, and the filtered field is scattered back into full gmsh node order (θ = 0, net-intrinsic, elsewhere) before the doping map, so `mesh_ref`'s node ordering and the `(n_design_cells, n_nodes)` mesh transfer are unchanged.

Nodes outside silicon are not design variables because nothing reads them: ChargeTransport gathers doping on the silicon subgrid, and every gyptis design cell is a rib triangle whose three vertices are silicon nodes. Their gradient is exactly zero unless the filter radius reaches a silicon node, in which case they dope the device from outside it — a parameterization with no physical referent either way. On the committed 555-node local mesh this drops 212 of the variables and shrinks the dense filter matrix, which is `(n_design, n_design)`, by about 2.6×.

Distinct from **design cells**, which are the optical DG0 cells of the rib interior. The doping design set spans slab + rib (carriers must flow to the contacts); only the rib interior has its permittivity modulated.

## Zero-referenced log doping map

The map from the signed design field to net doping, in cm⁻³:

```
N(θ) = sign(θ) · N_ref · (10^(span·|θ|) − 1),   N_ref = 10¹⁷ cm⁻³,  span = 2
```

Zero-referenced (`N(0) = 0`) and antisymmetric, so a single sign crossing is a single junction. It is C¹ through θ = 0: the derivative `N_ref · ln10 · span · 10^(span·|θ|)` is even and continuous, and is supplied as a custom JVP because plain autodiff of the `sign(θ)·|θ|` form collapses to zero exactly at the junction.

The reference concentration is a *depletion-modulator* concentration, not a near-intrinsic one. At |N| ≈ 3×10¹⁵ the −5 V depletion width (~1.6 µm) swamps the 500 nm × 220 nm rib, so the rib depletes fully and the reverse-bias carrier field carries no bulk-doping signal at all. Centring the span on 10¹⁷ keeps the seeded junction *partially* depleted — the regime a real carrier-depletion modulator works in — while |θ| = 1 still lands at the ≈ 10¹⁹ cm⁻³ ceiling.

## Doping ceiling

**|N| ≈ 10¹⁹ cm⁻³**, reached at |θ| = 1. The largest reverse-bias-stable concentration on the shared mesh, and within the boron/phosphorus solid-solubility and Boltzmann-statistics window that ChargeTransport.jl's model assumes.

## Doping sign convention

**PRISMO's net doping is donor-positive and in cm⁻³**: θ > 0 → N > 0 → n-type. That is the domain term; every field named "doping" that crosses a PRISMO boundary (design field, tesseract input schema, figures) uses it.

This matters because the opposite convention meets it inside the ChargeTransport tesseract: ChargeTransport.jl's `ParamsNodal.doping` is **acceptor-positive and SI** (its Poisson source makes charge neutrality read `p − n = doping`, so a positive entry is p-type). Confusing the two silently mirrors the device. The reconciliation is one constant, `PRISMO_DOPING_TO_CT` in `ct_common.jl`, applied at the single point where doping is written into the ChargeTransport system and undone on the VJP.

## Density filter

A linear convolution operator applied to θ *before* physics evaluation:

```
θ̃ = Hθ / Hsum,   H_ei = max(0, rmin − dist(e, i))
```

Defined in Andreassen et al. 2011 (88-line topology optimization code). Prevents checkerboard oscillations and enforces minimum length scale. Adjacent to but distinct from the sensitivity filter (99-line code), which operates on gradients rather than densities. Being linear and mean-preserving, it is sign-agnostic — it smooths a signed field without biasing its polarity.

**rmin**: filter radius (physical length, must span 2-3 mesh elements). In µm, matching the shared mesh's units. Default `--r-min 0.10` (3–4 elements of the 0.04 µm container mesh, comparable to implant straggle); the earlier 0.05 reached one neighbour and left checkerboard-like slab values.

## Doping profile

The spatial distribution of net donor/acceptor concentration across the waveguide cross-section, `N(θ(x))` at each mesh node. This is what the optimizer adjusts.

## Free-form doping optimization

The design variable is the doping at *every design node* (hundreds to thousands of variables). Not a parameterized junction (a few scalar parameters like junction depth, peak concentration, lateral offset). This is the project's mode.

## MMA (Method of Moving Asymptotes)

Svanberg's 1987 convex-separable approximation optimizer. Builds a conservative rational-function approximation of the objective and constraints around the current iterate, solves the separable subproblem via a dual method, and moves the asymptotes based on iterate behavior. The canonical topology optimization solver.

**Implementation:** NLopt (`NLOPT_LD_MMA`, PyPI `nlopt`), C library with Python bindings, driven **move-limited** (see below): one fresh MMA subproblem per outer step inside the move-limit box. There is no CCSAQ fallback any more.

## Move limit

The per-iteration cap Δ on how far any design variable may move (`--move-limit`, default 0.05 on θ). Each outer step runs a fresh NLopt MMA instance with bounds `x ± Δ` intersected with `[−1, 1]` and one trial evaluation, so no variable moves more than Δ per iteration whatever the asymptotes would do. A physics failure or a non-improving trial means "step too large": Δ is halved and the step re-proposed from the same iterate; an improving step is accepted and Δ regrows (×2, capped at the configured limit). The loop stops on ftol, max-iter, or after a bounded number of consecutive halvings (stall). Added after NLopt's unconstrained MMA drove the design to the doping rails with 13–16 junction flips per iteration and an unsolvable design at iteration 11. `outputs/checkpoint.json` (best θ + history) is written after every evaluation.

## NLopt

The C library (`github.com/stevengj/nlopt`, MIT license) providing the MMA implementation used in this project. Installed via `pip install nlopt` (precompiled wheels). Requires NumPy arrays — JAX gradients are evaluated to concrete NumPy before entering the NLopt call. Exposes the algorithm as `nlopt.LD_MMA`.

## PN junction phase shifter

A silicon rib waveguide with a lateral PN junction across the optical mode region. Reverse bias depletes carriers, changing the local refractive index via the plasma dispersion effect (Soref-Bennett), which shifts the effective index (neff) of the guided optical mode. The Δneff between 0 V and reverse bias is the figure of merit.

## Tesseract pipeline

Two containerized Tesseracts composed via `tesseract-jax`:

1. **ChargeTransport Tesseract** (Python 3.12 + Julia): [ChargeTransport.jl](https://github.com/WIAS-PDELib/ChargeTransport.jl) drift-diffusion solver on the silicon subdomain. Input: net doping N(x) on the shared mesh, bias voltage V. Output: carrier densities n(x), p(x). Differentiation: discrete adjoint (Jacobian transpose solve).
2. **gyptis Tesseract** (conda, FEniCS): EM eigenmode solver on the full domain. Input: permittivity field ε(x) on the design cells via Soref-Bennett from the carrier densities. Output: complex effective index neff. Differentiation: Hellmann-Feynman eigen-adjoint.

The Soref-Bennett coupling layer sits between them, in the app.

## Soref-Bennett coupling layer

The mapping from carrier densities to optical permittivity at λ = 1.55 μm:

```
Δn = −8.8×10⁻²² · ΔN_e − 8.5×10⁻¹⁸ · (ΔN_h)^0.8
Δα = 8.5×10⁻¹⁸ · ΔN_e + 6.0×10⁻¹⁸ · ΔN_h
```

Ref: Soref & Bennett, IEEE JQE 23(1):123–129, 1987. Coefficients here are for λ=1.55 μm.

## Shared mesh

**One** Gmsh 2D triangular mesh of the SOI cross-section, authored by the gyptis Tesseract and consumed by both solvers. It carries the full optical domain — oxide substrate, silicon slab, the silicon rib embedded in an oxide band, oxide clad, a matched PML frame, and the two Ohmic contact lines — refined over the rib so the eigensolve lands on the fundamental *guided* mode rather than a leaky one.

gyptis solves on the whole domain. **ChargeTransport is restricted to the silicon subdomain** (slab + rib): its tesseract extracts the silicon grid, gathers the full-mesh doping onto the silicon nodes, and scatters the silicon carriers and gradient back onto the full node set. Mesh coordinates are µm; ChargeTransport scales them to metres on load, because it solves in SI.

A run without containers has no gyptis to author the mesh, so `prismo.waveguide_mesh` writes a simpler rib mesh (no PML frame) in its place. That is a second *author*, not a second contract: ChargeTransport reads whichever file the run wrote, so the local author emits the same µm coordinates and the same `slab` / `rib_silicon` silicon groups. An earlier version authored metres under `silicon` / `oxide`, which made the µm-scaled `r_min` an all-pairs filter and left the Julia silicon lookup with no group it recognised.

See *Implementation choices*.

## Design cells

The DG0 cells of the silicon rib interior — the cells whose permittivity gyptis modulates, and the only place θ has optical effect. Inset from the PML by construction, so a varying ε never touches one. Their vertices are a subset of the **design nodes**; the slab carries design nodes but no design cells.

## Mesh transfer

The nodal-field → design-cell operator. Since both solvers share one mesh, it is an *exact local restriction*, not an interpolation: each design cell is a triangle of the shared mesh, so its value is the mean of its three vertex nodal values — weight 1/3 on each vertex, zero elsewhere. Silicon-only support and partition of unity hold by construction.

## Δneff (effective index modulation)

The figure of merit: `Δneff = Re[neff(V_bias)] − Re[neff(0)]`. **Signed** — depletion (the physical reverse-bias response) raises the index, so a physical optimum is positive; the sign is physically determined, so the optimizer maximizes Δneff directly (no `hypot`/magnitude fold, no epsilon floor). Larger Δneff → more efficient phase modulation → shorter device, lower power. The optimization maximizes this quantity.

**This project:** single bias pair V = 0 and V_bias = −5 V. Two ChargeTransport solves per iteration (one per bias state), two adjoint solves per gradient evaluation.

Loss penalty: see **Modal loss** — `--loss-weight w` optimizes `Δneff − w·α_mode`; the default `w = 0` optimizes Δneff alone and only reports the loss.

## VπLπ (modulation efficiency)

The field-standard modulation-efficiency headline, in V·cm: `VπLπ = |V_bias|·λ / (2·Δneff)` at the fixed −5 V bias (λ = 1.55 µm). Derived from Δneff assuming a linear phase response — the length for a π shift is `Lπ = λ/(2·Δneff)`. Smaller `|VπLπ|` is better. **Reported, not optimized** — signed Δneff is the optimized proxy; VπLπ carries its sign and diverges as Δneff → 0. See `vpi_lpi_v_cm` in `pipeline.py`.

## Modal loss (free-carrier absorption of the mode)

**`α_mode` [dB/cm] — the first-order propagation loss of the unbiased device from free-carrier absorption**, reported next to VπLπ for every run and, with `--loss-weight w > 0`, part of the objective `Δneff − w·α_mode` (`w` in neff per dB/cm). Without it the optimum of the problem as posed is "dope as hard as allowed wherever the mode is": swept charge grows as √N with no penalty, so the optimizer rails |θ| at the mode centre (0.87 V·cm at ~260 dB/cm).

Computed as `α_mode = (n_si/neff) · Σ_cell w_cell · α_cell`, with `α_cell` the absolute Soref–Bennett absorption `C_e·N_e + C_h·N_h` (cm⁻¹) of the **0 V** carriers on each design cell, and `w_cell = ∂(neff²)/∂ε_cell` the **mode-overlap weights** — the gyptis Hellmann–Feynman eigen-adjoint evaluated once at the uniform background (`pipeline.read_mode_overlap`) and frozen. An imaginary permittivity `Im ε = n_si·α·λ/(2π)` in a cell shifts `Im(neff²)` by `w·Im ε`, and the modal power loss `2k₀·Im neff` follows; the wavelength cancels. For a uniform core this is the textbook confinement-weighted loss `Γ·α·n_si/neff`; on the committed container mesh `Σ w ≈ 0.57` over the rib's 196 design cells. First order in the carrier perturbation (Δε ~ 1e-3 does not reshape the mode), and counted **on the design cells only** — slab doping in the mode tail is not yet penalized. The gradient flows through the same ChargeTransport adjoint at 0 V; no extra solve per iteration.

`VπLπ × α_mode` [V·dB] is the literature's efficiency–loss **figure of merit** (`loss_figure_of_merit_v_db`; 10–30 V·dB for good depletion modulators). Reported, not optimized: alone it favours ever-lighter doping (α ∝ N, Δneff ∝ √N), which is why the objective is the weighted sum. Drawn per iteration in `loss_convergence.pdf` and as the optimizer's path in the (α, Δneff) plane against iso-V·dB curves in `tradeoff.pdf` (the **trade-off figure**).

## Swept carriers (depletion figure)

`depletion_field.pdf` — `(n+p)(V_bias) − (n+p)(0 V)` per node at the reported design, drawn on the silicon elements with the tracked mode's |E| contours on top (`pipeline.carrier_fields`, `outputs.plot_depletion_field`). Negative where reverse bias depletes the junction — the only place the design changes the index — so the figure shows how much of the depleted region the mode sees, and how much doping sits in the mode for nothing but loss. Two warm ChargeTransport solves at the optimum, taken before the cold re-evaluation like the mode figure.

## Doping evolution (animation / replay)

`doping_evolution.{gif,mp4}` — the net doping the solvers saw at every optimizer evaluation, one frame per history record, on a colour scale fixed across the run, captioned with iteration, Δneff and modal loss; a trial the optimizer rejected (objective below the running best, see *Move limit*) is labelled as such. Every history record carries the raw design vector it evaluated (`design` in `checkpoint.json`), so `prismo animate` / `make animate` rebuilds the animation from the checkpoint and the run's `.msh` alone — filter, scatter and doping map, no solver. Checkpoints predating the key have nothing to replay.

## Junction seed (lateral / vertical / U)

`prismo run --seed {lateral,vertical,u}` picks the initial topology on the design nodes at `|θ| = 0.3` (`pipeline.seed_design_field`). `lateral` is the **seeded junction** above. `vertical`: in the rib, n-type below the rib's mid-height and p-type above it, with a p column along the rib's right wall (a quarter of the rib width) joining the p top to the right slab; the slab itself stays lateral. `u`: n-type wraps under (the lower third of the rib) and beside (the left wall and left slab) a p core, which with the right wall and right slab is p-type. Every seed keeps n-type on the left slab edge (anode) and p-type on the right (cathode at −5 V), so both carrier populations reach a contact and the seed is reverse-biased. The MMA optimum is local; the observed optimum from the lateral seed is an L-shaped junction, the 2D cross-section of the literature's L-/U-shaped junctions.

## Contact offset / domain width

Geometry knobs of the shared mesh: `--contact-offset` (gap from the rib edge to the near contact edge, default 0.2 µm; foundries use 0.5–1 µm) and `--domain-width` (the physical box width the slab spans, PML excluded; default 2.0 µm in the gyptis author, 3.0 µm in the local author). On the container path they reach the gyptis mesh author as `PRISMO_GYPTIS_CONTACT_OFFSET` / `PRISMO_GYPTIS_WIDTH`, read once at import like the mesh size; the contacts must stay inside the box. Widening the contact spacing takes the contact out of the mode tail and frees the optimizer to place intermediate doping between rib and contact.

## Mode index (targeted guided mode)

**Which guided mode the gyptis eigensolve tracks and Δneff is measured on.** `0` (default) is the fundamental — the largest-neff eigenpair inside the guided window `n_clad < neff < n_core`; `k` is the `k`-th guided mode in descending neff (for the 500 nm × 220 nm rib, index 1 is the first higher-order lateral mode). Ranked once, on the first solve of a geometry; every later solve of that geometry tracks the chosen branch by nearest eigenvalue, exactly as the fundamental is tracked, so an optimization follows one physical mode even when neighbouring branches swap rank. The index is part of the tracked-branch key and of the VJP session identity, so two mode targets on one geometry never share a branch. A first solve that resolves fewer guided modes than the target needs raises rather than returning a lower-order mode.

Set per run with `prismo run --mode-index k`; it is bound into the component bundle (`build_gyptis_components(mode_index=...)`), not threaded through `pipeline()`, because one run optimizes one mode. The headline mode figure is labelled with the targeted order.

## Single bias pair

The optimization uses exactly two ChargeTransport solves per iteration: one at V = 0 (reference), one at V = −5 V (max reverse bias). This captures the full bias-dependent depletion physics while keeping per-iteration cost at 2× the single-solve cost. More bias points (weighted sum) would increase cost linearly without guaranteed benefit for a carrier-depletion phase shifter.

## Seeded junction

The starting design: a signed lateral junction, θ = +0.3 left of the design nodes' median x and θ = −0.3 right of it, so |N| ≈ 3×10¹⁷ cm⁻³ on each side. That is the partially-depleted operating point — non-degenerate, reverse-bias convergent, and well inside the [−1, 1] bounds — and with the cathode on the right at −5 V it is the reverse-bias orientation. `sign(θ)` stays free, so the optimizer may move, dissolve, or reverse this junction. Every run path seeds it; a uniform start has no junction at all.

## Reverse bias

A negative voltage applied to the PN junction to widen the depletion region, sweeping out free carriers and thereby increasing the refractive index in the optical mode region. This project uses a fixed −5 V.

## Gradient validation

The primary headline deliverable: the composed adjoint (filter + doping map + ChargeTransport adjoint + Soref-Bennett + gyptis eigen-adjoint) checked against central finite differences along sampled θ directions, over a range of step sizes, and gated on a stated relative tolerance. Run by `prismo validate-gradient`; drawn as a relative-error-vs-step curve. See `outputs.validate_gradient`.

## Continuation

A topology-optimization technique where optimization parameters (filter radius, SIMP penalty) are gradually changed during the run to avoid local minima. **Not used in this project** — the run holds `r_min` fixed and there is no SIMP penalty to continue. Listed here because the term recurs in the topology-optimization literature this project borrows from, and because the ChargeTransport scripts separately use "continuation" for their own Newton homotopies (doping magnitude at equilibrium, the bias ramp, and the doping homotopy below), which is a different thing.

## SRH recombination

Shockley–Read–Hall recombination through mid-gap traps (τ = 100 ns, trap density ≈ nᵢ), **on** in the ChargeTransport system. Not because it changes the seeded junction — it does not, to five digits — but because without any generation/recombination the reverse-bias steady state of the free-form designs the optimizer proposes (rail-level doping, junction sign-flips, floating p-pockets) is **not unique**: the same doping solved to depletion on a cold ramp and to injection on a warm start, and the objective read two values for one θ. Thermal generation pins the minority quasi-Fermi level in depleted and floating regions and removes the spurious branch. `CT_SRH_LIFETIME_S` / `CT_SRH_TRAP_DENSITY_M3` in `ct_common.jl`.

## Warm start / doping homotopy / cold solve

The ChargeTransport worker keeps the last converged equilibrium and biased solutions as Newton starting points for the next design (**warm start**). When the direct warm start at −5 V fails, it continues by **doping homotopy at fixed bias** — `d(t) = d_prev + t·(d_new − d_prev)`, warm-starting each step — and only then falls back to the **cold** bias ramp from equilibrium. A **cold solve** is one with every warm solution dropped (the worker's `reset` operation): equilibrium from near-intrinsic doping, bias ramp from equilibrium — a function of the doping alone.

## Solve budget

The Julia-side wall-clock budget for one ChargeTransport request (`PRISMO_CT_SOLVE_BUDGET_S`, 120 s), checked inside every continuation loop. When it runs out the request returns a `SolveBudgetExceeded` error and the worker survives with its warm solutions; the Python request timeout (`PRISMO_CT_JULIA_TIMEOUT_S`, 600 s) is only a backstop for a hung process. The optimizer treats the failed request as a step too large (see *Move limit*).

## Cold re-evaluation

After the optimization, the run resets the ChargeTransport worker and solves the best design once more cold; warm and cold Δneff are both printed, VπLπ is computed from the cold value, and a relative mismatch above `COLD_REEVALUATION_RTOL` (1e-4) is a warning in the log and on the convergence figure. The headline is thus a property of the design, not of the solve history. `prismo validate-gradient --cold` resets before every finite-difference evaluation.

## Objective line scan / evaluation noise floor

The smoothness gauge: `prismo probe-objective` samples f(θ₀ + t·d) along one unit direction d at uniform spacing around a checkpoint design or the seed, fits a quadratic, and reports the fit residual, the white-noise amplitude implied by the samples' second differences, and the adjoint's directional derivative against the fitted slope. A **kink** shows as a structured residual that shrinks with the spacing; an **evaluation noise floor** — scatter that is a deterministic but non-smooth function of the design, from a solver's roundoff or tolerance — shows as a white residual that does not. The line scan found the 2e-3 relative floor behind an optimizer stall (PETSc's non-pivoting LU inside the gyptis shift-invert transform) and reads 2e-11 after the fix. See `outputs.scan_objective_line`, `outputs.probe_direction`.
