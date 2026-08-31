# Implementation choices

The decisions that shape the code, and why. Each is the version that survived;
the trade-offs are stated so the next change knows what it is trading.

## One mesh, authored by the optical solver

Both solvers discretize the same device on **one** Gmsh mesh. The gyptis
Tesseract builds the geometry — oxide substrate, silicon slab, a rib band that
is oxide either side of the embedded 500 nm × 220 nm silicon rib, oxide clad, a
matched PML frame, two Ohmic contact lines on the slab shoulders — and
publishes the mesh through a `write_mesh` operation. gyptis solves the whole
domain; the ChargeTransport Tesseract extracts the silicon subdomain
(`slab` + `rib_silicon`), gathers the full-mesh doping onto those nodes, and
scatters carriers and gradient back onto the full node set.

*Why.* Meshing the device twice meant carrying nodal carrier fields onto the
optical design cells by point location and barycentric interpolation — an
operator with no exact inverse and no natural test oracle that every gradient
had to cross — plus a silent node-ordering reconciliation between two readers
of two files. With one mesh the transfer is an exact restriction (each design
cell is a triangle of the mesh: weight 1/3 on each vertex), silicon-only support
and partition of unity hold by construction, and there is one node set.

*Trade-offs.* The optical solver owns the device mesh, so refinement is driven
by what the eigensolve needs (`--mesh-size`), and the app cannot mesh without a
running gyptis container — a container run that lacks it raises. Tagging the
contact lines leaves orphan DOFs on the embedded edges whose zero rows would
zero-pivot the eigensolve; they are pinned with a unit diagonal, which places
their eigenvalue far outside the guided window. Without containers
`prismo.waveguide_mesh` writes a simpler rib mesh (no PML frame) with the same
µm units and the same silicon group names — a second author, not a second
contract.


## A pivoting direct solver behind the eigensolve

The shift-invert transform $(A - \sigma B)^{-1} B$ of the Krylov–Schur
eigensolve is factorized by UMFPACK and the solver runs at tolerance $10^{-10}$.

*Why.* The assembled system is an indefinite saddle point ($A$ has zero
diagonals, $\lVert A\rVert \approx 5\times10^{6}$ against $\lVert B\rVert
\approx 10^{3}$). PETSc's native sparse LU has no threshold pivoting, and its
solves were accurate to about $10^{-5}$: the eigenvalue converged at any
requested tolerance to the *same* value whose true residual was $10^{-5}$, a
few $10^{-7}$ relative off the answer an independent ARPACK solve returned,
with a round-off pattern that moved with the input. On the objective that was
a $2\times10^{-3}$ relative white noise floor on $\Delta n_\mathrm{eff}$ while
the adjoint said the true change along the same line was $10^{-9}$ — the
optimizer stalled inside a shrinking move-limit box, with a 9 % bias on the
headline on top. UMFPACK, MUMPS and SuperLU all agree with ARPACK to
$10^{-13}$; UMFPACK was the fastest in this image. The objective line scan
(`prismo probe-objective`) that found this stays as the smoothness gauge for
any future solver or parameterization change.

*Trade-off.* The image must ship UMFPACK (the FEniCS base does); a build
without it fails the eigensolve outright rather than falling back to the
inaccurate native LU.

## Loss as a first-order perturbation with frozen overlap weights

The modal free-carrier loss is $\alpha_\mathrm{mode} = (n_\mathrm{Si}/n_\mathrm{eff})
\sum_c w_c \alpha_c$, with $\alpha_c$ the absolute Soref–Bennett absorption of
the 0 V carriers on each design cell and $w_c = \partial(n_\mathrm{eff}^2)/
\partial\varepsilon_c$ the eigen-adjoint evaluated once at the uniform
background and frozen. Its gradient flows through the 0 V ChargeTransport
adjoint already in the graph — no extra solver call.

*Why.* Optimizing $\Delta n_\mathrm{eff}$ alone drives $|\theta|$ to the rail
wherever the mode is (swept charge grows as $\sqrt N$ at no cost): a
$\Delta n_\mathrm{eff}$-only run reached 0.87 V·cm at hundreds of dB/cm. Real
depletion modulators are bounded by that loss. The alternative — a complex
permittivity in gyptis and $\mathrm{Im}\, n_\mathrm{eff}$ from a complex
eigenproblem — is exact but changes the assembled system, the tracking window,
the VJP and the image, whereas the overlap weights are exactly what the
Hellmann–Feynman adjoint already computes and the textbook
confinement-weighted loss falls out.

*Trade-offs.* First order and frozen: accurate as long as the background mode
is the mode (a strongly asymmetric rib permittivity or a higher-order target
would mis-weight it). Counted on the rib's design cells only; slab doping in
the mode tail is not penalized yet. The complex eigenproblem remains the
upgrade path behind the same `PipelineTerms` seam.

## A persistent Julia worker with warm starts

The ChargeTransport Tesseract's `tesseract_api.py` owns one long-lived Julia
process (`scripts/worker.jl`) and talks to it over JSON lines with NPY files
for arrays. The worker keeps the last converged equilibrium and biased
solutions as Newton starting points for the next design. When the direct warm
start at −5 V fails it continues by **doping homotopy at fixed bias**
($d(t) = d_\mathrm{prev} + t\,(d_\mathrm{new} - d_\mathrm{prev})$, warm-starting
each step) and only then falls back to the **cold bias ramp** from equilibrium.
A wall-clock **solve budget** (`PRISMO_CT_SOLVE_BUDGET_S`, 120 s) is checked
inside every continuation loop; when it runs out the request returns a
`SolveBudgetExceeded` error, the worker and its warm solutions survive, and the
optimizer treats the design as a step too large. A hung process is the only
thing the Python-side timeout (`PRISMO_CT_JULIA_TIMEOUT_S`, 600 s) kills.

*Why.* Julia's compile time and a cold bias ramp per evaluation would dominate
the iteration; a hard-but-solvable free-form design takes 12–70 s warm. A
worker that dies on one hard design would cost the whole run.

*Trade-off — state.* A warm-started answer may depend on the solve history.
The `apply` endpoint therefore has a `reset` operation (drop every warm
solution); `prismo run` re-solves the best design **cold** after the
optimization and prints warm and cold $\Delta n_\mathrm{eff}$ side by side
(a relative mismatch above $10^{-4}$ is flagged in the log and on the
convergence figure), and `validate-gradient --cold` resets before every
finite-difference evaluation. SRH recombination is on so that the reverse-bias
steady state is unique in the first place (see *Physics*). The heavy Julia
packages are precompiled into a project-built base image, so a `tesseract
build` after a code change relayers only the Python venv and scripts.

## Move-limited MMA

NLopt's MMA runs **one fresh subproblem per outer step** inside a trust box
$x \pm \Delta$ intersected with $[-1, 1]$, with a single trial evaluation. A
physics failure or a non-improving trial means "step too large": $\Delta$ is
halved and the step re-proposed from the same iterate; an improving step is
accepted and $\Delta$ regrows (×2, capped at `--move-limit`, default 0.05).
The loop stops on `ftol`, `max-iter`, or a bounded number of consecutive
halvings. The best feasible design and the full history are written to
`checkpoint.json` after every evaluation, so a killed run still yields a
result and `prismo animate` can replay it.

*Why.* Unconstrained MMA drove the design to the doping rails with a dozen
junction sign flips per iteration and produced designs the drift-diffusion
solve could not converge on. The move limit keeps every proposed design within
the solver's warm-start radius and turns a failed solve into a recoverable
event.

## Signed design field, silicon design nodes

One signed field $\theta \in [-1, 1]$ per **silicon** node (slab and rib):
$\operatorname{sign}(\theta)$ is the free P/N polarity and the junction is the
zero crossing, so the optimizer moves the junction by moving that crossing.
Nodes outside silicon carry no variable: nothing reads them (ChargeTransport
gathers doping on the silicon subgrid; every design cell is a rib triangle
whose vertices are silicon nodes), and their gradient is exactly zero unless
the filter radius reached them — a parameterization with no physical referent.
Dropping them shrinks the dense filter matrix by about 2.6× on the local mesh.

*Why one field, not donor + acceptor.* Two fields make counterdoping
representable and the junction a degenerate function of both; one field makes
a single sign crossing a single junction. Every run seeds a junction
(`--seed lateral|vertical|u` at $|\theta| = 0.3$, n-type on the left slab edge,
p-type on the right so both carrier populations reach a contact and the seed is
reverse-biased); a uniform start has no junction at all.
