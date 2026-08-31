# Physics and model

## Device

A silicon-on-insulator rib waveguide at $\lambda = 1.55\,\mu\mathrm{m}$:
500 nm × 220 nm silicon rib on a thin silicon slab, oxide below, beside and
above, two Ohmic contacts on the slab shoulders. The 2D cross-section is the
computational domain; the optical solver surrounds it with a PML frame. The
design space is the net doping $N(x)$ at every silicon node of the shared mesh.

## Design field and doping map

The design variable is a **signed field** $\theta_i \in [-1, 1]$ per silicon
node. It is first smoothed by a linear density filter of radius $r_\mathrm{min}$
(Andreassen et al. 2011):

$$
\tilde\theta_i = \frac{\sum_j H_{ij}\,\theta_j}{\sum_j H_{ij}}, \qquad
H_{ij} = \max\bigl(0,\; r_\mathrm{min} - \lVert x_i - x_j \rVert\bigr),
$$

which enforces a minimum feature size (an implant straggle) and removes
checkerboards; being linear and mean-preserving it is sign-agnostic. The
filtered field is mapped to a net doping in $\mathrm{cm^{-3}}$ by a
zero-referenced, antisymmetric log map

$$
N(\theta) = \operatorname{sign}(\theta)\, N_\mathrm{ref}\,
\bigl(10^{\,s\,|\theta|} - 1\bigr), \qquad
N_\mathrm{ref} = 10^{17}\,\mathrm{cm^{-3}},\; s = 2,
$$

so $\theta = 0$ is intrinsic, $|\theta| = 1$ is the doping ceiling
$|N| \approx 10^{19}\,\mathrm{cm^{-3}}$, $\theta > 0$ is n-type and the PN
junction is exactly the zero crossing of $\theta$. The map is $C^1$ through
zero (its derivative $N_\mathrm{ref}\ln 10\, s\, 10^{s|\theta|}$ is even and
continuous) and is given a custom JVP, because plain autodiff of the
$\operatorname{sign}(\theta)|\theta|$ form collapses to zero at the junction.
There is no SIMP penalization: intermediate values are physically realizable
doping levels.

Centring the span on $10^{17}$ keeps the seeded junction *partially* depleted
at −5 V — the regime a carrier-depletion modulator works in. At
$\sim 10^{15}\,\mathrm{cm^{-3}}$ the depletion width (~1.6 µm) swamps the rib
and the reverse-bias carrier field carries no bulk-doping signal at all.

## Carrier transport (ChargeTransport.jl)

Steady-state van Roosbroeck system on the silicon subdomain, solved by
[ChargeTransport.jl](https://github.com/WIAS-PDELib/ChargeTransport.jl) on top
of VoronoiFVM (finite volumes, Scharfetter–Gummel fluxes), with Boltzmann
statistics:

$$
-\nabla\cdot(\varepsilon_s \nabla\psi) = q\,(p - n + N), \qquad
\nabla\cdot \mathbf{J}_n = q R, \qquad
\nabla\cdot \mathbf{J}_p = -q R,
$$

$$
\mathbf{J}_n = -q\,\mu_n\, n\, \nabla\varphi_n, \qquad
\mathbf{J}_p = -q\,\mu_p\, p\, \nabla\varphi_p, \qquad
n = N_c\, \mathcal{F}(\eta_n), \quad p = N_v\, \mathcal{F}(\eta_p),
$$

where $\mathcal{F}$ is the Boltzmann exponential of the reduced distance
$\eta$ between each band edge and its quasi-Fermi potential, the unknowns are
$(\psi, \varphi_n, \varphi_p)$ per node and $N$ donor-positive
(ChargeTransport.jl's own `doping` is acceptor-positive and in SI; the sign and
unit flip is applied at the single point where doping enters the system, and
undone on the VJP). Shockley–Read–Hall recombination through mid-gap traps
($\tau = 100$ ns, trap density $\approx n_i$) is on: it does not change the
seeded junction to five digits, but without any generation/recombination the
reverse-bias steady state of the free-form designs the optimizer proposes
(rail-level doping, sign flips, floating p-pockets) is not unique — the same
doping solved to depletion on a cold ramp and to injection on a warm start.
Thermal generation pins the minority quasi-Fermi level in depleted and floating
regions and removes the spurious branch.

The contacts are Ohmic: at equilibrium they enforce local charge neutrality,
out of equilibrium $\psi = \psi_\mathrm{eq} + U$ as a Dirichlet condition. The
run uses one **bias pair**: $U = 0$ (reference) and $U = -5$ V on the p-side
contact (reverse bias). Mesh coordinates are µm; the solver scales them to
metres on load.

## Carriers to permittivity (Soref–Bennett)

The free-carrier plasma-dispersion model of Soref & Bennett (IEEE JQE 23, 1987)
at 1.55 µm, applied to the carrier *change* relative to equilibrium,
$\Delta N_e = n - n_\mathrm{eq}$, $\Delta N_h = p - p_\mathrm{eq}$
(in $\mathrm{cm^{-3}}$):

$$
\Delta n = -\bigl(8.8\times10^{-22}\,\Delta N_e + 8.5\times10^{-18}\,\Delta N_h^{0.8}\bigr),
\qquad
\Delta \alpha = 8.5\times10^{-18}\,\Delta N_e + 6.0\times10^{-18}\,\Delta N_h
\;[\mathrm{cm^{-1}}],
$$

extended antisymmetrically, $\Delta N^{B} \to \operatorname{sign}(\Delta N)\,
|\Delta N|^{B}$, because the injection-calibrated fractional power is undefined
for depletion ($\Delta N < 0$); with the odd extension depletion raises the
index and lowers absorption. The permittivity perturbation is the first-order
$\Delta\varepsilon = 2\, n_\mathrm{Si}\, \Delta n$ with $n_\mathrm{Si} = 3.4757$.
The nodal $\Delta\varepsilon$ is carried onto the optical design cells by the
mesh-transfer operator (each design cell is a triangle of the shared mesh, so
its value is the mean of its three vertices — an exact restriction).

## Optical mode (gyptis / FEniCS)

gyptis solves the full-vector eigenmode problem of the cross-section,
$\mathbf{E}(x,y,z) = (\mathbf{E}_t, E_z)(x,y)\, e^{-i k_z z}$, from Maxwell's
curl-curl equation $\nabla\times\nabla\times\mathbf{E} = k_0^2\,\varepsilon\,\mathbf{E}$
with $k_0 = 2\pi/\lambda$, discretized as a generalized eigenproblem

$$
A(\varepsilon)\, x = \lambda\, B(\varepsilon)\, x, \qquad \lambda = k_z^2,
\qquad n_\mathrm{eff} = k_z / k_0,
$$

on the whole domain (oxide, slab, rib, PML frame), with the rib interior's DG0
cells carrying the design permittivity $\varepsilon_\mathrm{Si} + \Delta\varepsilon$
and everything else constant. The solve is a shift-invert Krylov–Schur
eigensolve (SLEPc) that tracks one physical branch: the fundamental guided mode
($n_\mathrm{clad} < n_\mathrm{eff} < n_\mathrm{core}$) by default, or the
$k$-th guided mode with `--mode-index k`, followed by nearest eigenvalue across
designs so neighbouring branches swapping rank cannot redirect an optimization.

## Figures of merit

The **effective-index modulation** is signed,

$$
\Delta n_\mathrm{eff} = \mathrm{Re}\,n_\mathrm{eff}(V_\mathrm{bias}) - \mathrm{Re}\,n_\mathrm{eff}(0),
$$

positive for depletion, and is the quantity optimized. The reported efficiency
is the field-standard

$$
V_\pi L_\pi = \frac{|V_\mathrm{bias}|\,\lambda}{2\,\Delta n_\mathrm{eff}}
\quad [\mathrm{V\cdot cm}],
$$

derived from $\Delta n_\mathrm{eff}$ assuming a linear phase response (smaller
is better).

The **modal free-carrier loss** of the unbiased device is the first-order,
overlap-weighted Soref–Bennett absorption of the 0 V carriers on the design
cells,

$$
\alpha_\mathrm{mode} = \frac{n_\mathrm{Si}}{n_\mathrm{eff}} \sum_{c \in \text{design cells}}
w_c\, \alpha_c, \qquad
\alpha_c = C_e N_{e,c} + C_h N_{h,c}, \qquad
w_c = \frac{\partial (n_\mathrm{eff}^2)}{\partial \varepsilon_c}\Big|_{\text{background}},
$$

reported in dB/cm. $w_c$ is the mode-overlap weight the eigen-adjoint already
computes: an imaginary permittivity $\mathrm{Im}\,\varepsilon_c = n_\mathrm{Si}\,
\alpha_c\, \lambda / 2\pi$ in a cell shifts $\mathrm{Im}(n_\mathrm{eff}^2)$ by
$w_c\, \mathrm{Im}\,\varepsilon_c$, and the modal power loss $2 k_0\,
\mathrm{Im}\, n_\mathrm{eff}$ follows — for a uniform core this is the textbook
confinement-weighted loss $\Gamma\,\alpha\, n_\mathrm{Si}/n_\mathrm{eff}$.
The weights are evaluated once at the uniform background and frozen (the
carrier-induced $\Delta\varepsilon \sim 10^{-3}$ does not reshape the mode).

### Limits of the loss model

Three caveats travel with $\alpha_\mathrm{mode}$, and all three make it an
*underestimate* of what a fabricated device would measure.

It is counted on the design cells — the rib interior — only. The slab is
background silicon to the eigensolver, so doping that sits in the mode's
evanescent tail costs the objective nothing; the optimizer is free to place
loss there and not be charged for it. Reading the reported figure alongside the
doping map is the practical guard until the weights are carried onto the slab
cells too.

The frozen weights hold only while the mode stays where it was computed. A
design whose permittivity is strongly asymmetric across the rib, or a run
targeting a higher-order mode with `--mode-index`, moves the field enough that
weights taken at the uniform background mis-weight it — the first-order
estimate degrades before the formula does. Solving the complex eigenproblem
directly, with $\mathrm{Im}\,\varepsilon$ from the same Soref–Bennett
absorption, is the exact route and needs no change to the objective.

Finally, $\alpha_\mathrm{mode}$ is free-carrier absorption and nothing else.
Sidewall-roughness scattering, contact and metal absorption, and any loss
outside the rib are absent, so the number is a floor on the propagation loss of
the real cross-section rather than a prediction of it.


The literature's efficiency–loss **figure of merit** is
$V_\pi L_\pi \times \alpha_\mathrm{mode}$ in V·dB (10–30 V·dB for good
depletion modulators). It is reported, not optimized: minimized alone it
favours ever-lighter doping ($\alpha \propto N$, $\Delta n_\mathrm{eff} \propto \sqrt N$).

## Objective

$$
\max_{\theta \in [-1,1]^n}\; J(\theta) = \Delta n_\mathrm{eff}(\theta) - w\, \alpha_\mathrm{mode}(\theta),
$$

with `--loss-weight` $w$ in $n_\mathrm{eff}$ per dB/cm (default 0: the loss is
reported only). Without the penalty the optimum of the problem as posed is
"dope as hard as allowed wherever the mode is": swept charge grows as
$\sqrt N$ with no cost, so the optimizer rails $|\theta|$ at the mode centre.
With $w > 0$ the problem is a real trade-off.
