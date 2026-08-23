# Results

All figures are from one run on the container mesh —
`prismo run --use-containers --loss-weight 1e-5 --mesh-size 0.05 --r-min 0.1`
(0.05 µm silicon elements, 0.1 µm filter radius), 192 MMA iterations in
~62 min on a laptop; `outputs/` holds the PDFs of your own runs and
`make figures` refreshes `docs/figures/` from them.

## The gradient is validated before it is trusted

```{image} figures/gradient_validation.png
:width: 520px
:align: center
```

Composed adjoint vs central finite differences through filter → doping →
ChargeTransport (Julia, warm) → Soref–Bennett → gyptis: relative error
$\approx 2\times10^{-7}$ at $h = 10^{-3}$, following the $O(h^2)$ slope until
finite-difference round-off takes over (`make validate-gradient-containers`).

## The gradients do the work

```{image} figures/convergence.png
:width: 520px
:align: center
```

From the seeded lateral junction, MMA raises $\Delta n_\mathrm{eff}$ at −5 V
from $1.19\times10^{-4}$ to $3.52\times10^{-4}$ (×3), i.e. $V_\pi L_\pi$
from 3.26 to **1.10 V·cm**. Dips are rejected trials of the move-limited MMA,
kept in the record on purpose. (`prismo run` also re-solves the reported
design cold — worker reset, equilibrium from near-intrinsic, bias ramp — and
flags any warm/cold discrepancy, so the headline is a property of the design,
not of the solve path.)

```{image} figures/doping_evolution.gif
:width: 720px
:align: center
```

The net doping the two solvers saw at every evaluation: red n-type, blue
p-type, white the junction. The optimizer grows a curved junction that hugs the
optical mode.

## Before / after

```{image} figures/doping_field.png
:width: 760px
:align: center
```

Left: the seed, a lateral junction at $|N| \approx 3\times10^{17}\,\mathrm{cm^{-3}}$.
Right: the optimum — the junction curls into the rib and wraps the mode centre,
the 2D cross-section of the L-/U-shaped junctions the literature reaches by
hand, while the slab stays lightly doped where the mode does not reach.

## Where the modulation happens

```{image} figures/depletion_field.png
:width: 760px
:align: center
```

Carriers swept out between 0 V and −5 V at the optimum (orange, log scale),
under the mode's $|E|$ contours. The depleted region sits on the mode peak;
doping that the mode cannot see would be loss for nothing, and the objective
knows it.

## The loss is watched, not ignored

```{image} figures/loss_convergence.png
:width: 520px
:align: center
```

Modal free-carrier loss $\alpha$ of the unbiased device and the
efficiency–loss figure of merit $V_\pi L_\pi\cdot\alpha$ at every iteration.
$\alpha$ first dips (doping the mode cannot see is pruned), then the optimizer
buys ~2 dB/cm back — 4.29 to 6.34 dB/cm — where it pays in
$\Delta n_\mathrm{eff}$, and the figure of merit **halves, from 14.0 to
7.0 V·dB** (good depletion modulators sit at 10–30 V·dB).

```{image} figures/tradeoff.png
:width: 520px
:align: center
```

The same run as a path from seed to optimum in the $(\alpha, \Delta n_\mathrm{eff})$
plane against iso-$V_\pi L_\pi\cdot\alpha$ curves. `--loss-weight` moves the
optimum along this trade-off.

## The mode

```{image} figures/mode_field.png
:width: 420px
:align: center
```

The tracked fundamental guided mode of the rib on the shared mesh
(`--mode-index k` targets a higher-order one).

## Scope

2D cross-section, one bias pair (0 / −5 V), first-order (overlap-weighted)
loss on the rib cells only, Boltzmann statistics, no implant process model — a
prototype that points at the real device, not a tape-out.
