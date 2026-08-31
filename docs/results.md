# Results

All figures are from one run on the container mesh —
`prismo run --use-containers --seed u --loss-weight 1e-5 --mesh-size 0.05 --r-min 0.1 --bias-sweep-points 6`
(0.05 µm silicon elements, 0.1 µm filter radius), 192 MMA iterations in
~49 min on a laptop; `outputs/` holds the PDFs of your own runs and
`make figures` refreshes `docs/figures/` from them.

The U-shaped seed is the best of the three (`lateral`, `vertical`, `u`) under
identical settings: it reaches $V_\pi L_\pi = 0.62$ V·cm against 1.10 and
1.04 V·cm, at a comparable efficiency–loss figure of merit. The MMA optimum is
local, so the seed picks the basin.

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

From the seeded U-shaped junction, MMA raises $\Delta n_\mathrm{eff}$ at −5 V
from $1.24\times10^{-4}$ to $6.21\times10^{-4}$ (×5), i.e. $V_\pi L_\pi$
from 3.11 to **0.62 V·cm**. Dips are rejected trials of the move-limited MMA,
kept in the record on purpose. (`prismo run` also re-solves the reported
design cold — worker reset, equilibrium from near-intrinsic, bias ramp — and
flags any warm/cold discrepancy, so the headline is a property of the design,
not of the solve path.)

```{image} figures/doping_evolution.gif
:width: 720px
:align: center
```

The net doping the two solvers saw at every evaluation: red n-type, blue
p-type, white the junction. The optimizer drives the rib to the doping ceiling
and folds the U seed into a stacked junction whose interface runs across the
full rib width.

## Before / after

```{image} figures/doping_field.png
:width: 760px
:align: center
```

Left: the seed, a U-shaped junction at $|N| \approx 3\times10^{17}\,\mathrm{cm^{-3}}$
— n wrapped under and beside a p core. Right: the optimum — a p cap at the
doping ceiling over an n body, the junction running horizontally across the
whole rib width at the mode centre and curling down into the slab on the
right, while the outer slab stays at the seed where the mode does not reach.
Trading the seed's vertical junction walls for one wide horizontal interface is
what buys the fivefold $\Delta n_\mathrm{eff}$: junction area inside the mode is
the currency.

## Where the modulation happens

```{image} figures/depletion_field.png
:width: 760px
:align: center
```

Carriers swept out between 0 V and −5 V at the optimum (orange, log scale),
under the mode's $|E|$ contours. The depleted region now fills the rib
cross-section and the slab beneath it, covering the mode peak almost entirely —
against the seed, that overlap is the whole of the ×5. Doping the mode cannot
see would be loss for nothing, and the objective knows it: the outer slab is
left alone.

## The loss is watched, not ignored

```{image} figures/loss_convergence.png
:width: 520px
:align: center
```

Modal free-carrier loss $\alpha$ of the unbiased device and the
efficiency–loss figure of merit $V_\pi L_\pi\cdot\alpha$ at every iteration.
The optimizer spends loss — 2.67 to 11.9 dB/cm — wherever it pays in
$\Delta n_\mathrm{eff}$, which rises faster, so the figure of merit still
improves from 8.3 to **7.4 V·dB** (good depletion modulators sit at
10–30 V·dB). At $w = 10^{-5}$ the run travels along a near-constant
$V_\pi L_\pi\cdot\alpha$ line while $V_\pi L_\pi$ falls fivefold: the weight,
not the iteration count, is what moves the design across the trade-off.

```{image} figures/tradeoff.png
:width: 520px
:align: center
```

The same run as a path from seed to optimum in the $(\alpha, \Delta n_\mathrm{eff})$
plane against iso-$V_\pi L_\pi\cdot\alpha$ curves — the path tracks one of those
curves outwards. `--loss-weight` is what moves the optimum between them.

## The seed picks the basin

MMA finds a local optimum, so the starting topology matters. All three seeds
(`--seed lateral|vertical|u`) under identical settings, 192 iterations each:

```{list-table}
:header-rows: 1

* - seed
  - $\Delta n_\mathrm{eff}$
  - $\alpha$ [dB/cm]
  - $V_\pi L_\pi$ [V·cm]
  - $V_\pi L_\pi\cdot\alpha$ [V·dB]
* - `u`
  - $6.21\times10^{-4}$
  - 11.9
  - **0.62**
  - 7.40
* - `lateral`
  - $3.52\times10^{-4}$
  - 6.34
  - 1.10
  - 6.98
* - `vertical`
  - $3.74\times10^{-4}$
  - 8.84
  - 1.04
  - 9.16
```

The U seed wins by 1.8× on efficiency at a figure of merit within 6% of the
best, which is why it is the default and the run shown above. The three land
1.8× apart from one another on $V_\pi L_\pi$ — a spread larger than anything
the optimizer's own settings (filter radius, move limit, iteration count) move,
so a multi-start is worth more than tuning the solver. The `lateral` run is also
the one whose cold re-solve failed, leaving its number warm-path-dependent;
`u` and `vertical` re-solved cold to the digit.

## Across the operating range

```{image} figures/bias_sweep.png
:width: 760px
:align: center
```

The reported figures of merit against reverse bias, seed and optimized design
side by side — a post-run characterization (`--bias-sweep-points`), not part of
the objective, which sees only the −5 V operating point.
$\Delta n_\mathrm{eff}$ rises almost linearly to $6.21\times10^{-4}$ and stays
about ×5 the seed at every bias, so the gain is not an artefact of the one
voltage it was optimized at. The loss panel reads $\alpha$ from the carriers
**at each bias** rather than the objective's fixed 0 V value, so it falls as the
junction empties — 11.9 dB/cm unbiased to 1.6 dB/cm at −5 V — while the lightly
doped seed, already mostly depleted, barely moves.

The product follows: $V_\pi L_\pi\cdot\alpha$ improves from 4.10 V·dB at −1 V to
**0.98 V·dB at −5 V**, crossing below the seed near −2 V. Above that the seed's
lighter doping still wins on the product — the design was optimized at −5 V and
it shows. (These are bias-resolved $\alpha$ values; the 7.4 V·dB headline above
uses the objective's 0 V loss, which is the pessimistic reading.)

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
