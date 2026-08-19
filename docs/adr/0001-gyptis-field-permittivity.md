# 0001 — gyptis consumes a permittivity field on the design region

Status: Accepted (2026-08-19)

## Context

`CONTEXT.md` specifies gyptis' input as the permittivity **field ε(x)** on the
shared mesh. The implementation had drifted: the coupling collapsed the spatial
permittivity perturbation Δε(x) to a single scalar per subdomain (a `jnp.mean`
mapped onto one gyptis material domain), and the gyptis eigenmode solve modelled
the guide as horizontal layers with one scalar permittivity each.

The consequence was fatal for topology optimization: the gyptis-side gradient
with respect to the nodal perturbation was **uniform**, so any topology that
redistributed carriers without changing the *mean* Δε produced zero gyptis
gradient — defeating the entire purpose of spatial design.

## Decision

gyptis now consumes a true 2D permittivity field on an embedded design region,
with a field-valued Hellmann–Feynman adjoint. The per-layer scalar
approximation is retired.

- **Embedded design region.** A DG0 cell mask over the interior of the silicon
  core layer, inset from the layer's x-extent and y-edges so it provably never
  touches a PML (gyptis matches PML tensors through each layer's ε, so a varying
  ε touching a PML would break the matching). No gmsh geometry surgery is
  needed. The `LayeredBoxPML2D` geometry is kept.
- **Field injection.** The design permittivity is a scalar DG0 `dolfin.Function`
  written over `formulation._epsilon["core"]`, rebuilt from the scalar
  background first so the PMLs stay matched; non-design core cells keep the
  background scalar. Surrounding oxide/cladding/substrate stay constant.
- **Field-valued adjoint.** The hand-rolled Hellmann–Feynman adjoint is kept
  (pyadjoint does not tape the SLEPc eigensolve) and generalized to a field.
  Forward and adjoint share **one** two-sided SLEPc solve with explicit
  mode-tracking; the per-design-cell sensitivity is assembled in a single pass
  from the recovered left/right eigenvectors, with no per-cell Python loop.
- **Mesh transfer.** The gyptis design mesh does not coincide with the shared
  charge-transport mesh, so the nodal perturbation is carried onto the design
  cells by point-location (barycentric) interpolation — a static, silicon-only,
  partition-of-unity operator (`prismo.mesh_transfer`).
- **JAX driver unchanged.** The outer optimization driver (density filter +
  projection) stays in JAX; `gyptis.optimize` is not adopted.

## Consequences

- A topology change that redistributes carriers at fixed mean now moves neff and
  produces a spatially non-uniform design gradient.
- The refined-core forward lands on the fundamental **guided** mode
  (n_clad < neff < n_core) rather than the coarse-mesh leaky mode; the single-
  pass field VJP matches central finite differences to < 1e-6 relative error.
- The gyptis input schema carries `design_epsilon` (the differentiated field)
  plus constant `core_epsilon` / `clad_epsilon` / `substrate_epsilon`. The
  background solve receives a uniform field and contributes an exact zero design
  gradient (its ρ-independence is preserved).

Ref: `.scratch/gyptis-field-permittivity/` (spec + tickets 01–06),
`scripts/prototype_gyptis_eigen_adjoint.py`.
