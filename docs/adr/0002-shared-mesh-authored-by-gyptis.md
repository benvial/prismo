# 0002 — One shared mesh, authored by the gyptis Tesseract

Status: Accepted (2026-08-21)

Supersedes the "Mesh transfer" bullet of ADR 0001.

## Context

The two solvers meshed the same device twice. gyptis built a `LayeredBoxPML2D`
optical domain (oxide / silicon / oxide layers plus a PML frame) inside its own
Tesseract; the app separately built a rib-waveguide device mesh
(`prismo.waveguide_mesh`) and handed it to ChargeTransport. Nothing forced the
two discretizations to agree, so `prismo.mesh_transfer` carried nodal carrier
fields onto the gyptis design cells by point location and barycentric
interpolation, and the app maintained a node-ordering reconciliation between the
gmsh node set and what Julia's grid reader produced.

Three problems followed:

- **The interpolation was an unverifiable seam.** Every gradient that crossed it
  crossed an operator with no exact inverse and no natural test oracle.
- **Node-ordering drift was silent.** The same `.msh` file read by Python gmsh
  and by `simplexgrid_from_gmsh` could disagree on node count and order; a
  mismatch mirrored or scrambled the doping field rather than raising.
- **The layered optical model had no rib.** Without the rib geometry, the
  eigensolve tracked a *leaky* mode instead of the fundamental guided one, so
  the reported neff was not the device's neff.

## Decision

One gmsh geometry and one mesh, authored by the gyptis Tesseract and consumed by
both solvers.

- **`RibBoxPML2D(LayeredBoxPML2D)`** extends the gyptis geometry to emit the
  full SOI cross-section: oxide substrate, silicon slab, a rib band that is
  oxide either side of an embedded 500 nm × 220 nm silicon rib, oxide clad, a
  matched PML frame, two Ohmic contact lines on the slab shoulders, and
  `silicon` / `oxide` physical groups.
- **A `write_mesh` op** publishes that mesh (mesh text plus the per-design-cell
  vertices) so the app can author the shared mesh through gyptis instead of
  building its own.
- **gyptis solves the whole domain**; only the rib interior's DG0 cells are
  modulated.
- **ChargeTransport is restricted to the silicon subdomain.** Its Tesseract
  extracts the silicon grid from the shared mesh, gathers full-mesh doping onto
  the silicon nodes, and scatters the silicon carriers and gradient back onto
  the full node set.
- **Mesh transfer collapses to an exact restriction.** Every design cell is a
  triangle of the shared mesh, so the operator is weight 1/3 on each of its
  three vertex nodes and zero elsewhere — no point location, no interpolation
  error.

## Consequences

- The transfer operator is exact and its two invariants (silicon-only support,
  partition of unity) hold by construction rather than by tolerance.
- The node-ordering reconciliation is retired: there is one node set.
- The eigensolve lands on the fundamental **guided** mode (validated in the
  gyptis container at neff = 3.168 with 196 design cells).
- **Trade-off — the optical solver owns the device mesh.** Refinement is driven
  by what the eigensolve needs, not by what the drift-diffusion solve needs, and
  the PML-matching constraint now sits upstream of the charge transport. Any
  future device-side refinement request has to be satisfied inside a geometry
  whose primary obligation is optical.
- **Trade-off — a build-order coupling.** The app cannot mesh without a running
  gyptis Tesseract; a container run that lacks it raises rather than falling
  back. `prismo.waveguide_mesh` survives only as the non-container local path.
- **Cost — one surprise the layered model had hidden.** Tagging contact lines
  leaves orphan DOFs on the embedded edges, whose zero rows zero-pivot the
  eigensolve. They are pinned with a unit diagonal, placing their eigenvalue far
  outside the guided window. The earlier contact-free spike never hit this.

Ref: `.scratch/rethink-physics-optimization/` (spec, tickets 01 and 05).
