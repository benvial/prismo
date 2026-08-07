# 12 — DEVSIM 2D drift-diffusion upgrade

**What to build:** Upgrade the DEVSIM tesseract from 1D PN junction to 2D drift-diffusion on the shared Gmsh mesh. `apply()` dispatches on input array rank: 1D array → old `_build_1d_pn_junction` (test path), 2D field → new `_build_2d_pn_junction` that imports the `.msh` via `devsim.create_gmsh_mesh`, sets doping per-node, applies lateral contacts on slab shoulders, and solves drift-diffusion at the requested bias. VJP extracts the 2D Newton Jacobian (3N×3N, N = 2D nodes) and chains identically to 1D path.

**Blocked by:** 10

**Status:** resolved

- [x] `InputSchema` gains optional `mesh_ref: MeshRef` field; when absent, 1D fallback
- [x] 2D codepath: `devsim.create_gmsh_mesh` imports `.msh`, `set_node_values("NetDoping", ...)` per-node on 2D mesh, `CreateSiliconDriftDiffusion` (simple_physics only), solves at given bias
- [x] Contacts identified from `.msh` physical groups (`_read_mesh_physical_group_nodes` via gmsh, boundary-node intersection with `contact_anode`/`contact_cathode` physical groups). Fallback: boundary-edge detection + x < 0 split when gmsh unavailable.
- [x] 1D codepath preserved behind `mesh_ref` dispatch; seam tests use 1D, pipeline uses 2D
- [x] VJP on 2D Jacobian: extract `get_matrix_and_rhs(format="csc")` → adjoint solve → chain to doping. Verified with 3-node triangle smoke test (`test_2d_vjp_on_minimal_triangle_mesh`)
- [x] Seam tests (19) all pass (1D path)
- [x] Integration tests: `test_2d_uniform_doping_matches_1d`, `test_2d_vjp_matches_finite_difference`, `test_2d_apply_with_real_mesh` (all skip without devsim+gmsh); plus 3 new seam tests for 2D contact detection
