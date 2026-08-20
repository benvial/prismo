# Shared setup for the ChargeTransport drift-diffusion scripts.
#
# Included by worker.jl and warmup.jl. Keeping the system
# construction and the biased solve in one place means the PackageCompiler
# warmup exercises exactly the same method specializations the runtime
# scripts use (ticket 17).

using ChargeTransport
using ExtendableGrids
using Gmsh

# import (not using): ChargeTransport and VoronoiFVM export conflicting names
# (SolverControl, System, ...). All VoronoiFVM access stays qualified.
import VoronoiFVM

# ChargeTransport.jl is SI throughout: its ``constants.ε_0`` is in C/(V*m) and
# its own examples spell material data as ``1e17 / cm^3`` (= 1e23 m^-3) and
# ``1350 * cm^2 / (V*s)`` (= 0.135 m^2/(V*s)) -- the unit factors convert *into*
# metres. Every length, density and mobility below is therefore SI, and the
# gyptis-authored shared mesh (micrometres) is scaled into metres on load.
const MICROMETRES_TO_METRES = 1e-6

# PRISMO's design field is net doping in cm^-3, donor-positive: theta > 0 means
# n-type (the chargetransport tesseract InputSchema, and ``doping_from_theta``
# in pipeline.py). Two conventions separate it from what ``ParamsNodal.doping``
# wants:
#
#   * units -- SI, so cm^-3 -> m^-3 is a factor 1e6;
#   * sign  -- ChargeTransport's Poisson source is
#     ``-q*λ1*(Σ_α z_α n_α - doping)`` with ``z_n = -1, z_p = +1``, so charge
#     neutrality reads ``p - n = doping``: a positive entry is *p-type*. Its
#     ``Ex102_PIN_nodal_doping`` example confirms this, assigning ``+NDoping``
#     across the region it names ``regionAcceptor``.
#
# Both are folded into one constant, applied at the single point where doping
# is written into the system, and undone on the VJP (``ct_adjoint.jl``).
const PRISMO_DOPING_TO_CT = -1.0e6

# Carrier densities come back out of ChargeTransport in m^-3; the tesseract
# reports them in cm^-3 (OutputSchema), which is also what the Soref-Bennett
# coefficients expect.
const CT_DENSITY_TO_CM3 = 1.0e-6

# ExtendableSparse is a transitive dep (via VoronoiFVM), not in Project.toml,
# so it can only be reached through qualified access.
const ExtendableSparse = VoronoiFVM.ExtendableSparse

include(joinpath(@__DIR__, "contacts.jl"))

# Fallback 1D device (length 1 µm) used when no Gmsh mesh is supplied.
function generate_1d_mesh(n_nodes)
    L = 1e-6
    coord = collect(range(0.0, L, n_nodes))
    grid = simplexgrid(coord)
    cellmask!(grid, 0.0, L, 1)
    bfacemask!(grid, 0.0, 0.0, 1)
    bfacemask!(grid, L, L, 2)
    return grid
end

# Return dense ExtendableGrids region ids for named Gmsh physical groups.
function region_ids(mesh_path, dim)
    ids = Dict{String, Int}()
    gmsh.initialize()
    try
        gmsh.open(mesh_path)
        for (id, (_, tag)) in enumerate(gmsh.model.getPhysicalGroups(dim))
            ids[gmsh.model.getPhysicalName(dim, tag)] = id
        end
    finally
        gmsh.clear()
        gmsh.finalize()
    end
    return ids
end

edgekey(a, b) = a < b ? (a, b) : (b, a)

"""Smallest boundary-region index that is not one of the contacts.

The reconstructed silicon perimeter is a no-flux interface, so it needs a region
of its own. Filling it with a literal 1 is only safe while `contact_anode` is not
the first dim-1 physical group in the mesh -- on a mesh where it is, every
exterior face would be tagged as the anode and the whole rib would be solved as
one shorted ohmic contact. Picking the first free index instead keeps the three
regions distinct whatever order the mesh lists its physical groups in.
"""
function interface_bregion(anode_breg, cathode_breg)
    region = 1
    while region == anode_breg || region == cathode_breg
        region += 1
    end
    return region
end

"""Restrict a full optical mesh to silicon and reconstruct its exterior faces."""
function silicon_grid(full, silicon_ids, anode_breg, cathode_breg)
    selected = findall(region -> region in silicon_ids, full[CellRegions])
    isempty(selected) && error("shared mesh has no silicon cells")
    parent_cells = full[CellNodes][:, selected]
    parent_to_local = zeros(Int64, size(full[Coordinates], 2))
    node_parents = Int64[]
    for parent_node in parent_cells
        if parent_to_local[parent_node] == 0
            push!(node_parents, parent_node)
            parent_to_local[parent_node] = length(node_parents)
        end
    end
    cells = Int64[parent_to_local[parent_node] for parent_node in parent_cells]

    contact_segments = Tuple{Int64, Float64, Float64, Float64}[]
    for ibface in eachindex(full[BFaceRegions])
        bregion = full[BFaceRegions][ibface]
        if bregion == anode_breg || bregion == cathode_breg
            nodes = full[BFaceNodes][:, ibface]
            xy = full[Coordinates][:, nodes]
            push!(contact_segments, (bregion, minimum(xy[1, :]), maximum(xy[1, :]), xy[2, 1]))
        end
    end

    face_counts = Dict{Tuple{Int64, Int64}, Int64}()
    for cell in eachcol(cells)
        for (a, b) in ((cell[1], cell[2]), (cell[2], cell[3]), (cell[3], cell[1]))
            key = edgekey(a, b)
            face_counts[key] = get(face_counts, key, 0) + 1
        end
    end
    exterior = sort!([key for (key, count) in face_counts if count == 1])
    bfaces = reduce(hcat, (Int64[key[1], key[2]] for key in exterior))
    bfaceregions = fill(interface_bregion(anode_breg, cathode_breg), length(exterior))
    for (i, local_edge) in enumerate(exterior)
        parent_nodes = node_parents[[local_edge[1], local_edge[2]]]
        xy = full[Coordinates][:, parent_nodes]
        for (bregion, xmin, xmax, y) in contact_segments
            overlaps = max(minimum(xy[1, :]), xmin) <= min(maximum(xy[1, :]), xmax) + 1e-12
            if overlaps && all(abs.(xy[2, :] .- y) .< 1e-12)
                bfaceregions[i] = bregion
            end
        end
    end

    grid = simplexgrid(
        full[Coordinates][:, node_parents], cells, ones(Int64, size(cells, 2)), bfaces, bfaceregions,
    )
    grid[NodeParents] = node_parents
    return grid
end

# Build grid + ChargeTransport system for a full shared-mesh doping profile.
#
# Returns (ctsys, data, cathode_breg, n_bregions, node_parents). ``node_parents``
# maps silicon-grid nodes back to the full gyptis/Gmsh node order.
function build_ct_system(doping, mesh_path)
    n_nodes = length(doping)

    anode_breg = 1
    cathode_breg = 2

    if mesh_path != "" && isfile(mesh_path)
        # NOT simplexgrid(mesh_path): the Gmsh loader defaults to Float32
        # coordinates (and simplexgrid() does not forward a Tc kwarg),
        # which makes the VoronoiFVM system Float32 — 1e-10 Newton
        # tolerances are then unreachable and every solve fails.
        full = ExtendableGrids.simplexgrid_from_gmsh(mesh_path; Tc = Float64, Ti = Int64)
        size(full[Coordinates], 2) == n_nodes || error(
            "doping array length ($n_nodes) does not match shared mesh node count $(size(full[Coordinates], 2))",
        )
        cell_ids = region_ids(mesh_path, 2)
        bface_ids = region_ids(mesh_path, 1)
        silicon_ids = Int[
            cell_ids[name]
            for name in ("slab", "rib_silicon")
            if haskey(cell_ids, name)
        ]
        isempty(silicon_ids) && error("shared mesh has neither slab nor rib_silicon group")
        anode_breg = bface_ids["contact_anode"]
        cathode_breg = bface_ids["contact_cathode"]
        grid = silicon_grid(full, silicon_ids, anode_breg, cathode_breg)
        grid[Coordinates] .*= MICROMETRES_TO_METRES
        node_parents = Int.(grid[NodeParents])
    else
        grid = generate_1d_mesh(n_nodes)
        node_parents = collect(1:n_nodes)
    end

    grid_nnodes = size(grid[Coordinates], 2)
    silicon_doping = doping[node_parents]

    data = Data(grid, 2)
    data.modelType = Stationary

    data.bulkRecombination = set_bulk_recombination(
        iphin = 1, iphip = 2,
        bulk_recomb_Auger = false,
        bulk_recomb_radiative = false,
        bulk_recomb_SRH = false,
    )

    n_bregions = grid[NumBFaceRegions]
    if anode_breg <= n_bregions
        data.boundaryType[anode_breg] = OhmicContact
    end
    if cathode_breg <= n_bregions
        data.boundaryType[cathode_breg] = OhmicContact
    end

    constants = ChargeTransport.constants
    n_regions = grid[NumCellRegions]
    params = Params(n_regions, n_bregions, 2)

    # Silicon at 300 K, in SI base units (see MICROMETRES_TO_METRES above):
    # Nc = 2.8e19 cm^-3, Nv = 1.04e19 cm^-3, mu_n = 1350 cm^2/(V*s),
    # mu_p = 450 cm^2/(V*s).
    T = 300.0
    eps_si = 11.7 * constants.ε_0
    Nc = 2.8e25
    Nv = 1.04e25
    Eg = 1.12 * constants.q
    mu_n = 0.135
    mu_p = 0.045

    params.temperature = T
    params.chargeNumbers[1] = -1
    params.chargeNumbers[2] = 1
    for ireg in 1:n_regions
        params.dielectricConstant[ireg] = eps_si
        params.densityOfStates[1, ireg] = Nc
        params.densityOfStates[2, ireg] = Nv
        params.bandEdgeEnergy[1, ireg] = Eg
        params.bandEdgeEnergy[2, ireg] = 0.0
        params.mobility[1, ireg] = mu_n
        params.mobility[2, ireg] = mu_p
    end

    data.params = params

    paramsnodal = ParamsNodal(grid, 2)
    data.paramsnodal = paramsnodal
    set_doping!(data, silicon_doping)

    ctsys = System(grid, data, unknown_storage = :dense)
    return ctsys, data, cathode_breg, n_bregions, node_parents
end

# Update a reusable system without rebuilding its mesh/material data.
#
# ``doping`` is in PRISMO's convention (cm^-3, donor-positive). This is the only
# place doping is written into a ChargeTransport system, so it is the only place
# that needs to know about ``PRISMO_DOPING_TO_CT``.
function set_doping!(data, doping)
    for i in eachindex(doping)
        data.paramsnodal.doping[i] = PRISMO_DOPING_TO_CT * doping[i]
    end
    return nothing
end

# Put the system into its equilibrium contact configuration.
function configure_equilibrium!(ctsys, data)
    grid = ctsys.fvmsys.grid
    fvmsys = ctsys.fvmsys
    fvmsys.physics.data.calculationType = ChargeTransport.InEquilibrium
    for ibreg in grid[BFaceRegions]
        set_contact!(ctsys, ibreg, Δu = 0.0)
    end
    fvmsys.physics.data.λ1 = 1.0
    return nothing
end

# Match ChargeTransport._equilibrium_solve!'s selected parent node exactly.
function equilibrium_bpsi_parent_node(grid, ibreg)
    boundary_grid = subgrid(grid, [ibreg], boundary = true)
    parents = boundary_grid[NodeParents]
    isempty(parents) && error("missing boundary node for region $ibreg")
    return parents[1]
end

# Store the equilibrium quantities needed by a subsequent biased solve.
function store_equilibrium_contact_data!(ctsys, data, sol)
    grid = ctsys.fvmsys.grid
    params = data.params
    paramsnodal = data.paramsnodal
    ipsi = data.index_psi
    constants = data.constants

    for ibreg in grid[BFaceRegions]
        inode = equilibrium_bpsi_parent_node(grid, ibreg)
        params.bψEQ[ibreg] = sol[ipsi, inode]
    end

    for icc in data.electricCarrierList
        for ibreg in grid[BFaceRegions]
            # Same representative boundary node as the bψEQ loop above.
            # ``grid[BFaceNodes][ibreg]`` linear-indexes the (nodes-per-face x
            # n_bfaces) matrix by region number, which picks an arbitrary node on
            # a 2D mesh (it only coincides with the region's node in the 1D
            # fallback), corrupting the contact reference density.
            inode = equilibrium_bpsi_parent_node(grid, ibreg)
            Ncc = params.bDensityOfStates[icc, ibreg] +
                  paramsnodal.densityOfStates[icc, inode]
            Ecc = params.bBandEdgeEnergy[icc, ibreg] +
                  paramsnodal.bandEdgeEnergy[icc, inode]
            eta = params.chargeNumbers[icc] /
                  (constants.k_B * params.temperature / constants.q) *
                  ((sol[icc, inode] - sol[ipsi, inode]) +
                   Ecc / constants.q)
            params.bDensityEQ[icc, ibreg] = Ncc * data.F[icc](eta)
        end
    end

    data.calculationType = ChargeTransport.OutOfEquilibrium
    return nothing
end

# Equilibrium solve with doping-magnitude continuation.
#
# ChargeTransport's own embedding (equilibrium_solve!) ramps the nonlinear
# coupling λ1 from 1e-20 to 1; near λ1 ≈ 0 the Robin contact penalty
# (∝ λ1) vanishes, the potential problem is near-singular, and Newton
# either creeps without converging (n-type, |N| ≳ 1e21 cm⁻³) or assembles
# NaNs from overflowed Boltzmann factors (p-type, almost any magnitude).
#
# Instead we solve the FULL nonlinear problem (λ1 = 1) directly, but
# ramp the doping profile in factor-10 steps from a near-intrinsic
# magnitude, warm-starting each solve from the previous one. Every Newton
# solve then starts inside its basin of attraction (ticket 17).
function solve_equilibrium(ctsys, data, doping, control)
    grid = ctsys.fvmsys.grid
    fvmsys = ctsys.fvmsys

    configure_equilibrium!(ctsys, data)

    start_doping = 1e10  # near-intrinsic: converges from a zero start
    max_doping = maximum(abs.(doping); init = 0.0)
    nsteps = max(1, ceil(Int, log10(max(max_doping, start_doping) / start_doping)))
    scales = 10 .^ range(log10(start_doping / max(max_doping, start_doping)), 0.0,
                         length = nsteps + 1)

    sol = VoronoiFVM.unknowns(fvmsys, inival = 0.0)
    for s in scales
        set_doping!(data, doping .* s)
        sol = VoronoiFVM.solve(fvmsys, inival = sol, control = control)
    end

    store_equilibrium_contact_data!(ctsys, data, sol)
    return sol
end

# Try a nearby converged equilibrium as Newton's initial value. The established
# magnitude continuation remains the recovery path for failed warm starts.
function solve_equilibrium_with_warm_start(ctsys, data, doping, control, warm_start)
    warm_start === nothing && return solve_equilibrium(ctsys, data, doping, control)

    set_doping!(data, doping)
    configure_equilibrium!(ctsys, data)
    try
        sol = VoronoiFVM.solve(ctsys.fvmsys, inival = warm_start, control = control)
        store_equilibrium_contact_data!(ctsys, data, sol)
        return sol
    catch e
        if e isa VoronoiFVM.ConvergenceError || e isa VoronoiFVM.AssemblyError
            return solve_equilibrium(ctsys, data, doping, control)
        end
        rethrow()
    end
end

function make_solver_control()
    # ChargeTransport.SolverControl, not VoronoiFVM's — the two packages
    # export distinct types under the same name; the ChargeTransport solve
    # wrappers expect theirs (matches the forward path from ticket 03).
    control = ChargeTransport.SolverControl()
    control.abstol = 1e-10
    control.reltol = 1e-10
    control.maxiters = 100
    # ``max_round`` bounds VoronoiFVM's damping rounds.  Limiting it to five
    # makes the sharp P/N profiles reached by later MMA iterations fail before
    # their Newton residual can settle.  This is the package default.
    control.max_round = 1_000
    control.verbose = false
    return control
end

# Solve at an applied bias, starting from the equilibrium solution u0.
#
# A single Newton step from equilibrium fails to converge at large reverse
# bias (VoronoiFVM.ConvergenceError at -5 V, ticket 17). Ramp the cathode
# contact voltage, warm-starting each solve from the previous one. Newton's
# iteration count grows with the voltage step, so adapt: halve the step on
# ConvergenceError and regrow cautiously on success.
#
# The schedule constants are set by the hardest part of the ramp, which at
# depletion-modulator doping (|N| ~ 3e17 cm^-3) is depletion-region formation
# between 0 and -0.4 V. A 0.5 V start with a 1e-3 floor gives up there
# (AssemblyError from Boltzmann overflow at every bias past -0.5 V); starting
# at 0.1 V and allowing the step down to 1e-6 crosses it and reaches -5 V
# (ticket 06). Halving rather than quartering keeps the recovered step close
# to the largest one that still converges, so the retry count stays bounded.
const CT_BIAS_STEP_INITIAL = 0.1
const CT_BIAS_STEP_MIN = 1e-6
const CT_BIAS_MAX_FAILURES = 400  # hard bound on retries: never hang (ticket 17)

function solve_at_bias(ctsys, control, u0, bias_voltage, cathode_breg, n_bregions)
    if abs(bias_voltage) == 0.0 || cathode_breg > n_bregions
        return u0
    end

    step = CT_BIAS_STEP_INITIAL
    failures = 0
    v_applied = 0.0
    sol = u0
    while v_applied != bias_voltage
        v_next = v_applied +
                 sign(bias_voltage) * min(step, abs(bias_voltage - v_applied))
        set_contact!(ctsys, cathode_breg, Δu = v_next)
        try
            sol = solve(ctsys; inival = sol, control = control)
            v_applied = v_next
            step = min(step * 1.5, CT_BIAS_STEP_INITIAL)
        catch e
            # ConvergenceError: Newton exceeded maxiters.
            # AssemblyError: NaN in flux assembly (Boltzmann overflow during
            # a Newton overshoot). Both are retried with a smaller step.
            if (e isa VoronoiFVM.ConvergenceError || e isa VoronoiFVM.AssemblyError) &&
               failures < CT_BIAS_MAX_FAILURES && step / 2 >= CT_BIAS_STEP_MIN
                failures += 1
                step /= 2
                set_contact!(ctsys, cathode_breg, Δu = v_applied)
            else
                rethrow()
            end
        end
    end
    return sol
end

# Reuse a nearby bias-state Newton iterate when it converges immediately. This
# never weakens the continuation fallback used by ``solve_at_bias``.
function solve_at_bias_with_warm_start(
    ctsys,
    control,
    u0,
    bias_voltage,
    cathode_breg,
    n_bregions,
    warm_start,
)
    if abs(bias_voltage) == 0.0 || cathode_breg > n_bregions
        return u0
    end
    ctsys.fvmsys.physics.data.calculationType = ChargeTransport.OutOfEquilibrium
    warm_start === nothing && return solve_at_bias(
        ctsys, control, u0, bias_voltage, cathode_breg, n_bregions,
    )

    set_contact!(ctsys, cathode_breg, Δu = bias_voltage)
    try
        return solve(ctsys; inival = warm_start, control = control)
    catch e
        if e isa VoronoiFVM.ConvergenceError || e isa VoronoiFVM.AssemblyError
            set_contact!(ctsys, cathode_breg, Δu = 0.0)
            return solve_at_bias(ctsys, control, u0, bias_voltage, cathode_breg, n_bregions)
        end
        rethrow()
    end
end
