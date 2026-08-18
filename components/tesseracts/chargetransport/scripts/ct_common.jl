# Shared setup for the ChargeTransport drift-diffusion scripts.
#
# Included by forward.jl, adjoint.jl and warmup.jl. Keeping the system
# construction and the biased solve in one place means the PackageCompiler
# warmup exercises exactly the same method specializations the runtime
# scripts use (ticket 17).

using ChargeTransport
using ExtendableGrids
using Gmsh

# import (not using): ChargeTransport and VoronoiFVM export conflicting names
# (SolverControl, System, ...). All VoronoiFVM access stays qualified.
import VoronoiFVM

# ExtendableSparse is a transitive dep (via VoronoiFVM), not in Project.toml,
# so it can only be reached through qualified access.
const ExtendableSparse = VoronoiFVM.ExtendableSparse

include(joinpath(@__DIR__, "contacts.jl"))

# Fallback 1D device (length 1 µm) used when no Gmsh mesh is supplied.
function generate_1d_mesh(n_nodes)
    L = 1e-4
    coord = collect(range(0.0, L, n_nodes))
    grid = simplexgrid(coord)
    cellmask!(grid, 0.0, L, 1)
    bfacemask!(grid, 0.0, 0.0, 1)
    bfacemask!(grid, L, L, 2)
    return grid
end

# Build grid + ChargeTransport system for a doping profile.
#
# Returns (ctsys, data, cathode_breg, n_bregions). Silicon parameters at
# 300 K, no bulk recombination, Ohmic contacts on anode/cathode boundary
# regions (from the Gmsh physical groups when a mesh file is given).
function build_ct_system(doping, mesh_path)
    n_nodes = length(doping)

    anode_breg = 1
    cathode_breg = 2

    if mesh_path != "" && isfile(mesh_path)
        # NOT simplexgrid(mesh_path): the Gmsh loader defaults to Float32
        # coordinates (and simplexgrid() does not forward a Tc kwarg),
        # which makes the VoronoiFVM system Float32 — 1e-10 Newton
        # tolerances are then unreachable and every solve fails.
        grid = ExtendableGrids.simplexgrid_from_gmsh(mesh_path; Tc = Float64)
        contacts = get_breking_contacts(mesh_path)
        if haskey(contacts, :anode)
            anode_breg = contacts[:anode]
        end
        if haskey(contacts, :cathode)
            cathode_breg = contacts[:cathode]
        end
    else
        grid = generate_1d_mesh(n_nodes)
    end

    grid_nnodes = size(grid[Coordinates], 2)
    if grid_nnodes != n_nodes
        error("doping array length ($n_nodes) does not match mesh node count ($grid_nnodes)")
    end

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

    T = 300.0
    eps_si = 11.7 * constants.ε_0
    Nc = 2.8e19
    Nv = 1.04e19
    Eg = 1.12 * constants.q
    mu_n = 1350.0
    mu_p = 450.0

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
    for i in 1:grid_nnodes
        paramsnodal.doping[i] = doping[i]
    end
    data.paramsnodal = paramsnodal

    ctsys = System(grid, data, unknown_storage = :dense)
    return ctsys, data, cathode_breg, n_bregions
end

# Update a reusable system without rebuilding its mesh/material data.
function set_doping!(data, doping)
    for i in eachindex(doping)
        data.paramsnodal.doping[i] = doping[i]
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

    bnode = grid[BFaceNodes]
    for icc in data.electricCarrierList
        for ibreg in grid[BFaceRegions]
            inode = bnode[ibreg]
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
        for i in 1:length(doping)
            data.paramsnodal.doping[i] = doping[i] * s
        end
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
    control.maxiters = 50
    control.max_round = 5
    control.verbose = false
    return control
end

# Solve at an applied bias, starting from the equilibrium solution u0.
#
# A single Newton step from equilibrium fails to converge at large reverse
# bias (VoronoiFVM.ConvergenceError at -5 V, ticket 17). Ramp the cathode
# contact voltage, warm-starting each solve from the previous one. Newton's
# iteration count grows with the voltage step, so adapt: start with 0.5 V
# steps, quarter the step on ConvergenceError, regrow cautiously on success.
function solve_at_bias(ctsys, control, u0, bias_voltage, cathode_breg, n_bregions)
    if abs(bias_voltage) == 0.0 || cathode_breg > n_bregions
        return u0
    end

    step = 0.5
    min_step = 1e-3
    max_failures = 100  # hard bound on retries: never hang (ticket 17)
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
            step = min(step * 2, 0.5)
        catch e
            # ConvergenceError: Newton exceeded maxiters.
            # AssemblyError: NaN in flux assembly (Boltzmann overflow during
            # a Newton overshoot). Both are retried with a smaller step.
            if (e isa VoronoiFVM.ConvergenceError || e isa VoronoiFVM.AssemblyError) &&
               failures < max_failures && step / 4 >= min_step
                failures += 1
                step /= 4
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
