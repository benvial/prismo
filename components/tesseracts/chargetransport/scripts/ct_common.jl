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

# Shockley-Read-Hall parameters (ticket 17): mid-gap traps, 100 ns lifetime for
# both carriers, trap density of order n_i. Applied per region in
# ``build_ct_system``, which also records why SRH is on.
const CT_SRH_LIFETIME_S = 1.0e-7
const CT_SRH_TRAP_DENSITY_M3 = 1.0e16

# Wall-clock budget for one worker request (ticket 18). The continuation loops
# below (doping magnitude, bias ramp, doping homotopy) are each bounded in
# retries, but a hard design can still chain hundreds of Newton solves, and the
# Python side used to kill the whole Julia process on its own 25 s timeout --
# dropping every warm solution and paying a cold sysimage start plus a cold
# equilibrium continuation on the next request. The budget lives here instead:
# every loop checks it and the solve returns a descriptive error, so a hard
# design costs the optimizer one evaluation and the worker survives. The Python
# request timeout (``PRISMO_CT_JULIA_TIMEOUT_S`` in tesseract_api.py) is only a
# backstop for a truly hung process and must stay well above this.
const CT_SOLVE_BUDGET_S = parse(Float64, get(ENV, "PRISMO_CT_SOLVE_BUDGET_S", "120"))

struct SolveBudgetExceeded <: Exception
    elapsed::Float64
    budget::Float64
    stage::String
end

function Base.showerror(io::IO, e::SolveBudgetExceeded)
    print(
        io,
        "SolveBudgetExceeded: $(e.stage) exceeded the $(e.budget) s wall-clock " *
        "solve budget (PRISMO_CT_SOLVE_BUDGET_S) after $(round(e.elapsed; digits = 1)) s",
    )
end

const _solve_deadline = Ref(Inf)
const _solve_started = Ref(0.0)

"""Start the wall-clock budget for the request being served."""
function start_solve_budget!(budget_s = CT_SOLVE_BUDGET_S)
    _solve_started[] = time()
    _solve_deadline[] = _solve_started[] + budget_s
    return nothing
end

"""Throw ``SolveBudgetExceeded`` once the request has used up its budget."""
function check_solve_budget(stage::String)
    now = time()
    if now > _solve_deadline[]
        throw(SolveBudgetExceeded(now - _solve_started[], _solve_deadline[] - _solve_started[], stage))
    end
    return nothing
end

# The continuation loops retry on these and only these: Newton exceeded its
# iteration budget, or assembled a NaN from a Boltzmann overflow during an
# overshoot. Anything else (a budget exhaustion included) propagates.
recoverable_solve_error(e) = e isa VoronoiFVM.ConvergenceError || e isa VoronoiFVM.AssemblyError

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
            # Containment, not overlap: the shared mesh cuts the silicon bands at
            # the contact footprints (ticket 14), so a contact face coincides with
            # a contact segment exactly. Accepting any face that merely *touches*
            # one would also swallow the neighbouring shoulder face, silently
            # widening the contact to whatever the coarse slab meshing produced.
            contained = minimum(xy[1, :]) >= xmin - 1e-12 &&
                        maximum(xy[1, :]) <= xmax + 1e-12
            if contained && all(abs.(xy[2, :] .- y) .< 1e-12)
                bfaceregions[i] = bregion
            end
        end
    end

    grid = simplexgrid(
        full[Coordinates][:, node_parents], cells, ones(Int64, size(cells, 2)), bfaces, bfaceregions,
    )
    grid[NodeParents] = node_parents
    check_contacted_device(grid, cells, bfaceregions, anode_breg, cathode_breg)
    return grid
end

"""Fail loudly unless the silicon subgrid is one region reachable from both contacts.

A drift-diffusion solve on a silicon domain that falls into disconnected pieces
still converges, and converges quietly: any piece carrying no contact is a pure
Neumann problem whose quasi-Fermi levels are fixed only up to a free constant,
so it never responds to the applied bias and Newton settles on whichever member
of that family it drifts into. That is exactly how a non-conformal slab/rib
interface in the shared mesh left the design region floating, with the composed
objective reading a bias-independent gauge instead of depletion (ticket 14).
Nothing downstream can detect it, so the invariant is asserted here.
"""
function check_contacted_device(grid, cells, bfaceregions, anode_breg, cathode_breg)
    n_nodes = size(grid[Coordinates], 2)
    component = zeros(Int, n_nodes)
    neighbours = [Int[] for _ in 1:n_nodes]
    for cell in eachcol(cells)
        for (a, b) in ((cell[1], cell[2]), (cell[2], cell[3]), (cell[3], cell[1]))
            push!(neighbours[a], b)
            push!(neighbours[b], a)
        end
    end
    n_components = 0
    for start in 1:n_nodes
        component[start] == 0 || continue
        n_components += 1
        stack = [start]
        component[start] = n_components
        while !isempty(stack)
            node = pop!(stack)
            for other in neighbours[node]
                if component[other] == 0
                    component[other] = n_components
                    push!(stack, other)
                end
            end
        end
    end
    n_components == 1 || error(
        "silicon subgrid is not connected ($n_components pieces): the shared " *
        "mesh does not conform across the silicon material interfaces, so part " *
        "of the device cannot see the contacts",
    )
    for (name, bregion) in (("anode", anode_breg), ("cathode", cathode_breg))
        count(==(bregion), bfaceregions) > 0 || error(
            "silicon subgrid has no $name contact face; the contact lines in the " *
            "shared mesh do not lie on the silicon boundary",
        )
    end
    return nothing
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

    # Shockley-Read-Hall recombination is ON; Auger and radiative stay off.
    #
    # Not for the seeded junction: there the reverse-bias carrier profile is set
    # by the electrostatics and the contacts, and SRH was measured to leave the
    # whole 0 -> -5 V ramp unchanged to five digits (ticket 14). It is on because
    # the free-form designs the optimizer proposes after ~10 MMA iterations
    # (rail-level doping, a dozen junction sign-flips, floating p-pockets inside
    # the rib) make the generation-free drift-diffusion steady state NON-UNIQUE:
    # the same doping at the same -5 V solved to depletion on a cold ramp and to
    # injection (max |dpsi| = 5.4 V) on a warm start from the neighbouring
    # design, and the composed objective read two different values for one
    # theta. Thermal generation through mid-gap traps pins the minority
    # quasi-Fermi level in depleted and floating regions, which removes the
    # spurious branch: the failing design then solves on the first default ramp
    # and a warm start that used to land on the wrong branch fails Newton and
    # falls through to the ramp instead (ticket 17, diagnosis 2026-08-22).
    data.bulkRecombination = set_bulk_recombination(
        iphin = 1, iphip = 2,
        bulk_recomb_Auger = false,
        bulk_recomb_radiative = false,
        bulk_recomb_SRH = true,
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
        # SRH through mid-gap traps: tau_n = tau_p = 100 ns (a clean silicon
        # lifetime), trap density ~ n_i (1e10 cm^-3 = 1e16 m^-3) so the
        # recombination rate R = (np - n_i^2) / (tau_p (n + n_t) + tau_n (p + p_t))
        # saturates at the textbook mid-gap form. Same in every silicon region;
        # see the bulkRecombination note above for why it is on at all.
        for icc in 1:2
            params.recombinationSRHLifetime[icc, ireg] = CT_SRH_LIFETIME_S
            params.recombinationSRHTrapDensity[icc, ireg] = CT_SRH_TRAP_DENSITY_M3
        end
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
# ramp the doping profile from a near-intrinsic magnitude, warm-starting each
# solve from the previous one. Every Newton solve then starts inside its basin
# of attraction (ticket 17). The step is adaptive: a decade at a time is only
# the nominal schedule, and it is halved wherever Newton cannot cross.

# Smallest magnitude-continuation step, in decades of doping. The ramp is in
# log10 space, so this is the analogue of ``CT_BIAS_STEP_MIN`` for the bias ramp.
const CT_DOPING_STEP_MIN = 1e-4
const CT_DOPING_MAX_FAILURES = 400  # hard bound on retries: never hang

function solve_equilibrium(ctsys, data, doping, control)
    fvmsys = ctsys.fvmsys

    configure_equilibrium!(ctsys, data)

    start_doping = 1e10  # near-intrinsic: converges from a zero start
    max_doping = maximum(abs.(doping); init = 0.0)
    # Ramp the doping magnitude in log10 space from t0 (near-intrinsic) to 0
    # (the requested profile). A fixed one-decade step is only the *nominal*
    # schedule: a free-form design the optimizer proposes can make some decade
    # too hard for Newton, and an unguarded ramp then throws ConvergenceError
    # out of the whole forward solve. Halve the step on failure and regrow it on
    # success, mirroring the bias ramp in ``solve_at_bias``.
    t0 = log10(start_doping / max(max_doping, start_doping))
    step = t0 < 0.0 ? -t0 / max(1, ceil(Int, -t0)) : 1.0

    sol = VoronoiFVM.unknowns(fvmsys, inival = 0.0)
    set_doping!(data, doping .* 10.0^t0)
    sol = VoronoiFVM.solve(fvmsys, inival = sol, control = control)

    t = t0
    failures = 0
    while t < 0.0
        check_solve_budget("equilibrium doping continuation")
        t_next = min(t + step, 0.0)
        set_doping!(data, doping .* 10.0^t_next)
        try
            sol = VoronoiFVM.solve(fvmsys, inival = sol, control = control)
            t = t_next
            step = min(step * 1.5, 1.0)
        catch e
            if recoverable_solve_error(e) &&
               failures < CT_DOPING_MAX_FAILURES && step / 2 >= CT_DOPING_STEP_MIN
                failures += 1
                step /= 2
                set_doping!(data, doping .* 10.0^t)
            else
                rethrow()
            end
        end
    end

    set_doping!(data, doping)
    store_equilibrium_contact_data!(ctsys, data, sol)
    return sol
end

# Try a nearby converged equilibrium as Newton's initial value. The established
# magnitude continuation remains the recovery path for failed warm starts.
function solve_equilibrium_with_warm_start(ctsys, data, doping, control, warm_start)
    warm_start === nothing && return solve_equilibrium(ctsys, data, doping, control)

    set_doping!(data, doping)
    configure_equilibrium!(ctsys, data)
    check_solve_budget("equilibrium warm start")
    try
        sol = VoronoiFVM.solve(ctsys.fvmsys, inival = warm_start, control = control)
        store_equilibrium_contact_data!(ctsys, data, sol)
        return sol
    catch e
        if recoverable_solve_error(e)
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
        check_solve_budget("bias ramp")
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
            if recoverable_solve_error(e) &&
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

# Doping homotopy at fixed bias (ticket 18): continue from the last converged
# biased state at ``warm_doping`` to the requested ``doping`` along
# ``d(t) = warm_doping + t (doping - warm_doping)``, warm-starting each step.
# This is the natural continuation for an optimizer that moves the design a
# little per iteration: the bias stays where the warm solution already is, and
# only the doping moves. The cold bias ramp from equilibrium stays the recovery
# behind it.
#
# The system's equilibrium contact data (``bψEQ``, ``bDensityEQ``) belongs to
# ``doping``, so the intermediate states are warm starts, not physical
# solutions; the final step solves the requested system exactly.
const CT_HOMOTOPY_STEP_MIN = 1e-3
const CT_HOMOTOPY_MAX_FAILURES = 100

function solve_at_bias_by_doping_homotopy(
    ctsys, data, control, warm_sol, warm_doping, doping, bias_voltage, cathode_breg,
)
    length(warm_doping) == length(doping) || error("homotopy doping length mismatch")
    ctsys.fvmsys.physics.data.calculationType = ChargeTransport.OutOfEquilibrium
    set_contact!(ctsys, cathode_breg, Δu = bias_voltage)
    t = 0.0
    step = 1.0
    failures = 0
    sol = warm_sol
    try
        while t < 1.0
            check_solve_budget("doping homotopy")
            t_next = min(t + step, 1.0)
            set_doping!(data, warm_doping .+ t_next .* (doping .- warm_doping))
            try
                sol = solve(ctsys; inival = sol, control = control)
                t = t_next
                step = min(step * 1.5, 1.0)
            catch e
                if recoverable_solve_error(e) &&
                   failures < CT_HOMOTOPY_MAX_FAILURES && step / 2 >= CT_HOMOTOPY_STEP_MIN
                    failures += 1
                    step /= 2
                else
                    rethrow()
                end
            end
        end
    finally
        # Whatever happened, leave the system describing the requested doping.
        set_doping!(data, doping)
    end
    return sol
end

# Biased solve with the fallback chain of ticket 18:
#
#   1. direct Newton from the nearby biased warm solution (``warm_start``);
#   2. doping homotopy at fixed bias from that solution (needs ``warm_doping``,
#      the silicon doping the warm solution was converged on);
#   3. the cold bias ramp from the equilibrium solution ``u0``.
#
# Each stage is tried only when the previous one fails with a recoverable
# Newton error; the ramp's own continuation never weakens. A budget exhaustion
# propagates from whichever stage it hits.
function solve_at_bias_with_warm_start(
    ctsys,
    control,
    u0,
    bias_voltage,
    cathode_breg,
    n_bregions,
    warm_start;
    data = nothing,
    doping = nothing,
    warm_doping = nothing,
)
    if abs(bias_voltage) == 0.0 || cathode_breg > n_bregions
        return u0
    end
    ctsys.fvmsys.physics.data.calculationType = ChargeTransport.OutOfEquilibrium
    warm_start === nothing && return solve_at_bias(
        ctsys, control, u0, bias_voltage, cathode_breg, n_bregions,
    )

    check_solve_budget("biased warm start")
    set_contact!(ctsys, cathode_breg, Δu = bias_voltage)
    try
        return solve(ctsys; inival = warm_start, control = control)
    catch e
        recoverable_solve_error(e) || rethrow()
    end

    if data !== nothing && doping !== nothing && warm_doping !== nothing
        try
            return solve_at_bias_by_doping_homotopy(
                ctsys, data, control, warm_start, warm_doping, doping, bias_voltage,
                cathode_breg,
            )
        catch e
            recoverable_solve_error(e) || rethrow()
        end
    end

    set_contact!(ctsys, cathode_breg, Δu = 0.0)
    return solve_at_bias(ctsys, control, u0, bias_voltage, cathode_breg, n_bregions)
end
