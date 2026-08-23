#!/usr/bin/env julia

using JSON
using NPZ

include(joinpath(@__DIR__, "ct_common.jl"))
include(joinpath(@__DIR__, "ct_adjoint.jl"))

mutable struct WorkerState
    mesh_path::String
    mesh_key::String
    n_nodes::Int
    ctsys::Any
    data::Any
    cathode_breg::Int
    n_bregions::Int
    node_parents::Vector{Int}
    control::Any
    profile_key::String
    equilibrium_sol::Any
    warm_equilibrium::Any
    forward_solutions::Dict{Float64, Any}
    warm_bias_solutions::Dict{Float64, Any}
    # Silicon doping each warm biased solution was converged on, so a failed
    # direct warm start can continue by doping homotopy at fixed bias.
    warm_bias_dopings::Dict{Float64, Vector{Float64}}
end

WorkerState() = WorkerState(
    "",
    "",
    0,
    nothing,
    nothing,
    0,
    0,
    Int[],
    nothing,
    "",
    nothing,
    nothing,
    Dict{Float64, Any}(),
    Dict{Float64, Any}(),
    Dict{Float64, Vector{Float64}}(),
)

# Drop every warm solution. The mesh, system and material
# data stay: they are a deterministic function of the request, only the Newton
# starting points carry solve history. The next request on any profile then
# pays the full cold continuation -- equilibrium from near-intrinsic doping and
# the bias ramp from equilibrium -- which is what a cold re-evaluation wants.
function clear_warm_state!(state)
    state.profile_key = ""
    state.equilibrium_sol = nothing
    state.warm_equilibrium = nothing
    empty!(state.forward_solutions)
    empty!(state.warm_bias_solutions)
    empty!(state.warm_bias_dopings)
    return nothing
end

read_vector(path) = Float64.(vec(npzread(path)))

function rebuild_context!(state, doping, mesh_path, mesh_key)
    ctsys, data, cathode_breg, n_bregions, node_parents = build_ct_system(doping, mesh_path)
    state.mesh_path = mesh_path
    state.mesh_key = mesh_key
    state.n_nodes = length(doping)
    state.ctsys = ctsys
    state.data = data
    state.cathode_breg = cathode_breg
    state.n_bregions = n_bregions
    state.node_parents = node_parents
    state.control = make_solver_control()
    clear_warm_state!(state)
    return nothing
end

function ensure_profile!(state, doping, mesh_path, mesh_key, profile_key)
    if state.ctsys === nothing || state.mesh_path != mesh_path ||
       state.mesh_key != mesh_key || state.n_nodes != length(doping)
        rebuild_context!(state, doping, mesh_path, mesh_key)
    end
    if state.profile_key == profile_key
        return false
    end

    # Mark the old physics invalid before a warm solve can mutate its system.
    state.profile_key = ""
    state.equilibrium_sol = nothing
    empty!(state.forward_solutions)
    silicon_doping = doping[state.node_parents]
    # Label the stage: both the equilibrium and the biased solve surface as a
    # bare VoronoiFVM.ConvergenceError once the error string crosses the worker
    # boundary, which leaves a failure undiagnosable from the Python side.
    equilibrium_sol = try
        solve_equilibrium_with_warm_start(
            state.ctsys,
            state.data,
            silicon_doping,
            state.control,
            state.warm_equilibrium,
        )
    catch e
        error("equilibrium solve failed: $(sprint(showerror, e))")
    end
    state.profile_key = profile_key
    state.equilibrium_sol = deepcopy(equilibrium_sol)
    state.warm_equilibrium = deepcopy(equilibrium_sol)
    empty!(state.forward_solutions)
    return true
end

function forward_solution!(state, doping, mesh_path, mesh_key, profile_key, bias_voltage)
    profile_changed = ensure_profile!(state, doping, mesh_path, mesh_key, profile_key)
    if haskey(state.forward_solutions, bias_voltage)
        if abs(bias_voltage) == 0.0
            configure_equilibrium!(state.ctsys, state.data)
        else
            configure_biased_state!(
                state.ctsys,
                state.data,
                bias_voltage,
                state.cathode_breg,
                state.n_bregions,
            )
        end
        return state.forward_solutions[bias_voltage], profile_changed, false
    end

    if abs(bias_voltage) == 0.0
        configure_equilibrium!(state.ctsys, state.data)
        sol = deepcopy(state.equilibrium_sol)
        used_warm_start = !profile_changed
    else
        silicon_doping = doping[state.node_parents]
        sol = try
            solve_at_bias_with_warm_start(
                state.ctsys,
                state.control,
                deepcopy(state.equilibrium_sol),
                bias_voltage,
                state.cathode_breg,
                state.n_bregions,
                get(state.warm_bias_solutions, bias_voltage, nothing);
                data = state.data,
                doping = silicon_doping,
                warm_doping = get(state.warm_bias_dopings, bias_voltage, nothing),
            )
        catch e
            # Labelled like the equilibrium stage; a SolveBudgetExceeded keeps
            # its own message (it names the stage and the budget).
            error("biased solve at $(bias_voltage) V failed: $(sprint(showerror, e))")
        end
        used_warm_start = haskey(state.warm_bias_solutions, bias_voltage)
        state.warm_bias_dopings[bias_voltage] = copy(silicon_doping)
    end

    state.forward_solutions[bias_voltage] = deepcopy(sol)
    state.warm_bias_solutions[bias_voltage] = deepcopy(sol)
    return sol, profile_changed, used_warm_start
end

# Scatter the silicon-subgrid carrier densities back over the full mesh, in the
# cm^-3 the tesseract OutputSchema reports (ChargeTransport solves in m^-3).
function carrier_fields(sol, data, node_parents, n_nodes)
    electrons = zeros(Float64, n_nodes)
    holes = zeros(Float64, n_nodes)
    for (silicon_node, full_node) in enumerate(node_parents)
        electrons[full_node] =
            CT_DENSITY_TO_CM3 * get_density(sol, data, 1, 1; inode = silicon_node)
        holes[full_node] =
            CT_DENSITY_TO_CM3 * get_density(sol, data, 2, 1; inode = silicon_node)
    end
    return electrons, holes
end

function process_forward!(state, request)
    start_solve_budget!()
    doping = read_vector(String(request["doping_path"]))
    bias_voltage = Float64(JSON.parsefile(String(request["bias_path"]))["bias_voltage"])
    mesh_path = String(request["mesh_path"])
    mesh_key = String(request["mesh_key"])
    profile_key = String(request["profile_key"])
    output_path = String(request["output_path"])

    sol, profile_changed, used_warm_start = forward_solution!(
        state,
        doping,
        mesh_path,
        mesh_key,
        profile_key,
        bias_voltage,
    )
    electrons, holes = carrier_fields(sol, state.data, state.node_parents, length(doping))
    npzwrite(output_path, Dict("electrons" => electrons, "holes" => holes))
    return Dict(
        "ok" => true,
        "profile_changed" => profile_changed,
        "used_warm_start" => used_warm_start,
    )
end

function process_vjp!(state, request)
    start_solve_budget!()
    doping = read_vector(String(request["doping_path"]))
    bias_voltage = Float64(JSON.parsefile(String(request["bias_path"]))["bias_voltage"])
    mesh_path = String(request["mesh_path"])
    mesh_key = String(request["mesh_key"])
    profile_key = String(request["profile_key"])
    output_path = String(request["output_path"])
    state.ctsys === nothing && error("VJP requires a preceding forward solve")
    state.mesh_path == mesh_path || error("VJP mesh differs from retained forward state")
    state.mesh_key == mesh_key || error("VJP mesh configuration differs from retained forward state")
    state.profile_key == profile_key || error("VJP doping differs from retained forward state")
    haskey(state.forward_solutions, bias_voltage) || error(
        "VJP bias differs from retained forward state",
    )

    cot_n = read_vector(String(request["cotangent_electrons_path"]))
    cot_p = read_vector(String(request["cotangent_holes_path"]))
    length(cot_n) == length(doping) || error("electron cotangent length mismatch")
    length(cot_p) == length(doping) || error("hole cotangent length mismatch")
    # The cotangents are w.r.t. the cm^-3 densities ``carrier_fields`` reported;
    # the adjoint linearises the m^-3 state, so carry the same factor across.
    silicon_vjp = compute_doping_vjp(
        state.ctsys,
        state.data,
        deepcopy(state.forward_solutions[bias_voltage]),
        bias_voltage,
        state.cathode_breg,
        state.n_bregions,
        CT_DENSITY_TO_CM3 .* cot_n[state.node_parents],
        CT_DENSITY_TO_CM3 .* cot_p[state.node_parents],
    )
    vjp = zeros(Float64, length(doping))
    vjp[state.node_parents] = silicon_vjp
    npzwrite(output_path, vjp)
    return Dict("ok" => true)
end

function reply(payload)
    JSON.print(stdout, payload)
    print(stdout, "\n")
    flush(stdout)
end

function main()
    state = WorkerState()
    for line in eachline(stdin)
        try
            request = JSON.parse(line)
            operation = String(request["operation"])
            if operation == "shutdown"
                reply(Dict("ok" => true))
                break
            elseif operation == "forward"
                reply(process_forward!(state, request))
            elseif operation == "vjp"
                reply(process_vjp!(state, request))
            elseif operation == "reset"
                clear_warm_state!(state)
                reply(Dict("ok" => true))
            else
                error("unknown worker operation: $operation")
            end
        catch err
            reply(Dict("ok" => false, "error" => sprint(showerror, err)))
        end
    end
end

main()
