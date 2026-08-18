#!/usr/bin/env julia

using JSON
using NPZ

include(joinpath(@__DIR__, "ct_common.jl"))
include(joinpath(@__DIR__, "ct_adjoint.jl"))

function parse_adjoint_args()
    doping_path = ""
    cot_n_path = ""
    cot_p_path = ""
    bias_path = ""
    output_path = ""
    mesh_path = ""

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--doping" && i + 1 <= length(ARGS)
            doping_path = ARGS[i + 1]; i += 2
        elseif ARGS[i] == "--cotangent_n" && i + 1 <= length(ARGS)
            cot_n_path = ARGS[i + 1]; i += 2
        elseif ARGS[i] == "--cotangent_p" && i + 1 <= length(ARGS)
            cot_p_path = ARGS[i + 1]; i += 2
        elseif ARGS[i] == "--bias" && i + 1 <= length(ARGS)
            bias_path = ARGS[i + 1]; i += 2
        elseif ARGS[i] == "--output" && i + 1 <= length(ARGS)
            output_path = ARGS[i + 1]; i += 2
        elseif ARGS[i] == "--mesh" && i + 1 <= length(ARGS)
            mesh_path = ARGS[i + 1]; i += 2
        else
            i += 1
        end
    end
    return doping_path, cot_n_path, cot_p_path, bias_path, output_path, mesh_path
end

read_vector(path) = Float64.(vec(npzread(path)))

function main()
    doping_path, cot_n_path, cot_p_path, bias_path, output_path, mesh_path =
        parse_adjoint_args()
    doping = read_vector(doping_path)
    cot_n = read_vector(cot_n_path)
    cot_p = read_vector(cot_p_path)
    length(cot_n) == length(doping) || error("electron cotangent length mismatch")
    length(cot_p) == length(doping) || error("hole cotangent length mismatch")
    bias_voltage = Float64(JSON.parsefile(bias_path)["bias_voltage"])

    ctsys, data, cathode_breg, n_bregions = build_ct_system(doping, mesh_path)
    control = make_solver_control()
    equilibrium_sol = solve_equilibrium(ctsys, data, doping, control)
    sol = solve_at_bias(
        ctsys,
        control,
        equilibrium_sol,
        bias_voltage,
        cathode_breg,
        n_bregions,
    )
    vjp = compute_doping_vjp(
        ctsys,
        data,
        sol,
        equilibrium_sol,
        bias_voltage,
        cathode_breg,
        n_bregions,
        cot_n,
        cot_p,
    )
    npzwrite(output_path, vjp)
end

main()
