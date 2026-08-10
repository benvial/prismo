#!/usr/bin/env julia

using NPZ
using JSON

include(joinpath(@__DIR__, "ct_common.jl"))

function parse_args()
    doping_path = ""
    bias_path = ""
    output_path = ""
    mesh_path = ""

    args = ARGS
    i = 1
    while i <= length(args)
        if args[i] == "--doping" && i + 1 <= length(args)
            doping_path = args[i+1]; i += 2
        elseif args[i] == "--bias" && i + 1 <= length(args)
            bias_path = args[i+1]; i += 2
        elseif args[i] == "--output" && i + 1 <= length(args)
            output_path = args[i+1]; i += 2
        elseif args[i] == "--mesh" && i + 1 <= length(args)
            mesh_path = args[i+1]; i += 2
        else
            i += 1
        end
    end

    return doping_path, bias_path, output_path, mesh_path
end

function main()
    doping_path, bias_path, output_path, mesh_path = parse_args()

    doping_raw = npzread(doping_path)
    if ndims(doping_raw) > 1
        doping_raw = vec(doping_raw)
    end
    doping = Float64.(doping_raw)
    n_nodes = length(doping)

    bias = JSON.parsefile(bias_path)
    bias_voltage = Float64(bias["bias_voltage"])

    ctsys, data, cathode_breg, n_bregions = build_ct_system(doping, mesh_path)

    control = make_solver_control()

    u0 = solve_equilibrium(ctsys, data, doping, control)
    sol = solve_at_bias(ctsys, control, u0, bias_voltage, cathode_breg, n_bregions)

    electrons = zeros(Float64, n_nodes)
    holes = zeros(Float64, n_nodes)
    for inode in 1:n_nodes
        electrons[inode] = get_density(sol, data, 1, 1; inode = inode)
        holes[inode] = get_density(sol, data, 2, 1; inode = inode)
    end

    npzwrite(output_path, Dict("electrons" => electrons, "holes" => holes))
end

main()
