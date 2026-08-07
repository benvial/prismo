#!/usr/bin/env julia

using ChargeTransport
using ExtendableGrids
using NPZ
using JSON
using Gmsh

include(joinpath(@__DIR__, "contacts.jl"))

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

function generate_1d_mesh(n_nodes)
    L = 1e-4
    coord = reshape(collect(range(0.0, L, n_nodes)), 1, :)
    grid = simplexgrid(coord)
    cellmask!(grid, 0.0, L, 1)
    bfacemask!(grid, 0.0, 0.0, 1)
    bfacemask!(grid, L, L, 2)
    return grid
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

    anode_breg = 1
    cathode_breg = 2

    if mesh_path != "" && isfile(mesh_path)
        grid = simplexgrid(mesh_path)
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

    n_bregions = grid[NumBFaceRegions]
    if anode_breg <= n_bregions
        data.boundaryType[anode_breg] = OhmicContact
    end
    if cathode_breg <= n_bregions
        data.boundaryType[cathode_breg] = OhmicContact
    end
    # Remaining bregions keep their default (homogeneous Neumann / insulating).

    n_regions = grid[NumCellRegions]
    params = Params(n_regions, n_bregions, 2)

    eps_si = 11.7
    Nc = 2.8e19
    Nv = 1.04e19
    Eg = 1.12
    mu_n = 1350.0
    mu_p = 450.0

    params.chargeNumbers[1] = -1
    params.chargeNumbers[2] = 1
    for ireg in 1:n_regions
        params.dielectricConstant[ireg] = eps_si
        params.densityOfStates[1, ireg] = Nc
        params.densityOfStates[2, ireg] = Nv
        params.bandEdgeEnergy[1, ireg] = 0.0
        params.bandEdgeEnergy[2, ireg] = -Eg
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

    control = SolverControl()
    control.abstol = 1e-10
    control.reltol = 1e-10
    control.maxiters = 50
    control.verbose = false

    u0 = equilibrium_solve!(ctsys, control = control)

    if abs(bias_voltage) > 0.0 && cathode_breg <= n_bregions
        set_contact!(ctsys, cathode_breg, Δu = bias_voltage)
        sol = solve(ctsys; inival = u0, control = control)
    else
        sol = u0
    end

    electrons = zeros(Float64, n_nodes)
    holes = zeros(Float64, n_nodes)
    for inode in 1:n_nodes
        electrons[inode] = get_density(sol, data, 1, 1; inode = inode)
        holes[inode] = get_density(sol, data, 2, 1; inode = inode)
    end

    npzwrite(output_path, Dict("electrons" => electrons, "holes" => holes))
end

main()
