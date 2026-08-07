#!/usr/bin/env julia

using ChargeTransport
using ExtendableGrids
using VoronoiFVM
using NPZ
using JSON
using LinearAlgebra
using SparseArrays
using ExtendableSparseArrays
using Gmsh

include(joinpath(@__DIR__, "contacts.jl"))

const SPEC_E = 1
const SPEC_H = 2
const DOF_PSI = 3

function parse_adjoint_args()
    doping_path = ""
    cot_n_path = ""
    cot_p_path = ""
    bias_path = ""
    output_path = ""
    mesh_path = ""

    args = ARGS
    i = 1
    while i <= length(args)
        if args[i] == "--doping" && i + 1 <= length(args)
            doping_path = args[i+1]; i += 2
        elseif args[i] == "--cotangent_n" && i + 1 <= length(args)
            cot_n_path = args[i+1]; i += 2
        elseif args[i] == "--cotangent_p" && i + 1 <= length(args)
            cot_p_path = args[i+1]; i += 2
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

    return doping_path, cot_n_path, cot_p_path, bias_path, output_path, mesh_path
end

function generate_1d_mesh_adjoint(n_nodes)
    L = 1e-4
    coord = collect(range(0.0, L, n_nodes))
    grid = simplexgrid(coord)
    cellmask!(grid, 0.0, L, 1)
    bfacemask!(grid, 0.0, 0.0, 1)
    bfacemask!(grid, L, L, 2)
    return grid
end

function dof_index(nspec, node, spec)
    return (node - 1) * nspec + spec
end

function central_diff_density(sol, data, dof_idx, inode, epsilon)
    u_orig = sol.u[dof_idx]

    sol.u[dof_idx] = u_orig + epsilon
    n_plus = get_density(sol, data, SPEC_E, 1; inode = inode)
    p_plus = get_density(sol, data, SPEC_H, 1; inode = inode)

    sol.u[dof_idx] = u_orig - epsilon
    n_minus = get_density(sol, data, SPEC_E, 1; inode = inode)
    p_minus = get_density(sol, data, SPEC_H, 1; inode = inode)

    sol.u[dof_idx] = u_orig

    dn = (n_plus - n_minus) / (2epsilon)
    dp = (p_plus - p_minus) / (2epsilon)
    return dn, dp
end

function main()
    doping_path, cot_n_path, cot_p_path, bias_path, output_path, mesh_path =
        parse_adjoint_args()

    doping_raw = npzread(doping_path)
    if ndims(doping_raw) > 1
        doping_raw = vec(doping_raw)
    end
    doping = Float64.(doping_raw)
    n_nodes = length(doping)

    cot_n_raw = npzread(cot_n_path)
    if ndims(cot_n_raw) > 1
        cot_n_raw = vec(cot_n_raw)
    end
    cot_n = Float64.(cot_n_raw)

    cot_p_raw = npzread(cot_p_path)
    if ndims(cot_p_raw) > 1
        cot_p_raw = vec(cot_p_raw)
    end
    cot_p = Float64.(cot_p_raw)

    if length(cot_n) != n_nodes || length(cot_p) != n_nodes
        error("cotangent length mismatch: doping has $n_nodes nodes, " *
              "cot_n has $(length(cot_n)), cot_p has $(length(cot_p))")
    end

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
        grid = generate_1d_mesh_adjoint(n_nodes)
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
    params.chargeNumbers[SPEC_E] = -1
    params.chargeNumbers[SPEC_H] = 1
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

    control = SolverControl()
    control.abstol = 1e-10
    control.reltol = 1e-10
    control.maxiters = 50
    control.max_round = 5
    control.verbose = false

    u0 = equilibrium_solve!(ctsys, control = control)

    if abs(bias_voltage) > 0.0 && cathode_breg <= n_bregions
        set_contact!(ctsys, cathode_breg, Δu = bias_voltage)
        sol = solve(ctsys; inival = u0, control = control)
    else
        sol = u0
    end

    residual, J_ext = VoronoiFVM.evaluate_residual_and_jacobian(ctsys.fvmsys, sol.u)
    J_csc = SparseMatrixCSC(flush!(J_ext))

    nspec = num_species(ctsys)
    ndof = nspec * n_nodes
    dJ_dx = zeros(Float64, ndof)
    epsilon = 1e-8

    for k in 1:n_nodes
        e_idx = dof_index(nspec, k, SPEC_E)
        h_idx = dof_index(nspec, k, SPEC_H)
        psi_idx = dof_index(nspec, k, DOF_PSI)

        dn_de, dp_de = central_diff_density(sol, data, e_idx, k, epsilon)
        dn_dh, dp_dh = central_diff_density(sol, data, h_idx, k, epsilon)
        dn_dpsi, dp_dpsi = central_diff_density(sol, data, psi_idx, k, epsilon)

        dJ_dx[e_idx] = cot_n[k] * dn_de + cot_p[k] * dp_de
        dJ_dx[h_idx] = cot_n[k] * dn_dh + cot_p[k] * dp_dh
        dJ_dx[psi_idx] = cot_n[k] * dn_dpsi + cot_p[k] * dp_dpsi
    end

    lambda = J_csc' \ (-dJ_dx)

    vjp = zeros(Float64, n_nodes)
    for k in 1:n_nodes
        psi_idx = dof_index(nspec, k, DOF_PSI)
        vjp[k] = -lambda[psi_idx]
    end

    npzwrite(output_path, vjp)
end

main()
