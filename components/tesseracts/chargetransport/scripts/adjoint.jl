#!/usr/bin/env julia

using NPZ
using JSON
using LinearAlgebra
using SparseArrays

# VoronoiFVM access (qualified) comes from ct_common.jl.
include(joinpath(@__DIR__, "ct_common.jl"))

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

    ctsys, data, cathode_breg, n_bregions = build_ct_system(doping, mesh_path)

    control = make_solver_control()

    u0 = solve_equilibrium(ctsys, data, doping, control)
    sol = solve_at_bias(ctsys, control, u0, bias_voltage, cathode_breg, n_bregions)

    residual, J_ext = VoronoiFVM.evaluate_residual_and_jacobian(ctsys.fvmsys, sol.u)
    J_csc = SparseMatrixCSC(ExtendableSparse.flush!(J_ext))

    nspec = VoronoiFVM.num_species(ctsys.fvmsys)
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
