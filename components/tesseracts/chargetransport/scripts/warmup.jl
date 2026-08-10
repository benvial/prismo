#!/usr/bin/env julia

# PackageCompiler precompile workload (ticket 17).
#
# Executed once during the base-image build via
#   create_sysimage(...; precompile_execution_file = "warmup.jl")
# so the runtime scripts (forward.jl / adjoint.jl) start with all solver
# machinery already compiled to native code. Without this, the first
# equilibrium_solve! pays ~22 s of JIT compilation, blowing the 30 s
# apply() budget.
#
# MUST exercise the same functions/grid types as the runtime scripts:
#   - build_ct_system on the 1D fallback grid AND on a Gmsh 2D mesh file
#     (different grid types -> different method specializations)
#   - equilibrium_solve! + solve_at_bias with a reverse-bias ramp
#   - get_density extraction (also via central differences, adjoint path)
#   - evaluate_residual_and_jacobian + CSC conversion + sparse backsolve
#   - NPZ / JSON I/O

using NPZ
using JSON
using LinearAlgebra
using SparseArrays

# VoronoiFVM access (qualified) comes from ct_common.jl.
include(joinpath(@__DIR__, "ct_common.jl"))

const MINIMAL_TRIANGLE_MSH = """
\$MeshFormat
2.2 0 8
\$EndMeshFormat
\$PhysicalNames
2
1 1 "contact_anode"
1 2 "contact_cathode"
\$EndPhysicalNames
\$Nodes
3
1 0.0 0.0 0.0
2 1e-7 0.0 0.0
3 0.0 1e-7 0.0
\$EndNodes
\$Elements
4
1 1 2 1 1 1 2
2 1 2 2 2 2 3
3 1 2 1 3 3 1
4 2 2 1 4 1 2 3
\$EndElements
"""

function central_diff(sol, data, dof_idx, inode, epsilon)
    u_orig = sol.u[dof_idx]
    sol.u[dof_idx] = u_orig + epsilon
    n_plus = get_density(sol, data, 1, 1; inode = inode)
    sol.u[dof_idx] = u_orig - epsilon
    n_minus = get_density(sol, data, 1, 1; inode = inode)
    sol.u[dof_idx] = u_orig
    return (n_plus - n_minus) / (2epsilon)
end

function exercise(doping, mesh_path, bias_voltage)
    n_nodes = length(doping)
    ctsys, data, cathode_breg, n_bregions = build_ct_system(doping, mesh_path)
    control = make_solver_control()
    u0 = solve_equilibrium(ctsys, data, doping, control)
    sol = solve_at_bias(ctsys, control, u0, bias_voltage, cathode_breg, n_bregions)

    electrons = [get_density(sol, data, 1, 1; inode = i) for i in 1:n_nodes]
    holes = [get_density(sol, data, 2, 1; inode = i) for i in 1:n_nodes]

    # adjoint-path machinery
    residual, J_ext = VoronoiFVM.evaluate_residual_and_jacobian(ctsys.fvmsys, sol.u)
    J_csc = SparseMatrixCSC(ExtendableSparse.flush!(J_ext))
    nspec = VoronoiFVM.num_species(ctsys.fvmsys)
    dJ_dx = zeros(Float64, nspec * n_nodes)
    for k in 1:n_nodes
        dJ_dx[(k-1)*nspec+1] = central_diff(sol, data, (k-1)*nspec+1, k, 1e-8)
    end
    lambda = J_csc' \ (-dJ_dx)

    return electrons, holes, lambda
end

function main()
    mktempdir() do tmp
        # I/O paths used by the runtime scripts
        doping_path = joinpath(tmp, "doping.npy")
        bias_path = joinpath(tmp, "bias.json")
        out_path = joinpath(tmp, "carriers.npz")
        npzwrite(doping_path, fill(1e19, 16))
        open(bias_path, "w") do io
            JSON.print(io, Dict("bias_voltage" => -5.0))
        end
        doping = Float64.(vec(npzread(doping_path)))
        bias_voltage = Float64(JSON.parsefile(bias_path)["bias_voltage"])

        # 1D fallback grid at full reverse bias
        e1, h1, _ = exercise(doping, "", bias_voltage)
        npzwrite(out_path, Dict("electrons" => e1, "holes" => h1))

        # 2D Gmsh mesh path (physical-group contacts)
        mesh_path = joinpath(tmp, "tiny.msh")
        write(mesh_path, MINIMAL_TRIANGLE_MSH)
        exercise(fill(1e22, 3), mesh_path, -1.0)
    end
    println("warmup done")
end

main()
