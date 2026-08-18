using LinearAlgebra
using SparseArrays

const CT_SPEC_E = 1
const CT_SPEC_H = 2
const CT_DOF_PSI = 3
const CT_TINY_PENALTY_VALUE = 1e-10
const CT_DIRICHLET_PENALTY = 1e30

ct_dof_index(nspec, node, spec) = (node - 1) * nspec + spec

function central_diff_density(sol, data, dof_idx, inode, epsilon)
    u_orig = sol.u[dof_idx]

    sol.u[dof_idx] = u_orig + epsilon
    n_plus = get_density(sol, data, CT_SPEC_E, 1; inode = inode)
    p_plus = get_density(sol, data, CT_SPEC_H, 1; inode = inode)

    sol.u[dof_idx] = u_orig - epsilon
    n_minus = get_density(sol, data, CT_SPEC_E, 1; inode = inode)
    p_minus = get_density(sol, data, CT_SPEC_H, 1; inode = inode)

    sol.u[dof_idx] = u_orig

    return (n_plus - n_minus) / (2epsilon), (p_plus - p_minus) / (2epsilon)
end

function carrier_objective_state_gradient(ctsys, sol, data, cot_n, cot_p)
    nspec = VoronoiFVM.num_species(ctsys.fvmsys)
    n_nodes = length(cot_n)
    dJ_dx = zeros(Float64, nspec * n_nodes)
    epsilon = 1e-8

    for k in 1:n_nodes
        e_idx = ct_dof_index(nspec, k, CT_SPEC_E)
        h_idx = ct_dof_index(nspec, k, CT_SPEC_H)
        psi_idx = ct_dof_index(nspec, k, CT_DOF_PSI)

        dn_de, dp_de = central_diff_density(sol, data, e_idx, k, epsilon)
        dn_dh, dp_dh = central_diff_density(sol, data, h_idx, k, epsilon)
        dn_dpsi, dp_dpsi = central_diff_density(sol, data, psi_idx, k, epsilon)

        dJ_dx[e_idx] = cot_n[k] * dn_de + cot_p[k] * dp_de
        dJ_dx[h_idx] = cot_n[k] * dn_dh + cot_p[k] * dp_dh
        dJ_dx[psi_idx] = cot_n[k] * dn_dpsi + cot_p[k] * dp_dpsi
    end
    return dJ_dx
end

function residual_jacobian(ctsys, sol)
    _, jacobian_ext = VoronoiFVM.evaluate_residual_and_jacobian(ctsys.fvmsys, sol.u)
    return SparseMatrixCSC(ExtendableSparse.flush!(jacobian_ext))
end

function carrier_adjoint(ctsys, sol, data, cot_n, cot_p)
    jacobian = residual_jacobian(ctsys, sol)
    state_gradient = carrier_objective_state_gradient(ctsys, sol, data, cot_n, cot_p)
    return jacobian' \ (-state_gradient)
end

function for_each_boundary_node(callback, grid, nodefactors)
    bnodes = grid[BFaceNodes]
    bregions = grid[BFaceRegions]
    if ndims(bnodes) == 1
        for ibface in eachindex(bnodes)
            callback(bnodes[ibface], nodefactors[ibface], bregions[ibface])
        end
        return nothing
    end

    for ibface in axes(bnodes, 2), ilocal in axes(bnodes, 1)
        inode = bnodes[ilocal, ibface]
        inode == 0 && continue
        callback(inode, nodefactors[ilocal, ibface], bregions[ibface])
    end
    return nothing
end

function contact_boundary_regions(data, grid)
    return unique([
        ibreg for ibreg in grid[BFaceRegions] if data.boundaryType[ibreg] == OhmicContact
    ])
end

function explicit_doping_contraction(
    ctsys,
    data,
    adjoint;
    include_equilibrium_robin = false,
)
    fvmsys = ctsys.fvmsys
    grid = fvmsys.grid
    nspec = VoronoiFVM.num_species(fvmsys)
    n_nodes = length(VoronoiFVM.nodevolumes(fvmsys))
    volumes = VoronoiFVM.nodevolumes(fvmsys)
    q_lambda = data.constants.q * fvmsys.physics.data.λ1
    gradient = zeros(Float64, n_nodes)

    # Doping enters the bulk residual only through the local Poisson source.
    # This is the contracted discrete derivative, not a global residual FD.
    for inode in 1:n_nodes
        psi_idx = ct_dof_index(nspec, inode, CT_DOF_PSI)
        gradient[inode] = q_lambda * volumes[inode] * adjoint[psi_idx]
    end

    if include_equilibrium_robin
        nodefactors = VoronoiFVM.bfacenodefactors(fvmsys)
        robin_scale = q_lambda / CT_TINY_PENALTY_VALUE
        for_each_boundary_node(grid, nodefactors) do inode, factor, ibreg
            if data.boundaryType[ibreg] == OhmicContact
                psi_idx = ct_dof_index(nspec, inode, CT_DOF_PSI)
                gradient[inode] += robin_scale * factor * adjoint[psi_idx]
            end
        end
    end

    return gradient
end

function biased_boundary_cotangent(ctsys, data, adjoint)
    grid = ctsys.fvmsys.grid
    nspec = VoronoiFVM.num_species(ctsys.fvmsys)
    boundary_cotangent = zeros(Float64, length(data.params.bψEQ))
    nodefactors = VoronoiFVM.bfacenodefactors(ctsys.fvmsys)
    for_each_boundary_node(grid, nodefactors) do inode, _, ibreg
        if data.boundaryType[ibreg] == OhmicContact
            psi_idx = ct_dof_index(nspec, inode, CT_DOF_PSI)
            boundary_cotangent[ibreg] -= CT_DIRICHLET_PENALTY * adjoint[psi_idx]
        end
    end
    return boundary_cotangent
end

function equilibrium_boundary_adjoint(ctsys, data, equilibrium_sol, boundary_cotangent)
    grid = ctsys.fvmsys.grid
    nspec = VoronoiFVM.num_species(ctsys.fvmsys)
    configure_equilibrium!(ctsys, data)
    equilibrium_jacobian = residual_jacobian(ctsys, equilibrium_sol)
    rhs = zeros(Float64, size(equilibrium_jacobian, 1))
    for ibreg in contact_boundary_regions(data, grid)
        inode = equilibrium_bpsi_parent_node(grid, ibreg)
        psi_idx = ct_dof_index(nspec, inode, CT_DOF_PSI)
        rhs[psi_idx] -= boundary_cotangent[ibreg]
    end
    return equilibrium_jacobian' \ rhs
end

function configure_biased_state!(ctsys, data, bias_voltage, cathode_breg, n_bregions)
    data.calculationType = ChargeTransport.OutOfEquilibrium
    if cathode_breg <= n_bregions
        set_contact!(ctsys, cathode_breg, Δu = bias_voltage)
    end
    return nothing
end

function compute_doping_vjp(
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
    n_nodes = length(cot_n)
    length(cot_p) == n_nodes || error("cotangent length mismatch")

    if abs(bias_voltage) == 0.0 || cathode_breg > n_bregions
        configure_equilibrium!(ctsys, data)
        equilibrium_adjoint = carrier_adjoint(ctsys, sol, data, cot_n, cot_p)
        return explicit_doping_contraction(
            ctsys,
            data,
            equilibrium_adjoint;
            include_equilibrium_robin = true,
        )
    end

    configure_biased_state!(ctsys, data, bias_voltage, cathode_breg, n_bregions)
    biased_adjoint = carrier_adjoint(ctsys, sol, data, cot_n, cot_p)
    direct_gradient = explicit_doping_contraction(ctsys, data, biased_adjoint)

    boundary_cotangent = biased_boundary_cotangent(ctsys, data, biased_adjoint)
    equilibrium_adjoint = equilibrium_boundary_adjoint(
        ctsys,
        data,
        equilibrium_sol,
        boundary_cotangent,
    )
    return direct_gradient + explicit_doping_contraction(
        ctsys,
        data,
        equilibrium_adjoint;
        include_equilibrium_robin = true,
    )
end
