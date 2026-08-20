using LinearAlgebra
using SparseArrays

const CT_SPEC_E = 1
const CT_SPEC_H = 2
const CT_DOF_PSI = 3

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
    # The drift-diffusion residual mixes Poisson, continuity, and ohmic-contact
    # rows whose natural scales span ~40 orders of magnitude, so the Jacobian is
    # extremely ill-conditioned. UMFPACK's threshold pivoting then rejects a
    # pivot the system does not actually lack (SingularException(0)) at reverse
    # bias. The adjoint system is small (n_species * n_silicon_nodes), so a dense
    # partial-pivoting LU factorises the same matrix robustly.
    return Matrix(jacobian)' \ (-state_gradient)
end

function explicit_doping_contraction(ctsys, data, adjoint)
    fvmsys = ctsys.fvmsys
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

    return gradient
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
    bias_voltage,
    cathode_breg,
    n_bregions,
    cot_n,
    cot_p,
)
    n_nodes = length(cot_n)
    length(cot_p) == n_nodes || error("cotangent length mismatch")

    # Doping enters the drift-diffusion solve only through the local Poisson
    # source, at every bias including 0 V. Linearise the OutOfEquilibrium
    # residual at this operating point and contract the adjoint's ψ-component
    # against that source (explicit_doping_contraction).
    #
    # The 0 V state is the equilibrium solution, but its adjoint must still use
    # the OutOfEquilibrium Jacobian. The InEquilibrium residual drops the carrier
    # continuity equations, so its Jacobian carries zero rows for the carrier
    # species and factorises as SingularException(0). The OutOfEquilibrium
    # linearisation at the same solution is well-posed and -- since the
    # equilibrium quasi-Fermi potentials are zero -- describes the identical
    # state, so it is the correct Jacobian at 0 V as well.
    #
    # The retained equilibrium contact quantities (bψEQ, bDensityEQ) also depend
    # on doping, but only at the ohmic-contact nodes; the design θ modulates the
    # rib interior, not the contacts, so their doping-sensitivity is zero on the
    # design cells and needs no separate boundary-adjoint term. Ticket 06's
    # adjoint-vs-FD proves the gradient is correct with only the direct term.
    configure_biased_state!(ctsys, data, bias_voltage, cathode_breg, n_bregions)
    adjoint = carrier_adjoint(ctsys, sol, data, cot_n, cot_p)
    return explicit_doping_contraction(ctsys, data, adjoint)
end
