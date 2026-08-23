using LinearAlgebra
using SparseArrays

const CT_SPEC_E = 1
const CT_SPEC_H = 2
const CT_DOF_PSI = 3

# ``build_ct_system`` gives every silicon cell the same region index and the same
# material parameters, so any region reads the contact's densities of states and
# band-edge energies correctly.
const CT_SILICON_REGION = 1

# VoronoiFVM enforces a Dirichlet boundary condition by adding
# ``Dirichlet(Float64) = 1e30`` times the constraint to the row (see
# ``boundary_dirichlet!``). No physical drift-diffusion row of this device
# reaches 1e20, so a diagonal above that threshold identifies exactly the DOFs
# the penalty owns.
const CT_DIRICHLET_ROW_THRESHOLD = 1e20

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

"""Split the DOFs into the ones a Dirichlet penalty pins and the free rest."""
function dirichlet_partition(jacobian)
    n = size(jacobian, 1)
    pinned = [abs(jacobian[i, i]) >= CT_DIRICHLET_ROW_THRESHOLD for i in 1:n]
    return findall(pinned), findall(.!pinned)
end

"""Adjoint of the Dirichlet-eliminated system, scattered back over all DOFs.

The penalised rows carry a 1e30 diagonal, so the assembled Jacobian's condition
number is ~1e42 and the adjoint components on those rows -- which are O(1/1e30)
-- come back with no correct digits. Eliminating them first (the constrained
state is known exactly: ``x_c = bψEQ + Δu``) leaves the physical block, which a
dense partial-pivoting LU factorises accurately. The returned vector is zero on
the pinned rows; their sensitivity enters through
``pinned_dof_cotangent`` instead, which reads the well-scaled
``jacobian[free, pinned]`` coupling rather than a 1e30-amplified adjoint entry.
"""
function reduced_adjoint(jacobian, state_gradient, free)
    lambda = zeros(Float64, size(jacobian, 1))
    lambda[free] = Matrix(jacobian[free, free])' \ (-state_gradient[free])
    return lambda
end

function explicit_doping_contraction(ctsys, data, adjoint)
    fvmsys = ctsys.fvmsys
    nspec = VoronoiFVM.num_species(fvmsys)
    volumes = VoronoiFVM.nodevolumes(fvmsys)
    n_nodes = length(volumes)
    q_lambda = data.constants.q * fvmsys.physics.data.λ1
    gradient = zeros(Float64, n_nodes)

    # Doping enters the bulk residual only through the local Poisson source.
    # This is the contracted discrete derivative, not a global residual FD. On a
    # Dirichlet-pinned contact node the Poisson row is overridden by the
    # constraint, and ``adjoint`` is zero there, so no direct term is claimed
    # where the state does not respond to the local source.
    for inode in 1:n_nodes
        psi_idx = ct_dof_index(nspec, inode, CT_DOF_PSI)
        gradient[inode] = q_lambda * volumes[inode] * adjoint[psi_idx]
    end

    return gradient
end

"""``∂J/∂ψ_c`` for one Dirichlet-pinned potential DOF, via the reduced adjoint.

With ``x_c`` eliminated, ``dJ/dx_c = λ_free' * J[free, c] + ∂J/∂x_c``: the free
state responds to the constrained value through the ordinary coupling column,
which is physically scaled, unlike the 1e30 penalty entry on row ``c``.
"""
function pinned_dof_cotangent(jacobian, state_gradient, free, adjoint, c)
    return dot(adjoint[free], jacobian[free, c]) + state_gradient[c]
end

"""``dψ/dN`` of an ohmic contact's equilibrium potential w.r.t. its own doping.

At equilibrium ChargeTransport holds an ohmic contact at local charge
neutrality (``OhmicContactRobin``: ``Σ_α z_α n_α(ψ) = N`` at the boundary node,
enforced with a penalty ~1e10 times stronger than the bulk Poisson source it is
added to). ``bψEQ`` is therefore a *local* function of the doping at that
contact node, and differentiating the neutrality condition gives
``dψ/dN = 1 / (d/dψ Σ_α z_α n_α)`` -- for Boltzmann statistics ``-UT/(n+p)``.

Each carrier's ``dn_α/dψ`` is differenced separately and then summed with its
charge number so the two same-signed contributions never cancel, and the result
follows whatever ``data.F`` statistics the model uses.
"""
function contact_potential_doping_sensitivity(data, inode, psi)
    params = data.params
    paramsnodal = data.paramsnodal
    constants = data.constants
    thermal_voltage = constants.k_B * params.temperature / constants.q
    epsilon = 1e-6  # volts: ~4e-5 of the thermal voltage

    dcharge_dpsi = 0.0
    for icc in data.electricCarrierList
        z = params.chargeNumbers[icc]
        density_of_states = params.densityOfStates[icc, CT_SILICON_REGION] +
                            paramsnodal.densityOfStates[icc, inode]
        band_edge = params.bandEdgeEnergy[icc, CT_SILICON_REGION] +
                    paramsnodal.bandEdgeEnergy[icc, inode]
        eta(psi_value) = z / thermal_voltage * (-psi_value + band_edge / constants.q)
        density(psi_value) = density_of_states * data.F[icc](eta(psi_value))
        dcharge_dpsi += z * (density(psi + epsilon) - density(psi - epsilon)) / (2epsilon)
    end
    return 1.0 / dcharge_dpsi
end

"""Add the sensitivity that reaches the biased state through ``bψEQ``.

Out of equilibrium the ohmic contacts are Dirichlet conditions
``ψ = bψEQ[breg] + Δu`` (``OhmicContactDirichlet``, the default contact model),
and ``bψEQ`` is itself an output of the equilibrium solve. Doping therefore
moves the biased state twice: through the local Poisson source
(``explicit_doping_contraction``) and through the contact reference. At reverse
bias the rib is depleted and the second path carries most of the gradient, so
dropping it -- as an earlier version of this file did -- leaves the
composed gradient wrong by more than half.

The chain is ``∂J/∂bψEQ[breg]`` (summed over the region's pinned ψ DOFs) times
``dbψEQ[breg]/dN``, which is nonzero only at the one boundary node
``store_equilibrium_contact_data!`` reads ``bψEQ`` from.
"""
function equilibrium_contact_path!(
    gradient, ctsys, data, jacobian, state_gradient, free, adjoint,
)
    grid = ctsys.fvmsys.grid
    nspec = VoronoiFVM.num_species(ctsys.fvmsys)

    for ibreg in unique(grid[BFaceRegions])
        data.boundaryType[ibreg] == OhmicContact || continue
        boundary_grid = subgrid(grid, [ibreg], boundary = true)
        parents = boundary_grid[NodeParents]
        isempty(parents) && continue

        cotangent = 0.0
        for inode in parents
            psi_idx = ct_dof_index(nspec, inode, CT_DOF_PSI)
            abs(jacobian[psi_idx, psi_idx]) >= CT_DIRICHLET_ROW_THRESHOLD || continue
            cotangent += pinned_dof_cotangent(
                jacobian, state_gradient, free, adjoint, psi_idx,
            )
        end

        reference_node = parents[1]
        gradient[reference_node] += cotangent * contact_potential_doping_sensitivity(
            data, reference_node, data.params.bψEQ[ibreg],
        )
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

    # Linearise the OutOfEquilibrium residual at this operating point, whatever
    # the bias. The 0 V state is the equilibrium solution, but its adjoint must
    # still use the OutOfEquilibrium Jacobian: the InEquilibrium residual drops
    # the carrier continuity equations, so its Jacobian carries zero rows for the
    # carrier species and factorises as SingularException(0). The OutOfEquilibrium
    # linearisation at the same solution is well-posed and -- since the
    # equilibrium quasi-Fermi potentials are zero -- describes the identical
    # state, so it is the correct Jacobian at 0 V as well.
    configure_biased_state!(ctsys, data, bias_voltage, cathode_breg, n_bregions)

    jacobian = residual_jacobian(ctsys, sol)
    state_gradient = carrier_objective_state_gradient(ctsys, sol, data, cot_n, cot_p)
    _, free = dirichlet_partition(jacobian)
    adjoint = reduced_adjoint(jacobian, state_gradient, free)

    gradient = explicit_doping_contraction(ctsys, data, adjoint)
    equilibrium_contact_path!(
        gradient, ctsys, data, jacobian, state_gradient, free, adjoint,
    )
    # ``gradient`` is dJ/d(ParamsNodal.doping), in ChargeTransport's SI,
    # acceptor-positive convention. The caller differentiates w.r.t. PRISMO's
    # cm^-3 donor-positive field, so undo the affine change of variable
    # ``set_doping!`` applies (ct_common.jl).
    return gradient .* PRISMO_DOPING_TO_CT
end
