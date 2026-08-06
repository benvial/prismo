"""Shared Pydantic schemas for the photonic waveguide pipeline.

Types here cross the DEVSIM <-> gyptis container boundary: the shared Gmsh
mesh description, the Soref-Bennett carrier-to-permittivity coupling arrays,
and mesh-transfer metadata (ticket 07).
"""


from pydantic import BaseModel, Field


class MeshRef(BaseModel):
    """Reference to the shared Gmsh mesh both solvers operate on.

    Attributes:
        path: Filesystem path to the Gmsh .msh file.
        n_nodes: Number of mesh nodes (DEVSIM FV nodes).
        n_elements: Number of mesh elements (gyptis FEM elements).
        node_ordering: Convention label for node ordering (e.g.
            ``"gmsh"``, ``"x_then_y"``).
    """

    path: str
    n_nodes: int = 0
    n_elements: int = 0
    node_ordering: str = "gmsh"


class SorefBennettCoefficients(BaseModel):
    """Wavelength-dependent Soref-Bennett coefficients for silicon at 300 K.

    Models the free-carrier plasma dispersion effect:

        Delta_n = -(A_e * Delta_N_e^{B_e} + A_h * Delta_N_h^{B_h})
        Delta_alpha = C_e * Delta_N_e^{D_e} + C_h * Delta_N_h^{D_h}

    where Delta_N_e, Delta_N_h are electron/hole density changes [cm^{-3}],
    Delta_n is the refractive index change, and Delta_alpha is the absorption
    coefficient change [cm^{-1}].

    Default values are for lambda = 1.55 um from Soref & Bennett, IEEE JQE
    23(1), 1987, Table I.
    """

    wavelength_um: float = Field(default=1.55, ge=1.1, le=2.0)
    A_e: float = 8.8e-22
    B_e: float = 1.0
    C_e: float = 8.5e-18
    D_e: float = 1.0
    A_h: float = 8.5e-18
    B_h: float = 0.8
    C_h: float = 6.0e-18
    D_h: float = 1.0
    background_index: float = Field(
        default=3.4757, description="Unperturbed silicon refractive index at lambda"
    )


class SorefBennettResult(BaseModel):
    """Output of the Soref-Bennett coupling layer.

    Maps carrier density perturbations from DEVSIM into permittivity
    perturbations for gyptis.

    Attributes:
        delta_permittivity: Permittivity change per element/subdomain
            (dimensionless).  Derived from Delta_n via
            Delta_epsilon = 2 * n_bg * Delta_n (first-order expansion).
        delta_absorption: Absorption-coefficient change per element [cm^{-1}].
    """

    delta_permittivity: list[float]
    delta_absorption: list[float]


class CarrierDensityField(BaseModel):
    """Carrier density distribution from the DEVSIM drift-diffusion solve.

    Attributes:
        electrons: Electron concentration per mesh node [m^{-3}].
        holes: Hole concentration per mesh node [m^{-3}].
        equilibrium_electrons: Equilibrium (zero-bias) electron concentration
            per node [m^{-3}], for computing Delta_N.
        equilibrium_holes: Equilibrium hole concentration per node [m^{-3}].
    """

    electrons: list[float]
    holes: list[float]
    equilibrium_electrons: list[float] | None = None
    equilibrium_holes: list[float] | None = None


class PermittivityField(BaseModel):
    """Per-subdomain permittivity distribution for the gyptis eigenmode solve.

    Attributes:
        epsilon: Relative permittivity per subdomain/element (dimensionless).
        subdomain_labels: Optional human-readable labels for each subdomain
            (e.g. ``["core", "cladding", "substrate"]``).
    """

    epsilon: list[float]
    subdomain_labels: list[str] | None = None
