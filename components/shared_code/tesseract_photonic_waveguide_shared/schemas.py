"""Shared Pydantic schemas for the photonic waveguide pipeline.

Types here cross the DEVSIM <-> gyptis container boundary: the shared Gmsh
mesh description and the Soref-Bennett coupling arrays (ticket 07).

Placeholder — fields land when the coupling layer is implemented.
"""

from pydantic import BaseModel


class MeshRef(BaseModel):
    """Reference to the shared Gmsh mesh both solvers operate on."""

    path: str
    # TODO(ticket 07): node/element ordering convention for field transfer.
