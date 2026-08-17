"""Precompile gyptis/FEniCS forms during image build.

The runtime creates fresh containers, so warming FFC's cache while building the
image avoids repeating identical form compilation on every container start.
"""

import importlib.util
from collections import OrderedDict
from pathlib import Path

import dolfin
import gyptis


def main() -> None:
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    epsilon = [12.0, 11.0, 1.0, 1.0]
    thickness = 1.0 / len(epsilon)
    thicknesses = OrderedDict(
        (f"domain_{index}", thickness)
        for index in range(1, len(epsilon) + 1)
    )
    geometry = gyptis.geometry.LayeredBoxPML2D(
        2.0,
        thicknesses=thicknesses,
        pml_width=(0.5, 0.5),
    )
    geometry.build()
    epsilon_by_domain = {
        f"domain_{index}": value
        for index, value in enumerate(epsilon, start=1)
    }
    simulation = gyptis.Waveguide(
        geometry,
        epsilon=epsilon_by_domain,
        wavenumber=2.0 * 3.141592653589793 / 1.55,
    )

    # Compile weak forms, boundary conditions, matrix assembly, and eigensolver
    # setup. Runtime VJP reuses these same formulation structures.
    simulation.eigensolve(n_eig=4, target=2.0 * 3.141592653589793 / 1.55)

    # Exercise the public runtime VJP path so image builds fail when gradient
    # evaluation regresses.
    api_path = Path("/tesseract/tesseract_api.py")
    spec = importlib.util.spec_from_file_location("gyptis_tesseract_api", api_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load gyptis tesseract API for warmup")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    api.apply(api.InputSchema(epsilon=epsilon))
    api.vector_jacobian_product(
        api.InputSchema(epsilon=epsilon),
        {"epsilon"},
        {"neff_sq"},
        {"neff_sq": 1.0},
    )


if __name__ == "__main__":
    main()
