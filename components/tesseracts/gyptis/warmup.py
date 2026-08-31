"""Precompile gyptis/FEniCS forms during image build.

The runtime creates fresh containers, so warming FFC's cache while building the
image avoids repeating identical form compilation on every container start.
The Binder image runs the same script from the source tree (``binder/postBuild``).
"""

import importlib.util
from pathlib import Path


def main() -> None:
    """Run the served endpoints once so FEniCS JIT-compiles their forms here."""
    # Tesseract copies this script to /tmp and the API to /tesseract/; run from
    # the source tree (Binder), the API is the sibling file.
    api_path = Path(__file__).with_name("tesseract_api.py")
    if not api_path.exists():
        api_path = Path("/tesseract/tesseract_api.py")
    spec = importlib.util.spec_from_file_location("gyptis_tesseract_api", api_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load gyptis tesseract API for warmup")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)

    # The public forward and adjoint between them compile every weak form,
    # boundary condition, assembly kernel and eigensolver setup the endpoints
    # use, with the production geometry, element degree and domain structure.
    # Warming through the API rather than a hand-built problem also fails the
    # image build when either endpoint regresses. Changes to topology, element
    # degree or formulation should be checked against this coverage.
    design_epsilon = [api.DEFAULT_CORE_EPSILON for _ in api.design_cell_centroids()]
    inputs = api.InputSchema(design_epsilon=design_epsilon)
    api.apply(inputs)
    api.vector_jacobian_product(
        inputs,
        {"design_epsilon"},
        {"neff_sq"},
        {"neff_sq": 1.0},
    )


if __name__ == "__main__":
    main()
