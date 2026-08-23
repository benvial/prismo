# Components

- `shared_code/` — `prismo_shared`, the Pydantic schemas (mesh reference, Soref–Bennett coefficients, solver sessions) installed into both Tesseract images and the host app. This is the contract the solvers agree on.
- `tesseracts/chargetransport/` — Python 3.12 + Julia 1.10 image. `tesseract_api.py` owns one persistent Julia worker (`scripts/worker.jl`) that runs ChargeTransport.jl forward solves and the discrete-adjoint VJP. Built on the Julia base image from `Dockerfile.julia-base` (`make julia-base chargetransport`).
- `tesseracts/gyptis/` — conda image (legacy FEniCS, Python 3.10). Authors the shared mesh, solves the vector eigenmode problem, and returns the Hellmann–Feynman eigen-adjoint VJP per design cell.
- `tesseracts/.template/` — scaffold copied by `make new <name>`.

Each component has `test_cases/*.json` regression cases run by `make test <name>` against its built image, and `tests/` unit tests that run in-process where the solver is importable.
