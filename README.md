

# PRISMO

## **P**hotonic **R**econfigurable **I**ntegrated **S**emiconductor **M**ultiphysics **O**ptimization

Differentiable PN-junction photonic phase shifter: DEVSIM, gyptis, and ChargeTransport.jl composed as Tesseracts.

This is a **multi-Tesseract project**: a monorepo that combines several [Tesseracts](https://github.com/pasteurlabs/tesseract-core) into a differentiable pipeline for topology optimization of a reverse-biased PN-junction phase shifter. Each solver is a standalone Tesseract with its own automatic differentiation strategy — implicit adjoint (DEVSIM), eigen-adjoint (gyptis), and discrete adjoint (ChargeTransport.jl). Shared Pydantic schemas in `components/shared_code/` define the exchange format across solver boundaries, and the pipeline composition lives in `app/`.

New to Tesseract? Start with the [Tesseract Core docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/).

## Project structure

```bash
.
# 🚀 CI / CD
├── .github
│   └── workflows
│       ├── pre_commit.yml
│       └── test.yaml
# ✅ Code checks
├── .pre-commit-config.yaml
├── ruff.toml
# 🔧 Pipeline code
├── app
│   ├── chain.ipynb                  # Pipeline notebook (Tesseract composition, stubs)
│   ├── pyproject.toml
│   ├── requirements.txt             # tesseract-core
│   ├── tesseract_photonic_waveguide
│   │   ├── __init__.py
│   │   ├── main.py                  # CLI entrypoint (typer)
│   │   ├── density_filter.py        # Andreassen density filter (sparse H matrix)
│   │   ├── soref_bennett.py         # Carrier → permittivity coupling layer
│   │   ├── waveguide_mesh.py        # SOI rib waveguide Gmsh mesh builder
│   │   └── _version.py
│   └── tests
│       ├── test_main.py
│       ├── test_density_filter.py
│       ├── test_soref_bennett.py
│       └── test_waveguide_mesh.py
# 🧩 Component code
├── components
│   ├── shared_code
│   │   ├── pyproject.toml
│   │   └── tesseract_photonic_waveguide_shared
│   │       └── schemas.py           # Pydantic schemas (mesh, carrier/permittivity fields)
│   └── tesseracts
│       ├── .template                # Scaffold for `make new`
│       ├── devsim                   # Python 3.12 — DEVSIM drift-diffusion (implicit adjoint)
│       │   ├── tesseract_api.py
│       │   ├── tesseract_config.yaml
│       │   ├── tesseract_requirements.txt
│       │   ├── tests/
│       │   └── test_cases/
│       ├── gyptis                   # conda — gyptis/FEniCS EM eigenmode (eigen-adjoint)
│       │   ├── tesseract_api.py
│       │   ├── tesseract_config.yaml
│       │   ├── tesseract_environment.yaml
│       │   ├── tests/
│       │   └── test_cases/
│       └── chargetransport          # Python 3.12 + Julia — ChargeTransport.jl (discrete adjoint)
│           ├── tesseract_api.py
│           ├── tesseract_config.yaml
│           ├── tesseract_requirements.txt
│           ├── scripts/             # forward.jl, contacts.jl
│           ├── tests/
│           └── test_cases/
# 🛠️ Scripts
├── scripts
│   ├── gen_test_case.py             # Capture test case from payload (`make gen-tests`)
│   ├── prototype_devsim_adjoint.py
│   └── prototype_gyptis_eigen_adjoint.py
# 📁 Auxiliary files
├── LICENSE
├── Makefile
├── README.md
├── CONTEXT.md                       # Domain glossary
├── RULES.md                         # Hackathon brief
└── docs/
    ├── agents/                      # Agent workflow reference
    └── research/                    # Research memos
```

## Solver components

Three Tesseracts implement the same PN-junction phase-shifter problem with different physics engines and autodiff strategies:

| Tesseract | Engine | Language | Adjoint strategy |
|---|---|---|---|
| `devsim` | DEVSIM | Python 3.12 | Implicit (Newton Jacobian) |
| `gyptis` | gyptis / FEniCS | conda | Eigen-adjoint (Hellmann-Feynman) |
| `chargetransport` | ChargeTransport.jl | Python + Julia | Discrete adjoint |

Shared Pydantic schemas (`components/shared_code/`) define the mesh and carrier/permittivity field exchange format so any solver can be swapped into the pipeline.

## Usage

**Prerequisites:** [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) and a running Docker daemon (Tesseract builds and runs components as containers).

**Platforms:** Linux and macOS are supported. On Windows, use [WSL2](https://learn.microsoft.com/windows/wsl/). The `make` workflow assumes a POSIX shell.

```bash
# Create a new Tesseract component
$ make new mytess

# Create a component from a recipe (base | jax | pytorch)
$ make new mytess RECIPE=jax

# Build all components
$ make build

# Build a single Tesseract
$ make build devsim

# Test all components + app
$ make test

# Test a single component
$ make test devsim

# Test app only
$ make test app

# Run app end-to-end (stub)
$ make run

# Clean build artifacts and caches
$ make clean
```

## Adding regression test cases

Each component runs regression tests from JSON files in its `test_cases/`
directory (see `make test`). To capture one automatically, write a small file
holding an input `payload`, then let the component run it and record the output:

```bash
# payload.json:  {"inputs": {"vector": [1.0, 2.0, 3.0], "scale_factor": 2.0}}
$ make build mytess
$ make gen-tests mytess FILE=payload.json
```

This runs the `apply` endpoint and writes a ready-to-run test case (input
payload + captured `expected_outputs`) to `components/tesseracts/mytess/test_cases/`.
Review the result before committing — numeric tolerances (`atol`/`rtol`) and any
non-deterministic outputs may need hand-editing. Pass `ENDPOINT=` and `OUT=` to
target a different endpoint or output filename.
