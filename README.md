

# PRISMO

## **P**hotonic **R**econfigurable **I**ntegrated **S**emiconductor **M**ultiphysics **O**ptimization

Differentiable PN-junction photonic phase shifter: gyptis and ChargeTransport.jl composed as Tesseracts.

This is a **multi-Tesseract project**: a monorepo that combines several [Tesseracts](https://github.com/pasteurlabs/tesseract-core) into a differentiable pipeline for topology optimization of a reverse-biased PN-junction phase shifter. Each solver is a standalone Tesseract with its own automatic differentiation strategy — eigen-adjoint (gyptis) and discrete adjoint (ChargeTransport.jl). Shared Pydantic schemas in `components/shared_code/` define the exchange format across solver boundaries, and the pipeline composition lives in `app/`.

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
│   ├── pyproject.toml
│   ├── requirements.txt             # tesseract-core
│   ├── prismo
│   │   ├── __init__.py
│   │   ├── main.py                  # CLI entrypoint (typer): run, validate-gradient
│   │   ├── pipeline.py              # θ → Δneff, JAX-differentiable end to end
│   │   ├── optimizer.py             # NLopt MMA loop over the signed design field
│   │   ├── density_filter.py        # Andreassen density filter (sparse H matrix)
│   │   ├── soref_bennett.py         # Carrier → permittivity coupling layer
│   │   ├── mesh_transfer.py         # Nodal field → design-cell exact restriction
│   │   ├── differentiable_component.py  # Tesseract apply/VJP adapter
│   │   ├── outputs.py               # Headline figures + gradient validation
│   │   ├── waveguide_mesh.py        # Local (non-container) Gmsh mesh fallback
│   │   └── _version.py
│   ├── outputs/                     # Generated figures
│   └── tests
# 🧩 Component code
├── components
│   ├── shared_code
│   │   ├── pyproject.toml
│   │   └── prismo_shared
│   │       └── schemas.py           # Pydantic schemas (mesh, carrier/permittivity fields)
│   └── tesseracts
│       ├── .template                # Scaffold for `make new`
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
│           ├── scripts/             # worker.jl, ct_adjoint.jl, ct_common.jl, contacts.jl
│           ├── tests/
│           └── test_cases/
# 🛠️ Scripts
├── scripts
│   ├── gen_test_case.py             # Capture test case from payload (`make gen-tests`)
│   └── prototype_gyptis_eigen_adjoint.py
# 📁 Auxiliary files
├── LICENSE
├── Makefile
├── README.md
├── CONTEXT.md                       # Domain glossary
├── RULES.md                         # Hackathon brief
└── docs/
    ├── adr/                         # Architecture decision records
    ├── agents/                      # Agent workflow reference
    └── research/                    # Research memos
```

## Solver components

Two Tesseracts split the PN-junction phase-shifter problem across two physics engines and two autodiff strategies:

| Tesseract | Engine | Language | Solves on | Adjoint strategy |
|---|---|---|---|---|
| `gyptis` | gyptis / FEniCS | conda | Full optical domain | Eigen-adjoint (Hellmann-Feynman) |
| `chargetransport` | ChargeTransport.jl | Python + Julia | Silicon subdomain | Discrete adjoint |

Both consume **one shared mesh**, authored by the gyptis Tesseract's `write_mesh`
op (see [ADR 0002](docs/adr/0002-shared-mesh-authored-by-gyptis.md)). Shared
Pydantic schemas (`components/shared_code/`) define the mesh and
carrier/permittivity field exchange format so any solver can be swapped into the
pipeline.

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
$ make build gyptis

# Test all components + app
$ make test

# Test a single component
$ make test gyptis

# Test app only
$ make test app

# Run the optimization end-to-end against the real solvers
$ make run-containers

# Validate the composed adjoint against central finite differences
$ make validate-gradient-containers

# Clean build artifacts and caches
$ make clean
```

There is no stub solver path: `make run` and `make validate-gradient` without
the containers raise rather than substituting a fake forward or gradient. Tests
inject explicit doubles through the `components=` seam instead.

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
