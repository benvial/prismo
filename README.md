# Tesseract Photonic Waveguide

Differentiable PN-junction photonic phase shifter: DEVSIM + gyptis composed as Tesseracts

This is a **multi-Tesseract project**: a monorepo that combines several [Tesseracts](https://github.com/pasteurlabs/tesseract-core) into a pipeline application, with shared code, tests, and CI. Individual components live in `components/tesseracts/`, shared utilities in `components/shared_code/`, and the pipeline that chains them in `app/`.

New to Tesseract? Start with the [Tesseract Core docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/).

## Project structure

```bash
.
# 🚀 CI / CD
├── .github
│   └── workflows                    # CI/CD workflows
│       ├── pre_commit.yml           # Pre-commit hook checks
│       └── test.yaml                # Build components + run all tests (components & app)
# ✅ Code checks
├── .pre-commit-config.yaml          # Pre-commit configuration
├── ruff.toml                        # Ruff linter configuration
# 🔧 Pipeline code
├── app
│   ├── chain.ipynb                  # Notebook for running tesseracts and plotting outputs
│   ├── pyproject.toml               # App package configuration
│   ├── requirements.txt             # App runtime dependencies
│   ├── <myproject>                  # Main Python package
│   │   ├── __init__.py              # Package initialization
│   │   └── main.py                  # CLI entrypoint with typer
│   └── tests                        # App test suite
│       └── test_main.py             # Example test file
# 🧩 Component code
├── components                       # Tesseract components
│   ├── shared_code                  # Shared utilities across Tesseracts
│   │   ├── pyproject.toml           # Shared code package configuration
│   │   └── <myproject>_shared       # Shared code package
│   └── tesseracts                   # Individual tesseract implementations
│       └── <mytess>                 # Tesseract component, created via `make new`
│           ├── tesseract_api.py     # Tesseract API implementation
│           ├── tesseract_config.yaml # Tesseract configuration
│           ├── tesseract_requirements.txt # Tesseract dependencies
│           └── test_cases           # Regression test cases (*.json) for tesseract
# 📊 Data assets
├── data                             # Data assets directory
│   └── get_data.sh                  # Script to download/fetch data
# 🛠️ Scripts
├── scripts                          # Helper scripts
│   └── gen_test_case.py             # Capture a test case from a payload (used by `make gen-tests`)
# 📁 Auxiliary files
├── LICENSE                          # Project license
├── Makefile                         # Build automation (new, build, test, gen-tests, data, run)
└── README.md                        # Project documentation
```

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
$ make build mytess

# Test all components + app
$ make test

# Test a single component
$ make test mytess

# Test app only
$ make test app

# Run app end-to-end
$ make run
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

```bash
# Pull example data
$ make data
```
