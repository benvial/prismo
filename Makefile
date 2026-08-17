.PHONY: help new build test gen-tests data run run-containers clean

PYTHON ?= python

help:
	@echo "Available targets:"
	@echo "  new NAME [RECIPE=jax|pytorch] - Create a new Tesseract component (make new mytess RECIPE=jax)"
	@echo "  build [NAME]                  - Build all components or a single tesseract (make build mytess)"
	@echo "  julia-base NAME               - (Re)build the Julia base image for a component (only needed when its julia_env/*.toml change)"
	@echo "  test [NAME]                   - Test all components + app, a single component, or app only"
	@echo "  gen-tests NAME FILE=case.json - Capture a test case by running an input payload (make gen-tests mytess FILE=in.json)"
	@echo "  data                          - Pull example data"
	@echo "  run                           - Run app end-to-end"
	@echo "  run-containers                - Run app with both Docker containers"
	@echo "  clean                         - Remove build artifacts, caches, and temp files"

new:
	@set -e; \
	TESS_NAME="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$TESS_NAME" ]; then \
		read -p "Enter tesseract name: " TESS_NAME; \
		if [ -z "$$TESS_NAME" ]; then \
			echo "Error: Tesseract name cannot be empty"; \
			exit 1; \
		fi; \
	fi; \
	TESS_SLUG=$$(echo "$$TESS_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr '-' '_'); \
	TESS_DIR="components/tesseracts/$$TESS_SLUG"; \
	if [ -d "$$TESS_DIR" ]; then \
		echo "Error: Tesseract $$TESS_SLUG already exists"; \
		exit 1; \
	fi; \
	echo "Creating new tesseract: $$TESS_SLUG"; \
	if [ -n "$(RECIPE)" ]; then \
		echo "Recipe: $(RECIPE)"; \
		tesseract init --name "prismo_$$TESS_SLUG" --target-dir "$$TESS_DIR" --recipe "$(RECIPE)"; \
	else \
		tesseract init --name "prismo_$$TESS_SLUG" --target-dir "$$TESS_DIR"; \
	fi; \
	cp -r components/tesseracts/.template/* "$$TESS_DIR/"; \
	printf '\n../../shared_code\n' >> "$$TESS_DIR/tesseract_requirements.txt"; \
	echo "Tesseract component created at $$TESS_DIR"

julia-base:
	@set -e; \
	TESS_NAME="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$TESS_NAME" ]; then \
		echo "Usage: make julia-base <mytess>"; \
		exit 1; \
	fi; \
	TESS_DIR="components/tesseracts/$$TESS_NAME"; \
	if [ ! -f "$$TESS_DIR/Dockerfile.julia-base" ]; then \
		echo "Error: $$TESS_DIR/Dockerfile.julia-base not found (this component has no Julia base image)"; \
		exit 1; \
	fi; \
	echo "Building Julia base image for $$TESS_NAME (prismo_$${TESS_NAME}_julia_base:latest)..."; \
	docker build -f "$$TESS_DIR/Dockerfile.julia-base" -t "prismo_$${TESS_NAME}_julia_base:latest" "$$TESS_DIR"; \
	echo "Done. Now rebuild the component with: make build $$TESS_NAME"

# Tesseract builds with `docker buildx build`, which uses the default buildx
# builder. If that builder is a docker-container driver it cannot see locally
# built images (e.g. prismo_chargetransport_julia_base) and tries to pull them
# from a registry. Force the docker-driver builder, which shares the daemon's
# local image store. Override by exporting TESSERACT_DOCKER_BUILD_ARGS yourself.
export TESSERACT_DOCKER_BUILD_ARGS ?= --builder default

build:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Building all components..."; \
		for dir in components/tesseracts/*/; do \
			if [ -d "$$dir" ] && [ "$$(basename $$dir)" != ".template" ]; then \
				echo "Building $$(basename $$dir)..."; \
				tesseract build $$dir; \
			fi; \
		done; \
	else \
		echo "Building tesseract: $(filter-out $@,$(MAKECMDGOALS))"; \
		TESS_NAME=$(filter-out $@,$(MAKECMDGOALS)); \
		TESS_DIR="components/tesseracts/$$TESS_NAME"; \
		if [ -d "$$TESS_DIR" ]; then \
			tesseract build $$TESS_DIR; \
		else \
			echo "Error: Tesseract $$TESS_NAME not found"; \
			exit 1; \
		fi; \
	fi

test:
	@run_tesseract_tests() { \
		tess_dir="$$1"; \
		tess_slug=$$(basename "$$tess_dir"); \
		found=0; \
		for test_file in "$$tess_dir"/test_cases/*.json; do \
			[ -f "$$test_file" ] || continue; \
			found=1; \
			echo "  Running $$(basename $$test_file)..."; \
			result=$$(tesseract run prismo_$$tess_slug test @$$test_file); \
			echo "$$result"; \
			echo "$$result" | python -c 'import json,sys; sys.exit(0 if json.loads(sys.stdin.read(), strict=False).get("status")=="passed" else 1)' \
				|| { echo "  FAILED: $$(basename $$test_file)"; return 1; }; \
		done; \
		if [ "$$found" -eq 0 ]; then \
			echo "  WARNING: no test cases found in $$tess_dir/test_cases/ (add *.json files to test this component)"; \
		fi; \
	}; \
	if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Testing all components and app..."; \
		for dir in components/tesseracts/*/; do \
			if [ -d "$$dir" ] && [ "$$(basename $$dir)" != ".template" ]; then \
				echo "Testing $$(basename $$dir)..."; \
				run_tesseract_tests "$${dir%/}" || exit 1; \
			fi; \
		done; \
		echo "Testing app..."; \
		python -m pytest app; \
	elif [ "$(filter-out $@,$(MAKECMDGOALS))" = "app" ]; then \
		echo "Testing app only..."; \
		python -m pytest app; \
	else \
		echo "Testing tesseract: $(filter-out $@,$(MAKECMDGOALS))"; \
		TESS_DIR="components/tesseracts/$(filter-out $@,$(MAKECMDGOALS))"; \
		if [ -d "$$TESS_DIR" ]; then \
			run_tesseract_tests "$$TESS_DIR" || exit 1; \
		else \
			echo "Error: Tesseract $(filter-out $@,$(MAKECMDGOALS)) not found"; \
			exit 1; \
		fi; \
	fi

gen-tests:
	@set -e; \
	TESS_NAME="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$TESS_NAME" ]; then \
		echo "Usage: make gen-tests <mytess> FILE=path/to/payload.json [ENDPOINT=apply] [OUT=name.json]"; \
		echo "  FILE holds a payload, e.g. {\"inputs\": {...}}. The endpoint is run and its"; \
		echo "  output is captured into a ready-to-run test case under the component's test_cases/."; \
		exit 1; \
	fi; \
	if [ -z "$(FILE)" ]; then \
		echo "Error: FILE=path/to/payload.json is required"; \
		exit 1; \
	fi; \
	TESS_SLUG=$$(echo "$$TESS_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr '-' '_'); \
	TESS_DIR="components/tesseracts/$$TESS_SLUG"; \
	if [ ! -d "$$TESS_DIR" ]; then \
		echo "Error: Tesseract $$TESS_SLUG not found (create it first with 'make new $$TESS_SLUG')"; \
		exit 1; \
	fi; \
	TESS_IMAGE="prismo_$$TESS_SLUG"; \
	if ! COLUMNS=1000 tesseract list 2>/dev/null | grep -qw "$$TESS_IMAGE"; then \
		echo "Error: Tesseract image $$TESS_IMAGE is not built (build it first with 'make build $$TESS_SLUG')"; \
		exit 1; \
	fi; \
	ENDPOINT="$(ENDPOINT)"; [ -n "$$ENDPOINT" ] || ENDPOINT="apply"; \
	OUT="$(OUT)"; [ -n "$$OUT" ] || OUT="$$ENDPOINT.json"; \
	OUT_PATH="$$TESS_DIR/test_cases/$$OUT"; \
	echo "Capturing test case for prismo_$$TESS_SLUG ($$ENDPOINT) -> $$OUT_PATH"; \
	OUTPUTS=$$(tesseract run prismo_$$TESS_SLUG "$$ENDPOINT" @$(FILE)); \
	mkdir -p "$$TESS_DIR/test_cases"; \
	ENDPOINT="$$ENDPOINT" PAYLOAD_FILE="$(FILE)" OUTPUTS="$$OUTPUTS" \
		python scripts/gen_test_case.py > "$$OUT_PATH"; \
	echo "Wrote $$OUT_PATH (review it, then 'make test $$TESS_SLUG')"

data:
	@echo "Pulling data..."
	@bash data/get_data.sh

run:
	@echo "Running app..."
	@PYTHONPATH=app:components/shared_code:$${PYTHONPATH} $(PYTHON) -m prismo.main $(if $(RUN_ARGS),$(RUN_ARGS),$(filter-out $@,$(MAKECMDGOALS)))

run-containers:
	@echo "Running app with Docker containers..."
	@PYTHONPATH=app:components/shared_code:$${PYTHONPATH} $(PYTHON) -m prismo.main --use-containers $(if $(RUN_ARGS),$(RUN_ARGS),$(filter-out $@,$(MAKECMDGOALS)))

clean:
	@echo "Cleaning build artifacts, caches, and temp files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage .coverage.* 2>/dev/null || true
	rm -rf run_*/

# Allow make to accept tesseract names as targets without errors
%:
	@:
