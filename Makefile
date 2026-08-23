.PHONY: help install new build julia-base images test gen-tests run run-containers validate-gradient validate-gradient-containers probe-objective probe-objective-containers animate figures benchmark docs clean

PYTHON ?= python

help:
	@echo "Setup"
	@echo "  install                       - pip install the host app + shared schemas into the active Python env"
	@echo "  julia-base chargetransport    - Build the Julia base image (once; redo only when julia_env/*.toml change)"
	@echo "  build [NAME]                  - tesseract build all components, or one (make build gyptis)"
	@echo "  images                        - Show whether each component image is older than its sources (STALE = rebuild)"
	@echo "Run (containers = the real solvers)"
	@echo "  run-containers                - The optimization; figures + checkpoint.json in outputs/ (RUN_ARGS=\"--loss-weight 1e-5\")"
	@echo "  validate-gradient-containers  - Composed adjoint vs central finite differences"
	@echo "  probe-objective-containers    - Line-scan the objective along one direction (smoothness probe)"
	@echo "  animate                       - Rebuild doping_evolution.{gif,mp4} from outputs/checkpoint.json (no solver)"
	@echo "  figures                       - Snapshot outputs/*.pdf (+ gif) as PNGs into docs/figures/ for the README"
	@echo "  run | validate-gradient | probe-objective  - Same, but in-process (needs both solvers importable; otherwise raises)"
	@echo "Develop"
	@echo "  test [NAME|app]               - Component regression cases + app unit tests, one component, or app only"
	@echo "  new NAME [RECIPE=jax|pytorch] - Scaffold a new Tesseract component"
	@echo "  gen-tests NAME FILE=case.json - Capture a component test case from an input payload"
	@echo "  benchmark                     - Record cold/warm multiphysics callback timings"
	@echo "  docs                          - Build the Sphinx docs into docs/_build/html"
	@echo "  clean                         - Remove build artifacts, caches, and temp files"

install:
	@echo "Installing prismo_shared and the host app (editable)..."
	@$(PYTHON) -m pip install -e components/shared_code -e "app[dev]"

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

# Image staleness check: an image is STALE when a commit touching
# its component directory (or components/shared_code) is newer than the image.
images:
	@for dir in components/tesseracts/*/; do \
		name=$$(basename $$dir); \
		[ "$$name" != ".template" ] || continue; \
		image="prismo_$$name:latest"; \
		built=$$(docker image inspect --format '{{.Created}}' "$$image" 2>/dev/null | cut -c1-19); \
		if [ -z "$$built" ]; then \
			printf '%-32s not built\n' "$$image"; \
			continue; \
		fi; \
		changed=$$(git log -1 --format='%cI' -- "$$dir" components/shared_code); \
		commit=$$(git log -1 --format='%h' -- "$$dir" components/shared_code); \
		built_s=$$(date -d "$${built}Z" +%s 2>/dev/null || date -u -j -f '%Y-%m-%dT%H:%M:%S' "$$built" +%s 2>/dev/null || echo ""); \
		changed_s=$$(date -d "$$changed" +%s 2>/dev/null || date -j -f '%Y-%m-%dT%H:%M:%S%z' "$$(echo $$changed | sed 's/:\([0-9][0-9]\)$$/\1/')" +%s 2>/dev/null || echo ""); \
		if [ -z "$$built_s" ] || [ -z "$$changed_s" ]; then verdict=UNKNOWN; \
		elif [ "$$built_s" -ge "$$changed_s" ]; then verdict=OK; else verdict=STALE; fi; \
		printf '%-32s built %sZ  sources last changed %s (%s)  %s\n' "$$image" "$$built" "$$changed" "$$commit" "$$verdict"; \
	done

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
		$(PYTHON) -m pytest app; \
	elif [ "$(filter-out $@,$(MAKECMDGOALS))" = "app" ]; then \
		echo "Testing app only..."; \
		$(PYTHON) -m pytest app; \
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

run:
	@echo "Running app..."
	@PYTHONPATH=app:components/shared_code:$${PYTHONPATH} $(PYTHON) -m prismo.main run $(if $(RUN_ARGS),$(RUN_ARGS),$(filter-out $@,$(MAKECMDGOALS)))

run-containers:
	@echo "Running app with Docker containers..."
	@PYTHONPATH=app:components/shared_code:$${PYTHONPATH} $(PYTHON) -m prismo.main run --use-containers $(if $(RUN_ARGS),$(RUN_ARGS),$(filter-out $@,$(MAKECMDGOALS)))

validate-gradient:
	@echo "Validating composed gradient (adjoint vs finite differences)..."
	@PYTHONPATH=app:components/shared_code:$${PYTHONPATH} $(PYTHON) -m prismo.main validate-gradient $(if $(RUN_ARGS),$(RUN_ARGS),$(filter-out $@,$(MAKECMDGOALS)))

validate-gradient-containers:
	@echo "Validating composed gradient across the real CT + gyptis boundary..."
	@PYTHONPATH=app:components/shared_code:$${PYTHONPATH} $(PYTHON) -m prismo.main validate-gradient --use-containers $(if $(RUN_ARGS),$(RUN_ARGS),$(filter-out $@,$(MAKECMDGOALS)))

probe-objective:
	@echo "Scanning the objective along one direction..."
	@PYTHONPATH=app:components/shared_code:$${PYTHONPATH} $(PYTHON) -m prismo.main probe-objective $(if $(RUN_ARGS),$(RUN_ARGS),$(filter-out $@,$(MAKECMDGOALS)))

probe-objective-containers:
	@echo "Scanning the objective across the real CT + gyptis boundary..."
	@PYTHONPATH=app:components/shared_code:$${PYTHONPATH} $(PYTHON) -m prismo.main probe-objective --use-containers $(if $(RUN_ARGS),$(RUN_ARGS),$(filter-out $@,$(MAKECMDGOALS)))

animate:
	@echo "Animating the doping field from the checkpoint..."
	@PYTHONPATH=app:components/shared_code:$${PYTHONPATH} $(PYTHON) -m prismo.main animate $(if $(RUN_ARGS),$(RUN_ARGS),$(filter-out $@,$(MAKECMDGOALS)))

# README figures: rasterize the PDFs of the last run and shrink the animation
# (GitHub renders PNG/GIF inline; the pre-commit size cap is 3 MB).
figures:
	@mkdir -p docs/figures
	@for f in outputs/*.pdf; do \
		[ -f "$$f" ] || continue; \
		pdftoppm -png -r 150 -singlefile "$$f" "docs/figures/$$(basename $${f%.pdf})"; \
	done
	@if [ -f outputs/doping_evolution.mp4 ]; then \
		ffmpeg -v error -y -i outputs/doping_evolution.mp4 \
			-vf "fps=8,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
			docs/figures/doping_evolution.gif; \
	fi
	@ls -la docs/figures

docs:
	@$(MAKE) -C docs html

benchmark:
	@echo "Benchmarking multiphysics optimization callbacks..."
	@PYTHONPATH=app:components/shared_code:$${PYTHONPATH} $(PYTHON) scripts/benchmark_multiphysics_optimization.py $(BENCHMARK_ARGS)

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
