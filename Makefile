.PHONY: help install dev test test-py test-ts test-core lint lint-py lint-ts lint-core type-check type-check-core load-test chaos-test mutation-test docker-up docker-down clean

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
RUFF ?= $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
MYPY ?= $(if $(wildcard .venv/bin/mypy),.venv/bin/mypy,mypy)

help:
	@echo "Smart Vision AI Build & Quality System:"
	@echo "  make install     - Install Python and Node dependencies"
	@echo "  make dev         - Start development servers"
	@echo "  make test        - Run both Python and TypeScript test suites"
	@echo "  make test-py     - Run all Python core & integration unit tests"
	@echo "  make test-ts     - Run all TypeScript unit and integration tests"
	@echo "  make load-test   - Run 64 synthetic camera streams and report latency percentiles"
	@echo "  make chaos-test  - Run the synthetic streams with injected faults"
	@echo "  make mutation-test - Run Cosmic Ray mutation testing (requires cosmic-ray)"
	@echo "  make lint        - Run Ruff and ESLint"
	@echo "  make type-check  - Run Mypy and TypeScript compiler checks"
	@echo "  make docker-up   - Start PostgreSQL, Redis, FastAPI, and Web containers"
	@echo "  make docker-down - Stop Docker Compose stack"
	@echo "  make clean       - Remove cached bytecode, logs, and build artifacts"

install:
	pip install -r requirements.txt
	npm install

dev:
	docker compose up -d db redis
	npm run dev

test: test-py test-ts

test-py:
	$(PYTHON) -m unittest discover -s core/tests -v

test-core: test-py

test-ts:
	npm test

lint: lint-py lint-ts

lint-py:
	$(RUFF) check core

lint-core: lint-py

lint-ts:
	npm run lint

type-check:
	$(MYPY) core --ignore-missing-imports
	npm run type-check

type-check-core:
	$(MYPY) core --ignore-missing-imports

load-test:
	$(PYTHON) scripts/load_test_synthetic.py --cameras 64 --frames 300

chaos-test:
	$(PYTHON) scripts/load_test_synthetic.py --cameras 64 --frames 300 \
		--drop-every 7 --disconnect-every 23 --error-every 31 \
		--latency-spike-every 47 --latency-spike-seconds 0.05

MUTATION_CONFIG ?= cosmic-ray.toml
MUTATION_SESSION ?= mutation-session.sqlite

mutation-test:
	$(PYTHON) -m cosmic_ray.cli init --force $(MUTATION_CONFIG) $(MUTATION_SESSION)
	PATH="$(CURDIR)/.venv/bin:$$PATH" \
		$(PYTHON) -m cosmic_ray.cli exec $(MUTATION_CONFIG) $(MUTATION_SESSION)
	$(PYTHON) scripts/mutation_report.py $(MUTATION_SESSION)

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .mypy_cache .ruff_cache *.db
