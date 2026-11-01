.PHONY: help install dev test test-py test-ts lint lint-py lint-ts type-check docker-up docker-down clean

PYTHON ?= .venv/bin/python
RUFF ?= ruff
MYPY ?= mypy

help:
	@echo "Smart Vision AI Build & Quality System:"
	@echo "  make install     - Install Python and Node dependencies"
	@echo "  make dev         - Start development servers"
	@echo "  make test        - Run both Python and TypeScript test suites"
	@echo "  make test-py     - Run all Python core & integration unit tests"
	@echo "  make test-ts     - Run all TypeScript unit and integration tests"
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

test-ts:
	npm test

lint: lint-py lint-ts

lint-py:
	$(RUFF) check core

lint-ts:
	npm run lint

type-check:
	$(MYPY) core --ignore-missing-imports
	npm run type-check

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .mypy_cache .ruff_cache *.db
