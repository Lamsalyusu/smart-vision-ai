.PHONY: help install run test test-core lint-core type-check-core coverage benchmark proto-python docker-build docker-up docker-down clean

PYTHON ?= .venv/bin/python
RUFF ?= ruff
MYPY ?= mypy

help:
	@echo "SmartVision AI Developer & Evaluation Commands:"
	@echo "  make install     - Install requirements and setup environment"
	@echo "  make run         - Start the local development server (port 8000)"
	@echo "  make test        - Run the unit test suite (33 tests)"
	@echo "  make test-core   - Run the architecture core unit tests"
	@echo "  make lint-core   - Lint core Python modules"
	@echo "  make type-check-core - Strictly type-check dependency-light domain code"
	@echo "  make coverage    - Run tests with coverage and print report"
	@echo "  make benchmark   - Run latency and throughput profiler"
	@echo "  make docker-up   - Build and start containerized application"
	@echo "  make docker-down - Stop Docker containers"
	@echo "  make clean       - Remove cached bytecode and temporary files"

install:
	pip install -r requirements.txt

run:
	$(PYTHON) manage.py runserver 0.0.0.0:8000

test:
	$(PYTHON) manage.py test

test-core:
	$(PYTHON) -m unittest discover -s core/tests -q

lint-core:
	$(RUFF) check core

type-check-core:
	$(MYPY) --strict --disable-error-code no-any-return core/domain/perception core/domain/tracking/state.py core/domain/tracking/association.py

coverage:
	$(PYTHON) -m coverage run manage.py test
	$(PYTHON) -m coverage report
	$(PYTHON) -m coverage report > model_evaluation/test_coverage_report.txt

coverage-html:
	$(PYTHON) -m coverage run manage.py test
	$(PYTHON) -m coverage html
	$(PYTHON) -m coverage report > model_evaluation/test_coverage_report.txt

badge:
	./.venv/bin/coverage-badge -o coverage-badge.svg -f

lint:
	$(PYTHON) -m ruff check .

type-check:
	$(PYTHON) -m mypy .

benchmark:
	$(PYTHON) model_evaluation/benchmark_pipeline.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f .coverage
