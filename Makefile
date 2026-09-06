# Convenience wrapper. Every target maps to a `mmrag` CLI command, so nothing
# here is required -- `make` is optional on Windows.
.PHONY: help install install-dev up down reset lint fmt type test corpus ingest

help:
	@echo "install      pip install -e . (core + openai provider)"
	@echo "install-dev  install with dev tooling"
	@echo "up           start Postgres + Qdrant"
	@echo "down         stop services (keeps data)"
	@echo "reset        stop services and WIPE all indexed data"
	@echo "lint/fmt     ruff check / ruff format"
	@echo "type         mypy"
	@echo "test         pytest (skips integration + llm markers)"
	@echo "corpus       download the PDF corpus from the manifest"

install:
	python -m pip install -e ".[openai]"

install-dev:
	python -m pip install -e ".[openai,dev]"

up:
	docker compose up -d

down:
	docker compose down

reset:
	docker compose down -v

lint:
	ruff check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

type:
	mypy

test:
	pytest -m "not integration and not llm and not slow"

corpus:
	mmrag corpus download
