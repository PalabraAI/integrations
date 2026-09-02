PACKAGES := openai-agents vapi jambonz wyoming
PYTHON   ?= python3

.PHONY: help venv install lint format test test-live build docker clean

help:
	@echo "make install    - create .venv and install every package in editable mode with dev extras"
	@echo "make lint       - ruff check + ruff format --check"
	@echo "make format     - ruff format + autofix"
	@echo "make test       - offline unit tests for every package"
	@echo "make test-live  - live tests against the Palabra API (needs PALABRA_API_KEY)"
	@echo "make build      - wheels + sdists into <package>/dist"
	@echo "make docker     - build the three bridge images"

venv:
	test -d .venv || $(PYTHON) -m venv .venv

install: venv
	. .venv/bin/activate && pip install -q -U pip && \
	  for p in $(PACKAGES); do pip install -q -e "./$$p[dev]"; done

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	@for p in $(PACKAGES); do echo "== $$p"; (cd $$p && $(PYTHON) -m pytest -q -m "not live") || exit 1; done

test-live:
	@test -n "$$PALABRA_API_KEY" || { echo "PALABRA_API_KEY is not set"; exit 1; }
	@for p in $(PACKAGES); do echo "== $$p"; (cd $$p && $(PYTHON) -m pytest -q -m live -s) || exit 1; done

build:
	@for p in $(PACKAGES); do (cd $$p && rm -rf dist && $(PYTHON) -m build -q) || exit 1; done

docker:
	docker build -t palabra-vapi-bridge ./vapi
	docker build -t palabra-jambonz-bridge ./jambonz
	docker build -t wyoming-palabra ./wyoming

clean:
	rm -rf .ruff_cache */dist */build */*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
