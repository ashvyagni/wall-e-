PY := PYTHONPATH=src .venv/bin/python
VENV := .venv/bin

.PHONY: install test lint typecheck schemas check run sample viewer-install viewer-dev

install:
	uv pip install -p .venv/bin/python -e ".[dev,api]"
	@echo "NOTE: if imports fail with ModuleNotFoundError, run: chflags nohidden .venv/lib/python3.12/site-packages/*.pth"

test:
	$(PY) -m pytest

lint:
	$(VENV)/ruff check src tests scripts api bench

typecheck:
	$(VENV)/mypy src

schemas:
	$(PY) scripts/export_schemas.py

check: lint typecheck test
	cd contracts/ts && npm test && npm run typecheck

run:
	$(PY) -m walle.cli run $(CONFIG)

sample:
	$(PY) -m walle.cli run configs/sample_basic.json --out traces --reports reports

viewer-install:
	cd viewer && npm install --no-audit --no-fund

viewer-dev:
	cd viewer && npm run dev
