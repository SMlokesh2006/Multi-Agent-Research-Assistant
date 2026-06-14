.PHONY: dev cli eval test install lint

install:
	pip install -e ".[dev]"

dev:
	uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

cli:
	python -m src.cli

eval:
	python -m src.evaluation

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format src/ tests/
