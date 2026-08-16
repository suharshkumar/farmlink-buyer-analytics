.PHONY: setup test run clean

setup:
	python3 -m venv .venv && ./.venv/bin/pip install -U pip && ./.venv/bin/pip install -r requirements.txt

test:
	./.venv/bin/python -m pytest tests/ -q

run:
	./.venv/bin/python scripts/run_analysis.py

clean:
	rm -rf data/raw/* data/processed/* reports/results/*
