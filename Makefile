PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install lint test prototype synth prepare features backtest train monitor clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## create a virtualenv and install the project with dev extras
	$(PY) -m venv $(VENV) && $(BIN)/pip install -U pip && $(BIN)/pip install -e ".[dev]"

install: ## install into the active interpreter
	pip install -e ".[dev]"

lint: ## ruff lint + format check
	$(BIN)/ruff check src tests pipelines scripts && $(BIN)/ruff format --check src tests pipelines scripts

test: ## run the unit tests
	$(BIN)/pytest

prototype: ## full offline prototype: synth -> prepare -> features -> backtest -> train
	$(BIN)/leadtime run-all

synth: ; $(BIN)/leadtime synth
prepare: ; $(BIN)/leadtime prepare
features: ; $(BIN)/leadtime features
backtest: ; $(BIN)/leadtime backtest
train: ; $(BIN)/leadtime train
monitor: ## drift report: reference window ends 2018-06-01
	$(BIN)/leadtime monitor --reference-end 2018-06-01

olist: ## download the real Olist CSVs from Kaggle into data/raw
	$(BIN)/python scripts/download_olist.py --out data/raw

clean: ## remove generated data, models and reports
	rm -rf data/raw/*.csv data/processed/*.parquet models reports/*.parquet reports/*.csv reports/*.json reports/*.png reports/*.md mlruns lake
