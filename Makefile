PYTHON        ?= python3.13
VENV          ?= .venv
PY            := $(VENV)/bin/python

MODEL         ?= bdh
MODELS        := $(patsubst models/%/model.mk,%,$(wildcard models/*/model.mk))

DEVICE        ?= auto
ITERS         ?=
TOKENS        ?= 400
TEMP          ?= 0.8
TOPK          ?= 20
SAMPLES       ?= 1
PORT          ?= 8000

export PYTHONUNBUFFERED = 1

# Per-model defaults (PRESET, PROMPT) and the `model-setup` target (upstream code, datasets).
include models/$(MODEL)/model.mk

TRAIN_ARGS    := --preset $(PRESET) --device $(DEVICE) $(if $(ITERS),--iters $(ITERS))

.PHONY: help models all setup venv model-setup train sample export web test clean distclean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo "\nVariables: MODEL=$(MODEL) PRESET=$(PRESET) DEVICE=auto|cpu|mps|cuda ITERS=N PROMPT=... TOKENS=N TEMP=x TOPK=N SAMPLES=N"
	@echo "Models: $(MODELS)"

models: ## List available models
	@echo $(MODELS)

all: setup train sample export ## Whole pipeline for MODEL: setup, train PRESET, sample, export to web/MODEL/

setup: venv model-setup ## Create venv, install deps, fetch MODEL's upstream code and data

venv: $(PY)
$(PY): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt
	@touch $(PY)

train: setup ## Train MODEL/PRESET -> runs/MODEL/PRESET/
	$(PY) -m models.$(MODEL).train $(TRAIN_ARGS)

sample: setup ## Generate text from runs/MODEL/PRESET/checkpoint.pt
	$(PY) -m models.$(MODEL).sample --preset $(PRESET) --device $(DEVICE) \
		--prompt "$$(printf '$(PROMPT)')" --tokens $(TOKENS) --temperature $(TEMP) --top-k $(TOPK) --num-samples $(SAMPLES)

export: setup ## Export runs/MODEL/PRESET/checkpoint.pt to web/MODEL/model/ for the browser demo
	$(PY) -m models.$(MODEL).export --preset $(PRESET)

web: ## Serve the browser demos on http://localhost:PORT
	@echo "Open http://localhost:$(PORT)"
	$(PYTHON) -m http.server $(PORT) --directory web

test: setup ## Run tests for MODEL (CPU smoke tests, JS/PyTorch parity)
	$(PY) -m pytest -q tests/$(MODEL)

clean: ## Remove training runs of MODEL
	rm -rf runs/$(MODEL)

distclean: ## Remove all runs, venv, vendor code and datasets
	rm -rf runs $(VENV) vendor data .pytest_cache
