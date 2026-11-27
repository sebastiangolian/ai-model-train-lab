# BDH (Dragon Hatchling): upstream code pinned to a commit of pathwaycom/bdh, Tiny Shakespeare dataset.
PRESET        ?= small
PROMPT        ?= ROMEO:\n

BDH_UPSTREAM_URL ?= https://github.com/pathwaycom/bdh
BDH_UPSTREAM_REF ?= 400e1d23a1f3680297383c2f58b3ec420fa3a2fe

model-setup: vendor/bdh/bdh.py
	$(PY) -m models.bdh.data

vendor/bdh/bdh.py:
	rm -rf vendor/bdh
	git clone --quiet $(BDH_UPSTREAM_URL) vendor/bdh
	git -C vendor/bdh -c advice.detachedHead=false checkout --quiet $(BDH_UPSTREAM_REF)
