# nanoGPT: upstream model.py pinned to a commit of karpathy/nanoGPT, Tiny Shakespeare dataset (bytes).
PRESET        ?= small-long
PROMPT        ?= ROMEO:\n

NANOGPT_UPSTREAM_URL ?= https://github.com/karpathy/nanoGPT
NANOGPT_UPSTREAM_REF ?= 3adf61e154c3fe3fca428ad6bc3818b27a3b8291

model-setup: vendor/nanogpt/model.py
	$(PY) -m lab.data

vendor/nanogpt/model.py:
	rm -rf vendor/nanogpt
	git clone --quiet $(NANOGPT_UPSTREAM_URL) vendor/nanogpt
	git -C vendor/nanogpt -c advice.detachedHead=false checkout --quiet $(NANOGPT_UPSTREAM_REF)
