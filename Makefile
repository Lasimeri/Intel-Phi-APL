# Makefile: the one entry point of the APL engine. Each target is a thin
# call into scripts/ (bash), so every step also runs on its own and every
# rule lives in one place. `make` alone prints the targets.
#
#   make doctor          check toolchains, card, shared region, models
#   make host            host tools + host builds of every kernel (bit-exact models)
#   make card            VPU kernels (KNC clang) + card programs (knc-cc), ISA-audited
#   make check           host tests, then every gate on the card (SKIP, exit 77, if it is down)
#   make compdb          compile_commands.json for clangd, verified 0 diagnostics per file
#   make engine MODEL=X  compile qwen35.apl for a model, build both halves (default the 2B)
#   make run MODEL=X     tok/s: the card + the host's worker, 128 tokens (HOST_THREADS=1..14, default 4, the measured best)
#   make run-card MODEL=X  the same on the card alone, for comparison
#   make baseline        llama.cpp on the host CPU: tok/s + reference tokens/logits per model
#   make verify-models   sha256sum -c over input/models (11.86 GB, reads every byte)
#   make clean           remove out/
include config.mk

.PHONY: help doctor host card all check compdb engine run run-card baseline verify-models clean

MODEL ?= Qwen3.8-2B-Q8_0
HOST_THREADS ?= 4

help:
	@sed -n 's/^#   make/  make/p' Makefile

doctor:
	@scripts/doctor.sh

host:
	@scripts/build.sh host

card:
	@scripts/build.sh card

all:
	@scripts/build.sh host card

check: all
	@scripts/check.sh

compdb:
	@scripts/compdb.sh

engine: all
	@scripts/engine.sh compile $(MODEL)
	@scripts/engine.sh build $(MODEL)

run: engine
	@scripts/engine.sh split $(MODEL) 128 2 59 $(HOST_THREADS) 3

run-card: engine
	@scripts/engine.sh card $(MODEL) 128 2 59 3

baseline: host
	@scripts/baseline.sh

verify-models:
	@cd input/models && sha256sum -c SHA256SUMS

clean:
	rm -rf out
