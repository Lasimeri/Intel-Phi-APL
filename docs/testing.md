# How it is verified

`make check` builds everything, then runs host tests and every gate on the card. It exits 0 when everything
passed, 1 on any failure, 77 when the card did not answer. Each run leaves a record in `logs/gates/DATE/`,
starting with a provenance stamp. A full run takes about 3 minutes on the 2B.

The governing rule: **nothing is reported as working until it has run on the real card and its output has been
checked.** A passing compile is not verification.

## The compiler

| Test | What it does | Last result |
|---|---|---|
| `tests/apl/parse_oracle.sh` | writes each stage's form back as APL and runs it with the source on a small synthetic model in GNU APL | original == rendered == flat == inlined, 2,515 numbers |
| `tests/apl/shape_oracle.sh` | every array's real shape in GNU APL against the shape pass's inference | 826 rows, 3,214 lines, 0 mismatches, 0 rows never ran |
| `tests/apl/batch_oracle.sh` | three tokens as one call, as 2 + 1, and one by one, from zero state | worst difference 0 (limit 1e-12) |
| `tests/apl/delta_witness.sh` | `DeltaHead` against the delta rule as first written, on generic values | 4.441e-16; state 0; a wrong form differs by 4.221 |

`delta_witness.sh` uses the witness idea of arXiv:2603.21852 Sect. 2: a generally valid identity must hold at one
generic point, so evaluate both sides there. It is a sieve, not a proof, and rests on the algebra in
[model.md](model.md). The deliberately wrong form is there so the test can fail.

## The kernels

`src/card/knc.h` has each operation twice: KNC builtins, and a lane-by-lane host model. Every kernel is written once against it.

| Gate | Holds |
|---|---|
| Gate A (`src/gate/fma_*`) | FMA: card == host model == independent reference |
| Gate 2.1 (`src/gate/upconv_*`) | `sint8`, `uint8`, `1to16`, `float16` operands, every half pattern: card == reference, bit for bit |
| Alignment probes | an aligned operand runs; a misaligned one takes SIGSEGV as expected |
| `gemv_bench gemm` | `gemm` == one `gemv` per token, bit for bit, card and host model |
| `phi-isa-audit` | no instruction in any binary that KNC does not have |

## One real matrix

`tests/card/real_weight_gate.sh`, every path, 2B `token_embd.weight`: the file's sum on the card == on the host; card `y` ==
host-model `y` on all 248,320 lanes; within 1e-3 of the F64 reference (measured 4.764e-06).

## The whole engine

`tests/card/engine_gate.sh`, limits fixed before the first run. On the 2B:

| Check | Limit | Last result |
|---|---|---|
| Every layer against the F64 APL reference | 1e-2 | worst 1.099e-05, 24 layers |
| Logits against the F64 APL reference | NRMSE 1e-3 | passes |
| Logits against llama.cpp | `logit_cmp`: top-1, NRMSE 0.10, top-10 overlap 8 | same top-1, NRMSE 0.0215, overlap 10 |
| Generated tokens against llama.cpp, default K=1 | every one | 128 of 128 |
| The same at K=0 and K=2 | every one | 128 of 128 each |
| Every repeat generates the same tokens | yes | yes |
| Prompt in batches of at most 4, 2, 1 | 1e-2, same next token | worst 1.353e-05 at 1 |

The gap to llama.cpp is llama.cpp's own: its 8-bit activations account for the NRMSE. The F64 reference passed against llama.cpp
itself: top-1 357, NRMSE 0.0215, top 10 identical in order.

## Not bit-exact, on purpose

The VPU's `vexp223ps`, `vlog2ps`, `vrcp23ps`, `vrsqrt23ps` (1.17, 1.04, 0.90, 0.70 ULP) and the host's AVX2 functions are not
correctly rounded, so whole-model runs are held to the tolerances above. A change that should not move any number is held harder:
compare dumps from `scripts/engine.sh split MODEL N 2 59 4 1 DUMP_DIR` with `cmp`, as the fusion pass was (25 of 25 identical).

## What is not covered

- The layer-by-layer F64 reference exists for the 2B only. The 9B is held to llama.cpp's tokens in the runs recorded in `logs/engine/`, not in `make check`.
- K=3 is compiled and has been run with matching tokens, but is not gated.
- F32 subnormal behaviour of the VPU arithmetic: no gate reached that range.
- Sampling is not built; everything is greedy.

## Clean compile

`make compdb` writes `compile_commands.json` and verifies 0 clangd diagnostics per file. Builds use `-Wall -Wextra -Werror`.
