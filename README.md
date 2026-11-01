# Intel-Phi-APL

An inference engine whose model is written in **APL** (Kenneth E. Iverson's array
language, ISO/IEC 13751), compiled by a compiler that is itself written in APL, and whose
weight matrix products run on the **vector processing unit of an Intel Xeon Phi 7120A**
(Knights Corner, KNC), with the rest on the host's Ryzen 7 5800X.

Three rules shape everything:

- **The model's math is only in `src/apl/qwen35.apl`.** C holds primitives, transport and
  I/O, never a forward pass.
- **Every product of a Q8_0 weight runs on the KNC VPU, in F32 in the 512-bit registers.**
  Quantized weights enter only as up-converting memory operands. Products over the host's own
  arrays (the attention history, the delta rule's state) run on the host, where they measured
  faster than the VPU.
- **Everything is held to a reference.** GNU APL runs the same source in F64 as the test
  reference; llama.cpp is the external one.

Where things live:

| What | Where | Size, 9B |
|---|---|---|
| Q8_0 weights | streamed from the NVMe at load, O_DIRECT, then resident on the card | 9,512,288,256 B |
| Delta rule state `S` | host RAM, 6 levels | 301,989,888 B (arithmetic: 6 x 32 x 128 x 128 x 4 x 24 layers) |
| KV cache | host RAM, in the pinned shared region | 75,497,472 B reserved (`CTX` 1024) |
| Q8_0 bytes in host RAM | none | 0 B |
| Host process | peak RSS | 389.7 MB |

This needs the hardware: a Xeon Phi 7120A, its stack, and a Linux host with AVX2. The code is
published as a record of how far APL on KNC can be taken, with every number traceable to a log.

## Documentation

| Guide | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | how a token travels; what runs where and why; one layer as a conversation |
| [docs/model.md](docs/model.md) | `qwen35.apl` statement by statement, with shapes and constants |
| [docs/compiler.md](docs/compiler.md) | every pass, the table, placement, rounds, fusion, adding a primitive |
| [docs/runtime.md](docs/runtime.md) | form I tiles, `gemv` and `gemm`, the link, every setting |
| [docs/testing.md](docs/testing.md) | every gate, what it proves, what is not covered |
| [docs/measurements.md](docs/measurements.md) | the numbers over time, their records, everything tried and refused |

## Measured: tokens per second

GGUF models, 40-token prompt, 128 tokens generated, greedy, load excluded. **Alternating with
llama.cpp in one window**, 2026-11-01 14:46-14:54: two sessions each, 3 repeats each, all six
listed by range. Engine: card threads 2 per core in decode, 3 in the prompt, on cores 0-58;
4 host threads; one draft per pass (the default). llama.cpp: 8 threads on the 5800X, CPU only.
Another process used about 240 % of a CPU on the host throughout, and two others about 15 % each
were seen just after the window.

| Decode, tok/s | llama.cpp | Engine | Ratio of means |
|---|---|---|---|
| Qwen3.8-2B-Q8_0 | 14.126-14.570, mean 14.351 | **65.671-67.343, mean 66.626** | 4.64x |
| Qwen3.8-9B-Q8_0 | 3.324-3.526, mean 3.436 | **18.716-20.143, mean 19.642** | 5.72x |

| Prompt, tok/s | llama.cpp | Engine | Ratio of means |
|---|---|---|---|
| Qwen3.8-2B-Q8_0 | 98.84-124.76, mean 113.273 | 125.661-128.470, mean 126.875 | 1.12x |
| Qwen3.8-9B-Q8_0 | 23.51-28.63, mean 26.658 | 33.755-35.101, mean 34.310 | 1.29x |

llama.cpp's decode was 9 to 11 % below its own numbers that morning (2B 15.85-15.94, 9B
3.87-3.89), cause not verified. Against those morning numbers the ratios would be about 4.2x
and 5.1x (arithmetic).

Correctness: every engine run generated 128 of 128 tokens identical to llama.cpp's. Top-1 the
same and top 10 overlapping as `tests/host/logit_cmp` requires (NRMSE 0.0215 on the 2B, where
llama.cpp's own 8-bit activations account for the gap). Every layer within 1.099e-05 of the
F64 APL reference (limit 1e-2).

Record: `logs/engine/2026-11-01/144406-final-interleaved-fusion.txt`.

### Speculative decoding

Both models already carry a draft head (`blk.N.nextn.*`, the MTP block). The engine drafts K
tokens one at a time, then the trunk checks them in one pass of K+1 tokens. Only the tokens the
trunk agrees with stand, so the output is the same at any K; that is gated. The accept rule is
APL: `Z←+/∧\DR=AG`. A draft reads only the 32,768 commonest tokens' rows of the output matrix;
that cost one accepted token in 128 on the 9B (56/71 against 57/70) and saved most of that step.

| K | 9B decode, tok/s | 9B accepted | 2B decode, tok/s | 2B accepted |
|---|---|---|---|---|
| 0 | 15.040, 15.494, 15.709 | | 53.737, 49.942 | |
| **1** | **20.029, 20.157, 20.209** | 56/71, 78.9 % | **64.414, 64.235** | 57/69, 82.6 % |
| 2 | 18.218, 18.465 (build before fusion) | 72/108, 66.7 % | 65.945, 63.477 | 76/100, 76.0 % |
| 3 | 17.647 (build before fusion) | 80/139, 57.6 % | | |

Records: `142457-fusion-9B.txt`, `133719-mtp-9B-c.txt`, `144209-fusion-2B.txt`.

### Where a 9B token's time goes

Before fusion, K=0, `PROFILE=1`, 64.669 ms/token:

| What | ms/token | Share |
|---|---|---|
| `gemv`, matrices of 512 rows or more | 52.619 | 81.4 % |
| Waiting for the host | 8.534 | 13.2 % |
| DMA copies | 2.096 | 3.2 % |

Decode is memory-bound on the card: 8.43 GB read per token against a measured STREAM of
177.9 GB/s, a ceiling of about 18.4 tok/s without speculation (arithmetic). The host's delta
regions take 5.07-5.18 ms/token on 4 threads and 6.01-6.10 on one: only 15 % from four times
the threads, because they are bound by host DRAM (96 MiB of `S` per token, about 19.6 GB/s,
arithmetic), while their arithmetic alone needs about 0.3 ms (arithmetic).

The prompt, at about 29 ms/token: `gemm` 77 to 80 %, waiting for the host 14 to 16 %. At 16
tokens a pass it needs only about 23 GB/s (arithmetic), so it is not memory-bound, and it runs
at about 30 % of the F32 peak of 2416.76 GFLOPS (arithmetic: 61 cores x 1238.094 MHz x 16 lanes
x 2). That is the largest gap left.

## How it works

```
src/apl/qwen35.apl
  tk.apl    tokens, branchless masks
  ps.apl    Bunda-Gerth pair reduction: the tree as an inverted table
  fl.apl    flat table of operations, execution order
  in.apl    inline; ranked functions stay as map regions; names resolved by sorting
  sh.apl    shapes and constant folding by executing in the interpreter; the whitelist
  sc.apl    row order: rows that touch one state stay together
  em.apl    frame places by last use; placement; parallel maps; open rounds; C text
  fz.apl    fusion: a host chain of elementwise rows as one generated loop
  main.apl  one model's tensor index -> out/gen/MODEL/model_card.c, model_card_host.c
```

The compiler runs in GNU APL on the host, in the architecture of Hsu's Co-dfns: a table,
whole-array passes. The compile stage replaces run-time interpretation: names, shapes, constants,
placement and scheduling are settled once, and the card only executes what they reduce to.

- **Placement** is a two-state dynamic program over the table: Q8_0 products are forced to the
  card, scalars, map regions and the states to the host, and each free row goes where it is
  cheapest, a change of side costing what a crossing measured.
- **Frames are mirrored**: the host's lies in the shared region, the card's in card memory, at
  the same offsets, so a crossing is "copy this range".
- **Batches**: the same `qwen35.apl` runs NB tokens at once, for NB = 1, 2, 3, 4, 8, 16. One pass
  over a matrix serves every token (`gemm`). Decode is NB=1; a speculative check is NB=K+1.
- **State by position**: `S`, the conv ring and the KV cache are indexed by position. Taking
  back tokens tried ahead means starting again from an earlier position, with nothing to copy.
- **Open rounds**: where a phase begins with products, the card queues them before it waits for
  the host, and its threads read 128 KiB ahead through the wait.
- **Parallel maps**: a map the compiler proves independent runs on the host's threads.

### Fusion

Every elementwise row used to be its own pass with its own store: over 40 in one delta layer.
`fz.apl` makes each chain one loop, each row a line of a per-lane sequence. A value all of whose
readers are in the chain never reaches memory. A map's cell goes straight into its row, a
depthwise conv is one node, and `700⌊` under `exp`, there only for GNU APL's F64, is an alias.

| One 9B delta layer, `S` rows excluded | Before | After |
|---|---|---|
| Lanes stored | 364,928 (sum of the generated calls, arithmetic) | 78,656 (counted from the generated C), -78.4 % |
| Host primitives | `rt_ew` and 22 op cases, `rt_s_*`, `rt_redfirst`, ... | deleted; loops generated |

The fused build's layer outputs and logits are identical to the previous build's, bit for bit, in
25 of 25 dump files (2B, `-ffp-contract=off`). 9B, alternating, 128/128: K=1 19.771, 19.491,
19.643 to 20.029, 20.157, 20.209, +2.5 %; prompt 34.083 to 35.271, +3.5 %. Record:
`logs/engine/2026-11-01/142457-fusion-9B.txt`.

### What was tried and refused

Measured dead, so not to be proposed again without new evidence:

| Idea | Result | Record |
|---|---|---|
| Skipping silenced arms in the products | FFN `silu(g)` 16-row groups below 1e-2 of the max: 0.00 %; FFN 32-column blocks: 0.51 % | `140909-silence-stats.txt` |
| A decay shortcut for `S` | `E` geomean 0.78754, 60.92 % of heads at or above 0.95 | same |
| `gemv` deleted for `gemm` at T=1 | K=0 15.433 to 14.787, -4.2 %; K=1 tie | `143432-gemv-ablation-9B.txt` |
| A cache line of padding between the rows of `x` | bench +9.5 % at T=16; engine K=1 -0.9 %, prompt +2.3 % overlapping | `145919-gemm-xpad.txt`, `150252-xpad-9B.txt` |
| Prefetching the next tile's `x` | T=16 633 to 567 GFLOPS (older kernel) | `101153-gemm-pfx.txt` |
| Lossless packing of the weights | entropy 7.67-7.68 of 8 bits, zstd 1.040, lz4 1.000 | `120138-weight-compressibility.txt` |
| Low-rank weights, on the 3090 Ti | rank 1024 keeps 46.01-64.33 % of the energy, random control 45.60 % | `123414`, `123520` spectrum |
| EML (exp x - ln y) as the arithmetic | about 100x the instructions (arithmetic, not run) on a memory-bound decode | `input/refs/eml/` |

Streaming every `S` write cost the prompt 9.5 %: a later token of a batch reads what an earlier one
wrote. So only the last token of a batch streams (K=0 +2.1 %, K=1 +2.5 %, overlapping spreads).

## Using it

Needs: an Intel Xeon Phi 7120A served by the **Intel Phi 7120A stack** (`phi`, `phictl`,
`phi-isa-audit`, `toolchain/env.sh`), card up, `HOSTMEM 4G` in `~/.config/phi/cards`; the
**Intel-Phi-LLVM** KNC clang and the card musl sysroot; on the host `cc` (GCC 16.2.1),
`clangd` (23.1.4), bash, GNU make, a GNU APL 2.0 fixed as in `tools/gnu-apl/` (`APL_BIN`
overrides the interpreter; the unfixed 2.0-2 miscompiled under GCC 16); an AVX2 CPU; models
in `input/models/` (not in git).

The defaults in `config.mk` are the author's machine: set `PHI_TREE`, `PHI_LLVM_BIN` and
`PHI_SYSROOT_7120A` there, on the `make` line, or in the environment. The models are not in this
repository; `input/models/SHA256SUMS` names them.

```
make doctor                     # compilers, stack, card, shared region, models
make check                      # builds everything; host tests, then every gate on the card
make run                        # compile, build, run the 2B: tok/s with the host's worker
make run MODEL=Qwen3.8-9B-Q8_0  # the target model; the card's huge pages are raised and restored
make run HOST_THREADS=14        # more host threads (default 4)
make run-card                   # the card alone, for comparison
make baseline                   # llama.cpp on the host CPU: tok/s + reference tokens and logits
make compdb                     # compile_commands.json, verified 0 clangd diagnostics per file
make                            # every target
```

`APL_MTP=K` sets the drafts per pass (default 1, 0 turns speculation off, at most 3).
`APL_PROMPT_TPC` sets the card's threads per core in the prompt (default 3).

`make check` exits 0 when everything passed, 1 on any failure, 77 when the card did not
answer. Each run leaves a record with a provenance stamp in `logs/gates/DATE/`.

## How it is verified

Last run: 38 passed, 0 failed, 0 skipped (`logs/gates/2026-11-01/144415-check.txt`); clangd clean
over 49 files.

- **The compiler:** each stage's form is written back as APL and run with the source on a small
  synthetic model in GNU APL: original, rendered, flat and inlined forms identical over 2,515
  numbers; 3,214 real shapes against the inference, 0 mismatches (`tests/apl/parse_oracle.sh`,
  `shape_oracle.sh`).
- **Batches:** three tokens as one call, as 2 + 1 and one by one, identical (`batch_oracle.sh`).
- **The delta rule's rewrite:** `DeltaHead` takes Z from the old state, without reading the new.
  Against the rule as first written, on generic values at one point: 4.441e-16; state 0; a wrong
  form differs by 4.221, so the test can fail (`delta_witness.sh`). A sieve at one point, with
  the algebra behind it, not a proof.
- **The kernels:** written once against `src/card/knc.h`, which has KNC builtins and a lane-by-lane
  host model. Card == host model == independent reference, bit for bit; `gemm` == `gemv` of each
  token, bit for bit.
- **One real matrix, every path** (`tests/card/real_weight_gate.sh`).
- **The whole engine** (`tests/card/engine_gate.sh`), limits fixed before the first run: every
  layer within 1e-2 of the F64 APL reference (worst 1.099e-05); logits within NRMSE 1e-3 of it;
  llama.cpp's top-1, top 10 and every generated token, at K=0, 1 and 2; the prompt in batches of
  16, 4, 2 and 1.

The VPU's `vexp223ps`, `vlog2ps`, `vrcp23ps`, `vrsqrt23ps` (1.17, 1.04, 0.90, 0.70 ULP) and the
host's AVX2 functions are not correctly rounded, so whole-model runs are held to those tolerances,
not to bit-exactness.

## Open

- **The prompt at about 30 % of the F32 peak.** Laying out `x` differently has not carried over from
  the bench: see the table above. Next: find what stalls `tile8` in the engine itself, on the
  card's own counters if it has them. Heinecke et al. (IPDPS 2013) reach about 90 % of peak in DGEMM
  on KNC, without an up-converting load.
- **`S`'s host DRAM traffic, 96 MiB per token.** Deferring its writes (`S_base` plus pending
  `N`,`K` pairs, flushed every M tokens) would cut it about 44 % at M=16 and make rollback a
  count: about 21.3 tok/s on the 9B (estimate, unbuilt).
- **Decisions that are not the engine's to make:** ECC off (about +7 % decode, arithmetic, but a
  flash operation through Intel's signed module, not recommended); 4-bit FFN weights (the only
  remaining byte cut, lossy).
- Unexplained: prompt 38.2-38.5 tok/s on the 9B at 11:xx that day against about 35 now; the spread
  within a session (9B decode 18.716 to 20.143).
- **The KV cache's next form: an [Accelerated Square Array](https://github.com/gtgunar/Accelerated-square-Array)**
  (gtgunar). Today it is a fixed array of `CTX` 1024 positions. The plan: an upper vector of about
  √n references to lower vectors of about √n, so a peek stays O(1) and a mutation is O(√n), with no
  fixed reservation. Not built, so no numbers yet; see
  [docs/architecture.md](docs/architecture.md#the-kv-cache-and-the-accelerated-square-array).
- Sampling: greedy only. Rust monitoring TUI with a C backend: planned.
- `src/apl/ref_harness.apl` and `compiler/main.apl` each hold the constants and bindings.
- `phictl`'s forward looks to free no connection slot: after 32 forwarded connections since
  `phi up`, `phi sh` fails; the scripts use the control socket (`phi run`). Inferred from the
  journal and its client ports; the stack's source was not traced.

## Layout

```
Makefile, config.mk     entry point and path defaults
docs/                   the guides above
scripts/                lib.sh, doctor.sh, build.sh, check.sh, compdb.sh, baseline.sh,
                        apl_ref.sh (the F64 reference), engine.sh (compile, build, run)
src/apl/qwen35.apl      THE MODEL
src/apl/compiler/       the compiler, in APL
src/apl/gguf_read.apl, ref_harness.apl   the reference's reader and driver
src/card/               knc.h, team, gemv + gemm (matmul kernels), rt (primitives), stream, link,
                        cdma (copied, provenance inside), engine.c (the driver)
src/host/               gguf_reader, gguf_info, gguf_apl, apl_feed, ref_llama,
                        engine_host, rt_host + vec8.h (AVX2), link_host
src/gpu/                spectrum.c: offline weight analysis on the 3090 Ti, the record of a dead end
src/bench/              gemv_bench, real_gemv, cross_card, cross_host, mem_order, arena.h, bench_util.h
src/gate/               gate programs, kernels, references, judge
tests/                  apl/ (oracles), host/ (tools), card/ (gates), golden/
tools/gnu-apl/          the GNU APL 2.0 package and its fix
input/models/           Q8_0 models (gitignored) + SHA256SUMS
input/refs/             what this work cites + SHA256SUMS; the papers themselves are not in git
logs/                   run records
out/                    build outputs and generated C (gitignored, never hand-edited)
```

## References

Cited, and not redistributed here (see [input/refs/README.md](input/refs/README.md)): A. W. Hsu,
*Accelerating Information Experts through Compiler Design*, ARRAY 2015;
A. W. Hsu, *A data parallel compiler hosted on the GPU*, 2019; M. T. Goodrich, J. G. Kloss II,
*Tiered Vectors*; A. Odrzywołek, *All elementary functions from a single operator*, arXiv:2603.21852v2,
with a transcript of a3kino's video on it (its ablation rule shaped the fusion pass and the `gemv`
trial; its operator is not used).

Elsewhere: [Co-dfns](https://github.com/Co-dfns/Co-dfns) (AGPL-3.0, read for design only; no code
copied), [gtgunar/Accelerated-square-Array](https://github.com/gtgunar/Accelerated-square-Array),
[APL](https://en.wikipedia.org/wiki/APL_(programming_language)).

## License

No license has been chosen yet, so all rights are reserved for now. `src/card/cdma.*` is copied from the
same author's MIT-licensed Intel-Phi-Worldgen. `tools/gnu-apl/` packages and patches GNU APL (GPL).
