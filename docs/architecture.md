# Architecture

How a token travels through the engine, what runs where, and why. The model's math is in
[model.md](model.md), the compiler in [compiler.md](compiler.md), the kernels and settings in
[runtime.md](runtime.md).

```
╔═ compile, GNU APL on the desktop: src/apl/compiler/ ═════════════════════════════════════════╗
║ src/apl/qwen35.apl ─▶ tk ps fl in sh sc ─▶ em.apl (placement, rounds) ─▶ fz.apl (fusion)     ║
╚══════════╤═════════════════════════════════════════════════════════════╤═════════════════════╝
           ▼ model_card_host.c                                           ▼ model_card.c         
╔═ desktop: Ryzen 7 5800X, 4 threads ════════╗      ╔═ card: Xeon Phi 7120A, 59 cores ═════════╗
║ fz.apl loops: RMS SiLU conv, 1 store       ║──x──▶║ gemm T=2 checks, T=16 prompt, F32 zmm    ║
║ delta heads: S in RAM, 5.1 ms/token        ║      ║ gemv T=1, drafts over 32,768 tokens      ║
║ KV cache in the pinned shared region       ║◀──y──║ Q8_0 up-converting loads                 ║
║ K=1 accept in APL: Z←+/∧\DR=AG             ║      ║ 9,512,288,256 B weights resident         ║
║ peak RSS 389.7 MB (9B)                     ║      ║ 8.43 GB read per trunk pass (9B)         ║
╚════════════════════════════════════════════╝      ╚══════════════════════════════════════════╝
╔═ measured, card + host, 128 tokens, 128/128 == llama.cpp ════════════════════════════════════╗
║ 9B  decode 19.642 tok/s, llama.cpp 3.436, 5.72x   prompt 34.310, llama.cpp 26.658, 1.29x     ║
║ 2B  decode 66.626 tok/s, llama.cpp 14.351, 4.64x  prompt 126.875, llama.cpp 113.273, 1.12x   ║
║ gates: 38 passed, 0 failed, 0 skipped; clangd 49 files clean                                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

## The three programs

| Program | Runs on | Job |
|---|---|---|
| `out/card/engine-MODEL-split` (`src/card/engine.c` + generated `model_card.c`) | card | holds the Q8_0 weights, runs their products, follows the host's commands |
| `out/host/engine_host-MODEL` (`src/host/engine_host.c`, `rt_host.c` + generated `model_card_host.c`) | desktop | drives: knows the prompt, picks each next token, runs everything that is not a Q8_0 product |
| `out/host/apl_feed` | desktop | streams the model file from the NVMe to the card, once, at load |

`scripts/engine.sh split` starts all three in the right order. The compiler itself is not one of
them: it runs ahead of time, in GNU APL, and writes C.

## Why the card does not interpret APL

The compile stage removes the need. Names, shapes, constants, placement and scheduling are settled
once, ahead of time, so the card only executes the products they reduce to: 7 or 8 rows per layer.

GNU APL running the same `qwen35.apl` as the test reference took 586 s for 40 tokens on real 2B
weights, about 0.07 tok/s, against 127 tok/s compiled (host, F64, including its weight reads, so
this overstates what a tuned interpreter would cost). A thin interpreter over the products would
cost about 0.3 ms of a 65 ms token (arithmetic), and buy nothing: there is no model math on the card
to interpret, and recompiling takes about 5 s. This is also Co-dfns's choice. GNU APL does not run on
the card today.

## Where the data lives

| Data | Where | Why |
|---|---|---|
| Q8_0 weights | card memory, repacked to form I tiles | every byte is read every token; PCIe could not carry them |
| F32 weights (norms, `W_A`, `W_DT`, `W_CONV`) | host | read by host rows |
| Delta state `S`, conv ring `WIN` | host RAM | their rows measured faster on the host |
| KV cache `KC`, `VC` | host, pinned shared region | the attention products run next to it |
| Frame (temporaries) | mirrored: the same offsets on both sides | a crossing is "copy this range of the frame" |

The model file moves with O_DIRECT through a 256 MiB window of the shared region, at [768, 1024) MiB,
into card memory at 3.26 GB/s, and is repacked in place on the team at 7.22 GB/s. A 16-row group is the
same size in both forms, so there is never a second copy. It streams once; re-streaming on every token
would give about 0.34 tok/s (arithmetic) and is not built. The window's slots then hold the KV cache.

## The KV cache and the Accelerated Square Array

**Today** the KV cache is a plain fixed array: `KC` and `VC`, `[CTX HKV HD]` with `CTX` 1024, indexed by
position, in the pinned shared region. 75,497,472 B reserved on the 9B, 29,360,128 B on the 2B, whatever the
sequence's length. It stops at 1024 positions.

**The plan** is to take it to an [Accelerated Square Array](https://github.com/gtgunar/Accelerated-square-Array)
(gtgunar): instead of one vector, an upper vector of about √n references to lower vectors of about √n
elements. A peek stays O(1), and a mutation is O(√n), because it touches one lower vector and carries at
most √n changes through the rest. It belongs to the same family as Goodrich and Kloss's tiered vectors
(WADS 1999, doi:10.1007/3-540-48447-7_21).

What it would bring here:

- **No fixed reservation:** memory follows the sequence's real length, and contexts can pass 1024.
- **Natural blocks:** a lower vector can be one huge page, which suits the card's DMA, since it splits at every
  2 MiB page anyway. `rt_ip21_kv` and `rt_ip12_kv` already read the history where it lies.
- **Edits without moving the tail:** O(√n) to insert or remove mid-sequence, instead of moving everything after it.

What has to change from the published design:

- **Power-of-two tiers.** Its √n blocks index by division and modulo; integer division is slow on KNC, and shift
  and mask are branchless. Goodrich and Kloss size their levels this way too.
- **Plain appends gain nothing from the O(√n) mutation.** Plain inference only appends to the KV cache, so the
  gains above are the reservation and the blocks; the edits only pay if a client edits its context.

**Status: not built, so no measurements yet,** and it has to beat the fixed array on the engine's tok/s to
stay, like everything else in [measurements.md](measurements.md). No code from that repository is used here.

## One decode step, K=1

1. **Draft.** The MTP block (one attention layer) proposes the next token from the current token's
   embedding and hidden state. Its logits cover only the 32,768 commonest tokens.
2. **Check.** The trunk runs both the current and the drafted token as one batch, T=2. One pass over
   each weight matrix serves both tokens.
3. **Accept.** `Z←+/∧\DR=AG` in `qwen35.apl`: how many leading drafts the trunk agrees with.
4. **Roll back if needed.** State is indexed by position, so nothing is copied: the next batch simply
   starts from an earlier position. `S` keeps 6 levels and the ring 8, enough for what can be taken back.
5. **Catch up.** The MTP block is fed the true hidden states of the accepted positions.

Only tokens the trunk agrees with stand, so the output is the same at any K. That is gated at K=0, 1, 2.

## One layer, as a conversation

Each change of side is a crossing, so the compiler gathers rows to keep phases few. A 9B delta layer at
NB=1 has 4 card phases:

| Phase | Card | Host |
|---|---|---|
| 1 | queues 4 products, opens the round | norm of `x` |
| | reads ahead 128 KiB per thread while it waits | finishes, signals |
| | pulls `x`, runs `W_QKV W_GATE W_BETA W_ALPHA` | |
| | pushes 4 results | waits |
| 2 | | conv, delta heads on host threads, gating |
| | pulls, runs `W_OUT`, pushes | |
| 3 | | residual, norm |
| | pulls, runs `W_GATE2 W_UP`, pushes | |
| 4 | | `silu(g)×u` |
| | pulls, runs `W_DOWN`, pushes | |

Bulk moves by the card's DMA channel 3. The counters are words in host memory, read by the card through
`/dev/phihost` and written by it with the DMA's own status descriptor, so a host reader sees a counter
only after the data before it. Host to card and back: 64 B 5.03 us, 8 KiB 9.43 us, 24 KiB 18.44 us median.

## The prompt

The same `qwen35.apl` runs NB tokens at once, compiled for NB = 16, 8, 4, 3, 2 and 1. The prompt goes
through layer by layer in the largest batches that fit, and only the last token runs the output matrix.
A 40-token prompt is 16 + 16 + 8.

Inside a batch the matrix products are shared, but state is sequential: tokens take their turns on `S`,
because each reads the level the one before it wrote. The card runs 3 threads per core at NB of 4 or
more, where it is compute-bound, and parks the extra threads on a futex for decode, where it is
memory-bound and they would only compete.

## Threads

- **Card:** cores 0 to 58. 2 threads per core in decode, 3 in the prompt. Using all 60 cores measured 9 %
  slower.
- **Host:** 4 by default, CPUs 1 to 7 then their siblings 9 to 15, up to 14. Core 0's two threads, CPUs 0
  and 8, stay with the scheduler. On the 9B: 1 thread 14.855, 2 15.187, 4 15.433, 7 15.219, 14 13.508 tok/s: the delta regions are bound by host
  DRAM (see [measurements.md](measurements.md)).

## What this tree depends on

- The **Intel Phi 7120A stack** (`phi`, `phictl`, `phi-isa-audit`, `toolchain/env.sh`): boots and serves
  the card. Used as a tool, never patched.
- **Intel-Phi-LLVM**: the KNC clang for the VPU objects; `phi-isa-audit` checks every binary for
  instructions KNC does not have, and is a gate that code must satisfy, never override.
- **GNU APL 2.0**, fixed as in `tools/gnu-apl/`: hosts the compiler and the F64 reference.
- **llama.cpp**: the external reference, read-only and test only.
