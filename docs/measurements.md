# Measurements

The history of the engine's numbers, how they were taken, and what was tried and refused. Records are in
`logs/`; a name like `142457-fusion-9B.txt` is under `logs/engine/2026-11-01/`. Everything is the Xeon Phi 7120A +
Ryzen 7 5800X unless stated, greedy, 40-token prompt, GGUF Q8_0 models.

## How a number is taken

- **Alternating, on one machine.** Two builds or engines take turns in one window, repeated, and the spread is reported.
- **A warm-up repeat is dropped.** Rates are decode tok/s over generated tokens, load excluded.
- **Tokens are checked in every row.** A faster wrong answer is not a result.
- **Nothing else heavy on the host during a timed run.** Several results were thrown out because a stray `tar` or `strings` ran during them.
- **Arithmetic is labelled.** Estimates are replaced by measurements at the first opportunity.
- **The data wins.** When a result contradicts the expectation, the discrepancy is stated before any explanation.

## The numbers over one day, 9B

Decode and prompt, tok/s. llama.cpp in the morning: decode 3.87-3.89, prompt 24.5-28.4.

| Time | Change | Decode | Prompt | Record |
|---|---|---|---|---|
| 09:12 | card + host worker, one host thread | 7.755-8.279 | 7.7-8.3 | `091156-interleaved-baseline.txt` |
| 10:27 | threads, team rounds, tuned prefetch | 14.55-14.70 | 14.7-14.9 | `101905-final-interleaved.txt` |
| 11:22 | batched prompt | 14.58 | 31.2-31.4 | `111822-9B-batched.txt` |
| 11:4x | head-major delta map, parallel maps | 15.2-15.4 | 38.2-38.5 | `114722-head-major.txt`, `115236-par-heads.txt` |
| 12:2x | open rounds, read-ahead (64 tokens) | 15.80 mean | | `122232-open-rounds.txt` |
| 13:19 | speculation, K=1 | 16.44 | 34.1 | `131324-mtp-9B.txt` |
| 13:33 | `gemm` tile split | 17.19 | 32.1 | `132824-gemm-tiles-ab.txt` |
| 13:41 | queued rounds, read-ahead at T=2, draft head | 18.62-18.72 | 33.2 | `133719-mtp-9B-c.txt` |
| 14:05 | `S` write streamed for a batch's last token | 19.03 | 33.4 | `135901-stream-last-9B.txt` |
| 14:30 | fusion | 20.03-20.21 | 35.0 | `142457-fusion-9B.txt` |
| 14:54 | final tree against llama.cpp | **19.642 mean** | **34.310 mean** | `144406-final-interleaved-fusion.txt` |

Rows come from different windows, so compare within a row's record, not across rows.

Unexplained: the prompt was 38.2-38.5 at 11:4x and has been 33-36 since, and some rows differ within a session by 7 %.

## Where the time goes

9B, K=0, `PROFILE=1`, 64.669 ms/token, before fusion:

| What | ms/token | Share |
|---|---|---|
| `gemv` | 52.619 | 81.4 % |
| Waiting for the host | 8.534 | 13.2 % |
| DMA copies | 2.096 | 3.2 % |

The delta regions alone, timed inside `rt_par`, 128 tokens:

| Host threads | Inside the regions, ms/token | Whole token, ms |
|---|---|---|
| 4 | 5.073, 5.161, 5.175 | 64.4-64.7 |
| 1 | 6.100, 6.008, 6.037 | 64.5-65.1 |

Four times the threads for 15 %: 96 MiB of `S` per token through about 19.6 GB/s (arithmetic), while the lane arithmetic needs about
0.3 ms (arithmetic). These regions are DRAM-bound.

The card's decode is memory-bound: 8.43 GB per token against a STREAM of 177.9 GB/s, a ceiling of about 18.4 tok/s without
speculation (arithmetic). Speculation works because T=2 reads each matrix once for up to two tokens.

The prompt is not memory-bound: about 23 GB/s at T=16 (arithmetic), and about 30 % of the F32 peak of 2416.76 GFLOPS.

## Speculative decoding

| K | 9B tok/s | 9B accepted | 2B tok/s | 2B accepted | Record |
|---|---|---|---|---|---|
| 0 | 15.040, 15.494, 15.709 | | 53.737, 49.942 | | `142457-fusion-9B.txt`, `144209-fusion-2B.txt` |
| 1 | 20.029, 20.157, 20.209 | 56/71 | 64.414, 64.235 | 57/69 | same |
| 2 | 18.218, 18.465 (before fusion) | 72/108 | 65.945, 63.477 | 76/100 | `133719-mtp-9B-c.txt` |
| 3 | 17.647 (before fusion) | 80/139 | | | same |

Before the tile split, the draft head and fusion, speculation lost on the 2B (K=1 38.9-46.1 against K=0 48.7).

## Tried and refused

Measured, then removed or never built. Not to be proposed again without new evidence.

| Idea | Result | Record |
|---|---|---|
| Lossless compression of the weights | entropy 7.67-7.68 of 8 bits, zstd 1.040, lz4 1.000; activations zstd 1.079 | `120138-weight-compressibility.txt` |
| Low-rank weights | rank 1024 keeps 46.01-64.33 % of the energy, a random control 45.60 % | `123414`, `123520` spectrum |
| Reading tiles in a different order | a 10.5 % gain in a pure-read bench did not carry over; head 7.48-7.53 against 6.14-6.17 ms | `120844`, `121211` interleave |
| Warming `x` into L2 first | mid-layer 51.5-52.3 against 49.2-49.5 ms | `121707-x-prewarm.txt` |
| Prefetching `x` | T=16 633 to 567 GFLOPS, older kernel | `101153-gemm-pfx.txt` |
| Padding the rows of `x` | bench +9.5 % at T=16; engine K=1 -0.9 %, prompt +2.3 % overlapping | `145919-gemm-xpad.txt`, `150252-xpad-9B.txt` |
| Skipping silenced arms in the products | FFN `silu(g)` 16-row groups below 1e-2 of the max 0.00 %; FFN 32-col blocks 0.51 %; delta output 7.85 %, dropping them costing 8.2e-3 | `140909-silence-stats.txt` |
| A decay shortcut for `S` | `E` geomean 0.78754, 60.92 % of heads at or above 0.95 | same |
| `gemv` deleted for `gemm` at T=1 | K=0 15.433 to 14.787; K=1 tie | `143432-gemv-ablation-9B.txt` |
| More than 4 host threads | 9B: 1 thread 14.855, 2 15.187, 4 15.433, 7 15.219, 14 13.508 tok/s | `115236-par-heads.txt` |
| All 60 cores | 2B: 59 cores 45.810, 45.658; 60 cores 41.575, -9.1 % | `100243-levers3.txt` |
| EML as the arithmetic | about 100x the instructions (arithmetic, not run), on a memory-bound decode | below |

**Lesson of the padding row:** a bench's `x` stays resident in L2, while the engine's arrives fresh by DMA (likely the cause,
not verified). Layout ideas for `x` must be tested in the engine.

## The EML operator

arXiv:2603.21852 (Odrzywołek) shows `eml(x,y)=exp(x)-ln(y)` with the constant 1 generates every elementary function, a continuous
NAND. Tested as a source of optimizations for this engine:

| Idea | Verdict |
|---|---|
| EML as the arithmetic | Rejected. One FMA is at least 17 gates by the paper's Table 4 (x×y K=17, x+y K=19, gates = (K-1)/2), each about 6 KNC instructions (arithmetic). Negative weights need complex lanes, and Sect. 4.1 says compiled EML fails at isolated points, especially zero. Decode reads 8.43 GB per token whatever the arithmetic. |
| Silence an arm with a constant | Measured dead on the card's products (table above) |
| Cancel exp against ln | No lever: `E` is at most 1,024 lanes per token (arithmetic) |
| One uniform node | Became the fusion pass, generally: `fz.apl`. Stored lanes -78.4 % per delta layer, K=1 +2.5 %, prompt +3.5 % |
| Remove a button if the rest rebuilds it | Deleted `rt_ew` and its 22 cases, six `rt_s_*`, `rt_redfirst`, `rt_outer_add`, `rt_scale0n`, `rt_transpose102` from the host. Refused `gemv`. |
| Witness checks | `tests/apl/delta_witness.sh` |

The transfer is thin: these are general engineering rules the paper happens to illustrate. Most of the day's gain came from
speculation, not from this.

## What is left

| Lever | Estimate | Status |
|---|---|---|
| Find what stalls `tile8`, for the prompt at about 30 % of peak | unknown | next |
| Defer `S`'s writes: `S_base` plus pending `N`,`K`, flushed every M tokens, rollback a count | 9B about 21.3 tok/s (estimate); DRAM -44 % at M=16, lane-ops about +10 % | unbuilt |
| ECC off | about +7 % decode (arithmetic) | a flash operation through Intel's signed module, not recommended |
| 4-bit FFN weights | the only remaining byte cut | lossy |
| KV cache as an [Accelerated Square Array](https://github.com/gtgunar/Accelerated-square-Array) | contexts past `CTX` with no fixed reservation; O(√n) edits | unbuilt, see [architecture.md](architecture.md#the-kv-cache-and-the-accelerated-square-array) |
