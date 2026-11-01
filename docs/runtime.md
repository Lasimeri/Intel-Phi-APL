# Kernels and runtime

The C and assembly-level layer under the compiled APL: primitives, transport and I/O, no forward pass.

## Form I tiles

Q8_0 weights are repacked at load, in place. Rows go in groups of 16; per group, per 32-column block,
one 544 B tile:

```
512 B   int8: for each of the 32 columns, that column's 16 weights
 32 B   the 16 rows' f16 block scales
```

1.0625 B per weight, as in the GGUF. The product is the accumulation `+⌿` of weight columns scaled by x:

```
y16 += scales x sum_i (weights_i x x_i)     one vfmadd231ps per 16 weights
```

The weight enters as an up-converting memory operand (`{sint8}`), the scale as `{float16}`, `x` as `{1to16}`
or by register swizzle. Every sum and product is F32 in a zmm register. `Form R` (row-dot) is kept in the
benchmark as the alternative it beat at every setting.

## The two products

| | `gemv` (`src/card/gemv.c`) | `gemm` (`src/card/gemm.c`) |
|---|---|---|
| APL | `W+.×x` | `X+.×⍉W`, T tokens as rows |
| Used for | T=1: decode at K=0, drafts | T>=2: speculative checks, prompt |
| Cost per token | whole matrix | matrix ÷ T of memory, same arithmetic |

`Y[t]` is bit-identical to `gemv` of `X[t]`, gated. `gemm` was tried in place of `gemv` at T=1 and lost 4.2 % at K=0,
so both stay.

**Tile kernels.** `tile8`, `tile4`, `tile3` keep four columns of `x` in registers (`vbroadcastf32x4`) and reach them by
swizzle, because a memory broadcast per FMA costs an operand. `tile2` and `tile1` use memory operands. Inside a 544 B
tile, partial sums stay in registers across its 32 columns while the tile stays in L1.

**Splitting between threads.** With at least 8 groups a thread, whole groups go to threads. Below that the imbalance
is large (2.17 groups a thread is 38 % off), so tiles are split, with a chain carrying a group's partial sums from the
lower thread to the next. `gemm` does this only up to `TILE_MAX_T` (4): it gained K=1 decode 6.5 % and cost T=16 3.4 %.

**Passes.** At most `GEMM_PASS_T` (16) tokens go over the weights at once; above that `x` fights the weights for L2.

## Rounds

- `rt_gemv_q`, `rt_gemm_q` queue a product. `rt_gemv_go` runs the round: a thread does its share of each with no barrier
  between.
- `rt_gemv_open` starts the round before the data has arrived. Threads read ahead up to 128 KiB each
  (`gemv_i_prefill`, `gemm_i_prefill`) and stop at once when the round is ready. 128 KiB measured best; this took 9B
  decode from a 64-token mean of 15.073 to 15.798.
- `team_resize` parks extra threads on a futex between prompt and decode.

## Team

`src/card/team.c`: pinned threads that run one function together, with a tree fan-out of 4. Workers spin between jobs,
because a token's many short steps cannot afford to sleep. A round costs about 20 us, 25,000 cycles.

## The link

`src/card/link.h`. Frames are mirrored; a crossing is "copy this range".

| Command | Card does |
|---|---|
| `LINK_RESET` | clears state for a new sequence |
| `LINK_EMBED` | one embedding row |
| `LINK_STEP` | one layer, NB=1 |
| `LINK_BATCH` | one layer, the batch size's index in the command |
| `LINK_MTP` | the draft layer |
| `LINK_LOGITS` | the output matrix, for a batch |
| `LINK_DRAFT_LOGITS` | the draft's output matrix |
| `LINK_MARK` | the prompt has ended: resize the team, then acknowledge |
| `LINK_END` | stop |

**The mailbox is one slot.** A command with no card phase must be acknowledged before the next, or the next
overwrites it. Forgetting this once hung MTP: `LINK_MARK` followed by `LINK_LOGITS`.

## Host primitives

`src/host/rt_host.c`: AVX2 + FMA over F32. Most elementwise work is now generated loops, using `src/host/vec8.h`:

- `exp8`: `2^f` by its series to degree 6 on [-0.5, 0.5], clamped, about 2 ULP by its truncation error.
- `log8`: `ln m` by atanh's series to s^9 for |s| <= 0.172, about 2 ULP.

What remains are products over the host's own arrays:

| Primitive | Does |
|---|---|
| `rt_ip21` | `S+.×x`, 8 rows fused |
| `rt_scale_outer` | `(g×s)+a∘.×b` in one pass; streams when `R->last` |
| `rt_ip21_kv`, `rt_ip12_kv` | attention history products, reading the KV cache where it lies |
| `rt_index3`, `rt_cat`, `rt_sum`, `rt_maxv`, `rt_index_of` | structure |
| `rt_par`, `rt_fz` | a region or a chain on the host's team |

`R->last` is set by the token map: the last token of a batch streams its `S` write, because nothing reads it before the
next batch. Streaming every token cost the prompt 9.5 %, since a later token reads what an earlier one wrote.

## Settings

Run time. Sweeps only, unless noted: the defaults are what measured best.

| Variable | Default | Meaning |
|---|---|---|
| `APL_MTP` | 1 | drafts per pass, 0 to 3; 0 is no speculation |
| `APL_PROMPT_TPC` | 3 | card threads per core at NB of 4 or more |
| `APL_PROMPT_NB` | largest compiled | test only: largest batch in the prompt |
| `APL_HOST_PAR` | 131072 | lanes from which a host primitive uses the team |
| `APL_PREFILL` | 131072 | bytes read ahead per thread in an opened round |
| `APL_PF_L2`, `APL_PF_L1` | 4096, 2048 | `gemv` prefetch distance, bytes |
| `APL_BIG_L2`, `APL_BIG_L1` | 32768, 3072 | the same, for 100,000 rows or more |
| `APL_GEMM_L2`, `APL_GEMM_L1` | 4096, 2048 | `gemm` prefetch distance, bytes |
| `APL_PAR_ROWS` | 512 | rows from which `gemv` uses the team |
| `APL_NO_TILES` | unset | whole groups only |
| `APL_CARD_ENV` | unset | scripts: settings passed to the card program |
| `PROFILE=1` | unset | scripts: run the profile build and print where a token's time goes |

`gemv_bench` only: `APL_XPAD` (padding between rows of `x`), `APL_PASS`, `APL_TILES`.

Build: see the head of `config.mk` and `scripts/lib.sh`. `PHI_TREE`, `PHI_LLVM_BIN`, `PHI_SYSROOT_7120A`, `PHI_CARD`,
`HOST_CC`, `LLAMA_DIR`, `APL_BIN`, `APL_CL_INCLUDE` can all be overridden on the command line or from the environment. Their
defaults are the author's machine.

## Keeping out of each other's way

Other programs may share the card. DMA channels: 0 and 1 `phictl`, 2 and 5 Cortex, 3 this engine, 4 vgd, 6 pwgd, 7 the AVX-512
wrapper. Shared region: this engine uses [768, 1024) MiB.
